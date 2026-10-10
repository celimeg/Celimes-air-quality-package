"""TEMPO NO2 gap filling, tropospheric and stratospheric columns filled independently."""

import copy
import warnings

import numpy as np
import pandas as pd

from ..utils import append_table, parse_datetime, to_utc_naive
from ._core import cosine_similarity, fill_hour
from ._hours import LRU, HourIndex, last_written_datetime

HOUR_BIN_SIZE = 3      # 3-hour bins: 8 per day
MIN_STD = 1e-6

# Per-column settings from TEMPO_NO2_GAP_FILL.ipynb
NO2_FIELD_CONFIGS = {
    "troposphere": {
        "label": "Tropospheric",
        "column": "vertical_column_troposphere",
        "sim_features": ["u_wind_850mb_ms", "v_wind_850mb_ms", "wspd850",
                         "u_wind_250mb_ms", "v_wind_250mb_ms", "wspd250",
                         "temp_2m_K", "rh_2m_pct", "pbl_height_m",
                         "eff_cloud_fraction", "shear_0_1km"],
        "feature_weights": {"u_wind_850mb_ms": 2.0, "v_wind_850mb_ms": 2.0, "wspd850": 2.0,
                            "u_wind_250mb_ms": 1.0, "v_wind_250mb_ms": 1.0, "wspd250": 1.0,
                            "pbl_height_m": 2.0, "eff_cloud_fraction": 1.5,
                            "shear_0_1km": 1.0, "temp_2m_K": 0.75, "rh_2m_pct": 0.5},
        "max_context_hours": 1,
        "use_pbl_driver": True,
        "wind_u_features": ["u_wind_850mb_ms", "u_wind_250mb_ms"],
        "wind_v_features": ["v_wind_850mb_ms", "v_wind_250mb_ms"],
    },
    "stratosphere": {
        "label": "Stratospheric",
        "column": "vertical_column_stratosphere",
        "sim_features": ["u_wind_250mb_ms", "v_wind_250mb_ms", "wspd250", "eff_cloud_fraction"],
        "feature_weights": {"u_wind_250mb_ms": 2.0, "v_wind_250mb_ms": 2.0, "wspd250": 2.0,
                            "eff_cloud_fraction": 0.75},
        "max_context_hours": 2,
        "use_pbl_driver": False,
        "wind_u_features": ["u_wind_250mb_ms"],
        "wind_v_features": ["v_wind_250mb_ms"],
    },
}

# Derived features: name -> (u column, v column)
_DERIVED = {"wspd850": ("u_wind_850mb_ms", "v_wind_850mb_ms"),
            "wspd250": ("u_wind_250mb_ms", "v_wind_250mb_ms"),
            "shear_0_1km": ("shear_u_0_1km_per_s", "shear_v_0_1km_per_s")}
_HRRR_STAT_FEATURES = ["u_wind_850mb_ms", "v_wind_850mb_ms", "wspd850",
                       "u_wind_250mb_ms", "v_wind_250mb_ms", "wspd250",
                       "temp_2m_K", "rh_2m_pct", "pbl_height_m", "shear_0_1km"]


def get_season(month):
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA"}.get(month, "SON")


def _add_derived(df):
    df = df.copy()
    for name, (u, v) in _DERIVED.items():
        if u in df and v in df and name not in df:
            df[name] = np.hypot(df[u], df[v])
    return df


def compute_group_stats(hrrr_df, merged_df, hour_bin_size=HOUR_BIN_SIZE):
    """Mean/std of each similarity feature per (season, 3-hour bin).

    HRRR features come from the full HRRR table (all hours, all pixels);
    eff_cloud_fraction comes from the merged TEMPO table. Also returns
    whole-dataset stats used when a group's std is missing or ~0.
    """
    h = _add_derived(hrrr_df)
    dt = parse_datetime(h["Datetime"])
    h["Season"], h["HourBin"] = dt.dt.month.map(get_season), dt.dt.hour // hour_bin_size
    m = merged_df.copy()
    dtm = parse_datetime(m["Datetime"])
    m["Season"], m["HourBin"] = dtm.dt.month.map(get_season), dtm.dt.hour // hour_bin_size

    feats = [f for f in _HRRR_STAT_FEATURES if f in h]
    group_stats = {}
    g = h.groupby(["Season", "HourBin"])
    means, stds = g[feats].mean(), g[feats].std()
    for key in means.index:
        group_stats[key] = {f: (means.loc[key, f], stds.loc[key, f]) for f in feats}
    if "eff_cloud_fraction" in m:
        gc = m.groupby(["Season", "HourBin"])["eff_cloud_fraction"]
        mc, sc = gc.mean(), gc.std()
        for key in mc.index:
            group_stats.setdefault(key, {})["eff_cloud_fraction"] = (mc.loc[key], sc.loc[key])
    global_stats = {f: (h[f].mean(), h[f].std()) for f in feats}
    if "eff_cloud_fraction" in m:
        global_stats["eff_cloud_fraction"] = (m["eff_cloud_fraction"].mean(), m["eff_cloud_fraction"].std())
    return group_stats, global_stats


def _zscore_grid(grid, feature, season, hour_bin, group_stats, global_stats):
    mean, std = group_stats.get((season, hour_bin), {}).get(feature, (np.nan, np.nan))
    if pd.isna(std) or std < MIN_STD:
        mean, std = global_stats[feature]
    if pd.isna(std) or std < MIN_STD:
        return np.full(grid.shape, np.nan)
    return (grid - mean) / std


def tempo_no2_gap_filling(
    data, output_path, start, end, hrrr_data=None,
    fields=("troposphere", "stratosphere"), field_configs=None,
    temporal_weight=0.20, temporal_decay=0.5, fill_flag_penalty=0.5,
    fill_flag=200, negative_fill_flag=201, min_neighbors=2, max_window=9,
    hour_bin_size=HOUR_BIN_SIZE, qc_column="main_data_quality_flag",
):
    """Fill gaps in hourly TEMPO NO2, each column (trop/strat) independently.

    A pixel is a valid anchor when main_data_quality_flag == 0 and the
    column is >= 0; everything else is filled. Meteorological similarity
    uses z-scores per (season, 3-hour bin).

    Parameters
    ----------
    data : DataFrame
        Regridded TEMPO merged with HRRR (see io.combine_tempo_hrrr).
    output_path : str
        CSV (or .h5) appended hour by hour; resumes after the last hour.
    start, end : str or Timestamp
        Inclusive; a date-only end includes its 23:00 hour.
    hrrr_data : DataFrame, optional
        The full HRRR table for this site, used for the z-score statistics
        (as in the notebook). If omitted, the merged table is used.
    fields : which of NO2_FIELD_CONFIGS to fill.
    field_configs : dict, optional
        Override NO2_FIELD_CONFIGS entries (same keys).

    Output columns added, per field
    -------------------------------
    filled_<column>       gap-filled values
    <column>_qc_filled    QC flag, replaced by 200 where filled, 201 where the
                          original pixel had QC 0 but a negative value
    """
    cfgs = copy.deepcopy(NO2_FIELD_CONFIGS)
    if field_configs:
        for k, v in field_configs.items():
            cfgs.setdefault(k, {}).update(v)
    cfgs = {k: cfgs[k] for k in fields}

    data = _add_derived(data)
    hi = HourIndex(data)
    group_stats, global_stats = compute_group_stats(hrrr_data if hrrr_data is not None else data,
                                                    data, hour_bin_size)
    met_cols = sorted({f for c in cfgs.values() for f in c["sim_features"] + c["wind_u_features"]
                       + c["wind_v_features"]} | {"pbl_height_m"})
    missing = [c for c in met_cols + [qc_column] + [c["column"] for c in cfgs.values()] if c not in hi.df]
    if missing:
        raise KeyError(f"data is missing columns: {missing}")

    start_ts, end_ts = to_utc_naive(start), to_utc_naive(end)
    if end_ts == end_ts.normalize():
        end_ts += pd.Timedelta(hours=23)
    last = last_written_datetime(output_path) if output_path.lower().endswith(".csv") else None
    if last is not None:
        print(f"Resuming after {last}")
        start_ts = max(start_ts, last + pd.Timedelta(hours=1))

    cache = LRU(maxsize=8)

    def load(ts, column):
        idx = hi.rows(ts)
        if len(idx) == 0:
            return None
        qc = hi.df[qc_column].to_numpy(dtype=float)[idx]
        val = hi.df[column].to_numpy(dtype=float)[idx]
        good = ~np.isnan(qc) & (qc == 0) & ~np.isnan(val) & (val >= 0)
        g = {"value": hi.grid(idx, None, np.where(good, val, np.nan)),
             "qc": hi.grid(idx, qc_column),
             "neg": hi.grid(idx, None, (~np.isnan(qc) & (qc == 0) & ~np.isnan(val) & (val < 0)).astype(float))}
        if np.all(np.isnan(g["value"])):
            return None
        for c in met_cols:
            g[c] = hi.grid(idx, c)
        g["season"], g["hour_bin"] = get_season(ts.month), ts.hour // hour_bin_size
        return g

    def features(g, cfg):
        return np.stack([_zscore_grid(g[f], f, g["season"], g["hour_bin"], group_stats, global_stats)
                         for f in cfg["sim_features"]])

    n_done = 0
    for ts in pd.date_range(start_ts, end_ts, freq="h"):
        idx = hi.rows(ts)
        if len(idx) == 0:
            continue
        df_out = hi.df.iloc[idx].copy()
        r, c = hi.row[idx], hi.col[idx]
        for key, cfg in cfgs.items():
            col = cfg["column"]
            df_out[f"filled_{col}"] = np.nan
            df_out[f"{col}_qc_filled"] = np.nan
            g = cache.get_or((ts, col), lambda: load(ts, col))
            if g is None:
                continue
            weights = [cfg["feature_weights"][f] for f in cfg["sim_features"]]
            f_cur = features(g, cfg)
            priors = []
            for k in range(1, cfg["max_context_hours"] + 1):
                t_ctx = ts - pd.Timedelta(hours=k)
                g_ctx = cache.get_or((t_ctx, col), lambda t=t_ctx: load(t, col))
                if g_ctx is None:
                    continue
                sim = cosine_similarity(f_cur, features(g_ctx, cfg), weights)
                w_t = temporal_weight * sim * temporal_decay ** (k - 1)
                with np.errstate(invalid="ignore"):
                    was_gap = (g_ctx["qc"] != 0) | np.isnan(g_ctx["qc"]) | (g_ctx["value"] < 0)
                priors.append((np.where(was_gap, w_t * fill_flag_penalty, w_t), g_ctx["value"]))

            raw_mask = ~np.isnan(g["value"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                u = np.nanmean([g[f] for f in cfg["wind_u_features"]], axis=0)
                v = np.nanmean([g[f] for f in cfg["wind_v_features"]], axis=0)
            pbl = g["pbl_height_m"] if cfg["use_pbl_driver"] else None
            filled = fill_hour(g["value"], ~raw_mask, u, v, pbl, priors, min_neighbors, max_window,
                               abs_values=True)
            changed = ~np.isclose(filled, g["value"], equal_nan=True)
            flag = np.where(changed, np.where(g["neg"] == 1, negative_fill_flag, fill_flag), g["qc"])
            df_out[f"filled_{col}"] = filled[r, c]
            df_out[f"{col}_qc_filled"] = flag[r, c]

        append_table(df_out, output_path, key="no2_filled")
        n_done += 1
        if ts.hour == 0:
            print(f"{ts.date()} ...")
    print(f"Done: {n_done} hours processed -> {output_path}")
