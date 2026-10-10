"""GOES ABI AOD gap filling (HRRR meteorology by default)."""


import numpy as np
import pandas as pd

from ..utils import append_table, haversine_km, to_utc_naive
from ._core import cosine_similarity, fill_hour
from ._hours import LRU, HourIndex, last_written_datetime

# Defaults from GOES_AOD_GAP_FILL.ipynb (HRRR version)
AOD_HRRR_SIM_FEATURES = ["u_wind_850mb_ms", "v_wind_850mb_ms", "u_wind_250mb_ms", "v_wind_250mb_ms",
                         "temp_2m_K", "rh_2m_pct", "pbl_height_m"]
AOD_HRRR_SPEED_FEATURES = [("wspd850", "u_wind_850mb_ms", "v_wind_850mb_ms"),
                           ("wspd250", "u_wind_250mb_ms", "v_wind_250mb_ms")]
AOD_HRRR_FEATURE_WEIGHTS = {"u_wind_850mb_ms": 2.0, "v_wind_850mb_ms": 2.0,
                            "u_wind_250mb_ms": 1.0, "v_wind_250mb_ms": 1.0,
                            "temp_2m_K": 0.5, "rh_2m_pct": 0.5, "pbl_height_m": 1.75,
                            "wspd850": 2.0, "wspd250": 0.5}

# The original package's ERA5 configuration, for reproducing older runs
AOD_ERA5_CONFIG = dict(
    sim_features=["u10", "v10", "u100", "v100", "blh", "Temperature", "Humidity"],
    speed_features=[("wspd10", "u10", "v10"), ("wspd100", "u100", "v100")],
    feature_weights={"u10": 2.0, "v10": 2.0, "u100": 1.5, "v100": 1.5, "blh": 2.0,
                     "Temperature": 0.5, "Humidity": 0.5, "wspd10": 2.0, "wspd100": 1.5},
    wind_u_features=["u100"], wind_v_features=["v100"], pbl_feature="blh",
)

AOD_OUTPUT_COLUMNS = ["Pixel Number", "Latitude", "Longitude", "Datetime",
                      "Weighted_AOD", "AOD_DQF0", "Weighted_DQF", "avg_DQF", "pct_DQF0",
                      "filled_AOD", "pct_DQF0_filled"]


def goes_aod_gap_filling(
    data, output_path, start, end,
    pct_dqf0_thresh=10, max_context_hours=2, sza_threshold=70.0,
    temporal_weight=0.20, temporal_decay=0.5, fill_flag_penalty=0.5, fill_flag=200,
    sim_features=None, speed_features=None, feature_weights=None,
    wind_u_features=("u_wind_850mb_ms", "u_wind_250mb_ms"),
    wind_v_features=("v_wind_850mb_ms", "v_wind_250mb_ms"),
    pbl_feature="pbl_height_m",
    min_neighbors=2, max_window=9,
    target_point=None, save_radius_km=None,
    output_columns="default",
):
    """Fill gaps in hourly GOES ABI AOD using spatial structure, meteorology and prior hours.

    Parameters
    ----------
    data : DataFrame
        Regridded AOD merged with meteorology (see io.combine_goes_hrrr):
        Pixel Number, Latitude, Longitude, Datetime, Weighted_AOD, pct_DQF0,
        plus the met columns named in sim_features / wind / pbl settings.
    output_path : str
        CSV (or .h5) appended hour by hour. A rerun resumes after the last
        hour written.
    start, end : str or Timestamp
        Inclusive. A date-only end includes its 23:00 hour.
    pct_dqf0_thresh : float
        Pixels with pct_DQF0 below this are treated as gaps.
    sza_threshold : float
        Hours (and context hours) with solar zenith above this at the domain
        centre are written without filling.
    max_context_hours, temporal_weight, temporal_decay, fill_flag_penalty
        Temporal prior settings: H-1..H-n, max weight, decay per extra hour,
        and the penalty for a context pixel that was itself a gap.
    sim_features, speed_features, feature_weights
        Meteorological state vector for the cosine similarity. Defaults are
        the HRRR set (850/250 mb winds, 2 m T and RH, PBL, plus 850/250 mb
        wind speed). For the original ERA5 setup pass **AOD_ERA5_CONFIG.
    wind_u_features, wind_v_features, pbl_feature
        Wind (averaged across the listed levels) and PBL used to weight
        neighbours in the update step.
    target_point, save_radius_km
        If both given, only pixels within save_radius_km of (lat, lon) are
        written (all pixels are still used for filling).
    output_columns : "default", "all", or list
        "default" writes the AOD columns only; "all" also writes the met columns.

    Output columns added
    --------------------
    filled_AOD       gap-filled AOD (original value where it was valid)
    pct_DQF0_filled  pct_DQF0, replaced by fill_flag (200) where the value changed
    """
    sim_features = list(sim_features or AOD_HRRR_SIM_FEATURES)
    speed_features = list(speed_features if speed_features is not None else AOD_HRRR_SPEED_FEATURES)
    feature_weights = dict(feature_weights or AOD_HRRR_FEATURE_WEIGHTS)
    weight_vec = [feature_weights[f] for f in sim_features] + [feature_weights[n] for n, _, _ in speed_features]

    hi = HourIndex(data)
    met_cols = sorted(set(sim_features) | {c for _, u, v in speed_features for c in (u, v)}
                      | set(wind_u_features) | set(wind_v_features) | ({pbl_feature} if pbl_feature else set()))
    missing = [c for c in met_cols + ["Weighted_AOD", "pct_DQF0"] if c not in hi.df]
    if missing:
        raise KeyError(f"data is missing columns: {missing}")

    start_ts = to_utc_naive(start)
    end_ts = to_utc_naive(end)
    if end_ts == end_ts.normalize():
        end_ts += pd.Timedelta(hours=23)
    last = last_written_datetime(output_path) if output_path.lower().endswith(".csv") else None
    if last is not None:
        print(f"Resuming after {last}")
        start_ts = max(start_ts, last + pd.Timedelta(hours=1))

    cache = LRU(maxsize=max_context_hours + 2)

    def load(ts, center):
        """Grids for one hour, or None (SZA too high, no rows, or no valid AOD)."""
        idx = hi.rows(ts)
        if len(idx) == 0 or hi.sza(ts, *center) > sza_threshold:
            return None
        pct = hi.df["pct_DQF0"].to_numpy(dtype=float)[idx]
        aod = hi.df["Weighted_AOD"].to_numpy(dtype=float)[idx].copy()
        aod[pct < pct_dqf0_thresh] = np.nan
        g = {"aod": hi.grid(idx, None, aod), "pct_dqf0": hi.grid(idx, "pct_DQF0")}
        if np.all(np.isnan(g["aod"])):
            return None
        for c in met_cols:
            g[c] = hi.grid(idx, c)
        feats = [g[f] for f in sim_features] + [np.hypot(g[u], g[v]) for _, u, v in speed_features]
        g["features"] = np.stack(feats)
        return g

    cols_out = None
    n_done = 0
    for ts in pd.date_range(start_ts, end_ts, freq="h"):
        idx = hi.rows(ts)
        if len(idx) == 0:
            continue
        df_raw = hi.df.iloc[idx].copy()
        df_raw["filled_AOD"] = np.nan
        df_raw["pct_DQF0_filled"] = np.nan
        center = (float(df_raw["Latitude"].mean()), float(df_raw["Longitude"].mean()))

        g = cache.get_or((ts, center), lambda: load(ts, center))
        if g is not None:
            context = []
            for k in range(1, max_context_hours + 1):
                t_ctx = ts - pd.Timedelta(hours=k)
                g_ctx = cache.get_or((t_ctx, center), lambda t=t_ctx: load(t, center))
                if g_ctx is not None:
                    context.append((k, g_ctx))

            priors = []
            for k, g_ctx in context:
                sim = cosine_similarity(g["features"], g_ctx["features"], weight_vec)
                w_t = temporal_weight * sim * temporal_decay ** (k - 1)
                was_gap = (g_ctx["pct_dqf0"] < pct_dqf0_thresh) | np.isnan(g_ctx["pct_dqf0"])
                priors.append((np.where(was_gap, w_t * fill_flag_penalty, w_t), g_ctx["aod"]))

            adjustable = (g["pct_dqf0"] < pct_dqf0_thresh) | np.isnan(g["pct_dqf0"])
            u = np.mean([g[c] for c in wind_u_features], axis=0)
            v = np.mean([g[c] for c in wind_v_features], axis=0)
            filled = fill_hour(g["aod"], adjustable, u, v, g[pbl_feature] if pbl_feature else None,
                               priors, min_neighbors, max_window)
            changed = ~np.isclose(filled, g["aod"], equal_nan=True)
            flag = np.where(changed, fill_flag, g["pct_dqf0"])
            r, c = hi.row[idx], hi.col[idx]
            df_raw["filled_AOD"] = filled[r, c]
            df_raw["pct_DQF0_filled"] = flag[r, c]

        if target_point is not None and save_radius_km is not None:
            d = haversine_km(df_raw["Latitude"].to_numpy(), df_raw["Longitude"].to_numpy(), *target_point)
            df_raw = df_raw[d <= save_radius_km]
        if cols_out is None:
            if output_columns == "default":
                cols_out = [c for c in AOD_OUTPUT_COLUMNS if c in df_raw]
            elif output_columns == "all":
                cols_out = list(df_raw.columns)
            else:
                cols_out = list(output_columns)
        if not df_raw.empty:
            append_table(df_raw[cols_out], output_path, key="aod_filled")
        n_done += 1
        if ts.hour == 0:
            print(f"{ts.date()} ...")
    print(f"Done: {n_done} hours processed -> {output_path}")
