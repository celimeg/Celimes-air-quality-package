# Verbatim from TEMPO_NO2_GAP_FILL.ipynb (stats cell, single-hour helper cell, full-run cell),
# without the plotting/run sections. Reference implementation for tests.
# Expects df, hrrr_df, tempo, group_stats, global_stats, output_path in globals.
import numpy as np
import pandas as pd

ALL_MET_FEATURES = [
    "u_wind_850mb_ms", "v_wind_850mb_ms", "wspd850",
    "u_wind_250mb_ms", "v_wind_250mb_ms", "wspd250",
    "temp_2m_K", "rh_2m_pct", "pbl_height_m",
    "eff_cloud_fraction", "shear_0_1km",
]

HRRR_STAT_FEATURES = [f for f in ALL_MET_FEATURES if f != "eff_cloud_fraction"]

MIN_STD = 1e-6
HOUR_BIN_SIZE = 3  # 3-hour bins -> 8 bins/day, captures diurnal cycle without per-hour sparsity


def get_season(month):
    if month in (12, 1, 2):
        return "DJF"
    elif month in (3, 4, 5):
        return "MAM"
    elif month in (6, 7, 8):
        return "JJA"
    else:
        return "SON"


def compute_group_stats(hrrr_df, merged_df, hour_bin_size=HOUR_BIN_SIZE):
    """
    Mean/std per feature, grouped by (Season, HourBin) -- captures both
    the seasonal cycle (PBL/wind climatology shifts month to month) and
    the diurnal cycle (PBL depth, wind, cloud cover all vary by time of
    day) without the per-pixel-per-day cost that made this step slow.
    """
    hrrr = hrrr_df.copy()
    hrrr["wspd850"]     = np.hypot(hrrr["u_wind_850mb_ms"], hrrr["v_wind_850mb_ms"])
    hrrr["wspd250"]     = np.hypot(hrrr["u_wind_250mb_ms"], hrrr["v_wind_250mb_ms"])
    hrrr["shear_0_1km"] = np.hypot(hrrr["shear_u_0_1km_per_s"], hrrr["shear_v_0_1km_per_s"])

    dt = pd.to_datetime(hrrr["Datetime"])
    hrrr["Season"]  = dt.dt.month.map(get_season)
    hrrr["HourBin"] = dt.dt.hour // hour_bin_size

    merged = merged_df.copy()
    dt_m = pd.to_datetime(merged["Datetime"])
    merged["Season"]  = dt_m.dt.month.map(get_season)
    merged["HourBin"] = dt_m.dt.hour // hour_bin_size

    group_stats = {}

    grouped = hrrr.groupby(["Season", "HourBin"])
    means = grouped[HRRR_STAT_FEATURES].mean()
    stds  = grouped[HRRR_STAT_FEATURES].std()
    for key in means.index:
        entry = group_stats.setdefault(key, {})
        for feat in HRRR_STAT_FEATURES:
            entry[feat] = (means.loc[key, feat], stds.loc[key, feat])

    grouped_cloud = merged.groupby(["Season", "HourBin"])["eff_cloud_fraction"]
    m_cloud, s_cloud = grouped_cloud.mean(), grouped_cloud.std()
    for key in m_cloud.index:
        group_stats.setdefault(key, {})["eff_cloud_fraction"] = (m_cloud.loc[key], s_cloud.loc[key])

    # Fallback for a (season, hour_bin) group with too few samples for a
    # meaningful std (rare, but possible for cloud fraction in sparse bins)
    global_stats = {feat: (hrrr[feat].mean(), hrrr[feat].std()) for feat in HRRR_STAT_FEATURES}
    global_stats["eff_cloud_fraction"] = (merged["eff_cloud_fraction"].mean(), merged["eff_cloud_fraction"].std())

    return group_stats, global_stats


def zscore(value, feature, season, hour_bin, group_stats, global_stats):
    mean, std = group_stats.get((season, hour_bin), {}).get(feature, (np.nan, np.nan))
    if pd.isna(std) or std < MIN_STD:
        mean, std = global_stats[feature]
    if pd.isna(value) or pd.isna(std) or std < MIN_STD:
        return np.nan
    return (value - mean) / std



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from scipy.ndimage import median_filter, prewitt

# ============================================================
# USER PARAMETERS
# NOTE: this chunk assumes the statistics chunk has already been run
# and ALL_MET_FEATURES, HOUR_BIN_SIZE, get_season, zscore, group_stats,
# global_stats already exist in the namespace.
# ============================================================
raw_datetime    = "1/3/2024 19:00"
FILL_FLAG_VALUE = 200        # output QC flag for filled pixels

TEMPORAL_WEIGHT   = 0.20
TEMPORAL_DECAY    = 0.5
FILL_FLAG_PENALTY = 0.5

MIN_NEIGHBORS = 2
MAX_WINDOW    = 9

# ============================================================
# PER-FIELD CONFIG
# ============================================================
FIELD_CONFIGS = {
    "troposphere": {
        "label": "Tropospheric",
        "column": "vertical_column_troposphere",
        "sim_features": [
            "u_wind_850mb_ms", "v_wind_850mb_ms", "wspd850",
            "u_wind_250mb_ms", "v_wind_250mb_ms", "wspd250",
            "temp_2m_K", "rh_2m_pct", "pbl_height_m",
            "eff_cloud_fraction", "shear_0_1km",
        ],
        "feature_weights": {
            "u_wind_850mb_ms": 2.0, "v_wind_850mb_ms": 2.0, "wspd850": 2.0,
            "u_wind_250mb_ms": 1.0, "v_wind_250mb_ms": 1.0, "wspd250": 1.0,
            "pbl_height_m": 2.0, "eff_cloud_fraction": 1.5,
            "shear_0_1km": 1.0, "temp_2m_K": 0.75, "rh_2m_pct": 0.5,
        },
        "max_context_hours": 1,
        "use_pbl_driver": True,
        "wind_u_features": ["u_wind_850mb_ms", "u_wind_250mb_ms"],  # blended for 4b
        "wind_v_features": ["v_wind_850mb_ms", "v_wind_250mb_ms"],
    },
    "stratosphere": {
        "label": "Stratospheric",
        "column": "vertical_column_stratosphere",
        "sim_features": ["u_wind_250mb_ms", "v_wind_250mb_ms", "wspd250", "eff_cloud_fraction"],
        "feature_weights": {
            "u_wind_250mb_ms": 2.0, "v_wind_250mb_ms": 2.0, "wspd250": 2.0,
            "eff_cloud_fraction": 0.75,
        },
        "max_context_hours": 2,
        "use_pbl_driver": False,          # no PBL equivalent aloft -- wind-only driver in 4b
        "wind_u_features": ["u_wind_250mb_ms"],
        "wind_v_features": ["v_wind_250mb_ms"],
    },
}

# ============================================================
# PIXEL GEOMETRY -- from the footprint grid spacing itself
# ============================================================
lat_res  = np.abs(np.diff(np.sort(tempo["Latitude"].unique()))).mean()
lon_res  = np.abs(np.diff(np.sort(tempo["Longitude"].unique()))).mean()
half_lat = lat_res / 2
half_lon = lon_res / 2
pixel_size = (lat_res + lon_res) / 2

# ============================================================
# HELPER: IS THIS VALUE A TRUSTWORTHY ANCHOR?
# qc==0 alone isn't enough -- a flag-0 pixel can still carry a
# negative retrieved column near the detection limit. Negative
# values are excluded from acting as gap-filling anchors (treated
# the same as a failed-QC pixel) even though the flag says "good."
# ============================================================
def valid_value(qc, value):
    return (not pd.isna(qc)) and qc == 0 and (not pd.isna(value)) and value >= 0

# ============================================================
# HELPER: LOAD ONE HOUR FROM df INTO GRIDS (generic across fields)
# ============================================================
def load_hour(datetime_str, value_column, lon_idx, lat_idx, shape):
    df_h = df[df["Datetime"] == datetime_str].copy()
    if df_h.empty:
        print(f"  Skipping {datetime_str} -- no data")
        return None

    def empty():
        return np.full(shape, np.nan)

    g = {k: empty() for k in ["value", "pixel_num", "qc_flag", "neg_override"] + ALL_MET_FEATURES}

    for _, r in df_h.iterrows():
        i = lat_idx.get(r["Latitude"])
        j = lon_idx.get(r["Longitude"])
        if i is None or j is None:
            continue

        qc  = r["main_data_quality_flag"]
        val = r[value_column]

        g["value"][i, j]        = val if valid_value(qc, val) else np.nan
        g["pixel_num"][i, j]    = r["Pixel Number"]
        g["qc_flag"][i, j]      = qc
        g["neg_override"][i, j] = bool((not pd.isna(qc)) and qc == 0 and (not pd.isna(val)) and val < 0)

        g["u_wind_850mb_ms"][i, j] = r["u_wind_850mb_ms"]
        g["v_wind_850mb_ms"][i, j] = r["v_wind_850mb_ms"]
        g["wspd850"][i, j]         = np.hypot(r["u_wind_850mb_ms"], r["v_wind_850mb_ms"])
        g["u_wind_250mb_ms"][i, j] = r["u_wind_250mb_ms"]
        g["v_wind_250mb_ms"][i, j] = r["v_wind_250mb_ms"]
        g["wspd250"][i, j]         = np.hypot(r["u_wind_250mb_ms"], r["v_wind_250mb_ms"])
        g["temp_2m_K"][i, j]       = r["temp_2m_K"]
        g["rh_2m_pct"][i, j]       = r["rh_2m_pct"]
        g["pbl_height_m"][i, j]    = r["pbl_height_m"]
        g["eff_cloud_fraction"][i, j] = r["eff_cloud_fraction"]
        g["shear_0_1km"][i, j]     = np.hypot(r["shear_u_0_1km_per_s"], r["shear_v_0_1km_per_s"])

    n_valid = np.sum(~np.isnan(g["value"]))
    if n_valid == 0:
        print(f"  Skipping {datetime_str} -- no valid pixels")
        return None

    dt = pd.to_datetime(datetime_str, format="%m/%d/%Y %H:%M")
    g["datetime"] = datetime_str
    g["season"]   = get_season(dt.month)
    g["hour_bin"] = dt.hour // HOUR_BIN_SIZE
    print(f"  Loaded {datetime_str} -- valid anchor pixels: {n_valid}")
    return g

# ============================================================
# HELPER: Z-SCORED, WEIGHTED COSINE SIMILARITY BETWEEN TWO HOURS
# ============================================================
def met_cosine_similarity(g_current, g_context, features, weights):
    shape = g_current["value"].shape
    sim = np.full(shape, 0.0)
    season_cur, hb_cur = g_current["season"], g_current["hour_bin"]
    season_ctx, hb_ctx = g_context["season"], g_context["hour_bin"]

    for i in range(shape[0]):
        for j in range(shape[1]):
            pixel = g_current["pixel_num"][i, j]
            if np.isnan(pixel):
                continue

            vec_cur, vec_ctx, w_vec = [], [], []
            skip = False
            for feat in features:
                zc = zscore(g_current[feat][i, j], feat, season_cur, hb_cur, group_stats, global_stats)
                zx = zscore(g_context[feat][i, j], feat, season_ctx, hb_ctx, group_stats, global_stats)
                if np.isnan(zc) or np.isnan(zx):
                    skip = True
                    break
                vec_cur.append(zc)
                vec_ctx.append(zx)
                w_vec.append(weights[feat])

            if skip:
                continue

            vec_cur = np.array(vec_cur) * np.array(w_vec)
            vec_ctx = np.array(vec_ctx) * np.array(w_vec)

            norm_cur = np.linalg.norm(vec_cur)
            norm_ctx = np.linalg.norm(vec_ctx)
            if norm_cur < 1e-12 or norm_ctx < 1e-12:
                continue

            sim[i, j] = np.dot(vec_cur / norm_cur, vec_ctx / norm_ctx)

    return np.clip(sim, 0, 1)

# ============================================================
# STEP 2: TWO-PASS ADAPTIVE NEIGHBOR FILL
# ============================================================
def adaptive_neighbor_fill(data, i, j):
    for size in range(3, MAX_WINDOW + 1, 2):
        half = size // 2
        vals = []
        for ii in range(max(0, i - half), min(data.shape[0], i + half + 1)):
            for jj in range(max(0, j - half), min(data.shape[1], j + half + 1)):
                if ii == i and jj == j:
                    continue
                if not np.isnan(data[ii, jj]):
                    vals.append(data[ii, jj])
        if len(vals) >= MIN_NEIGHBORS:
            return np.median(vals)
    return np.nan


def two_pass_interp(grid):
    m1 = np.zeros_like(grid, bool)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if not np.isnan(grid[y, x]):
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                            if np.isnan(grid[ny, nx]):
                                m1[ny, nx] = True

    g1 = grid.copy()
    for y in range(g1.shape[0]):
        for x in range(g1.shape[1]):
            if m1[y, x]:
                g1[y, x] = adaptive_neighbor_fill(grid, y, x)

    m2 = np.zeros_like(g1, bool)
    for y in range(g1.shape[0]):
        for x in range(g1.shape[1]):
            if not np.isnan(g1[y, x]):
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < g1.shape[0] and 0 <= nx < g1.shape[1]:
                            if np.isnan(g1[ny, nx]):
                                m2[ny, nx] = True

    g2 = g1.copy()
    for y in range(g2.shape[0]):
        for x in range(g2.shape[1]):
            if m2[y, x]:
                g2[y, x] = adaptive_neighbor_fill(g1, y, x)

    return g2

# ============================================================
# STEP 4a/4c: EDGE-AWARE INTERPOLATION
# ============================================================
def edge_direction_update(i, j, grid, theta):
    vals, weights = [], []
    for ii in range(max(0, i - 1), min(grid.shape[0], i + 2)):
        for jj in range(max(0, j - 1), min(grid.shape[1], j + 2)):
            if ii == i and jj == j:
                continue
            if np.isnan(grid[ii, jj]):
                continue
            vx, vy = jj - j, ii - i
            d = np.hypot(vx, vy) + 1e-12
            gvec = np.array([np.cos(theta[ii, jj]), np.sin(theta[ii, jj])])
            align = abs(np.dot([vx / d, vy / d], gvec))
            w = (1 - align) / d
            vals.append(grid[ii, jj])
            weights.append(w)
    if not vals:
        return grid[i, j]
    weights = np.nan_to_num(weights)
    return (np.sum(np.array(vals) * weights) / np.sum(weights)
            if np.sum(weights) else np.median(vals))

# ============================================================
# STEP 4b: WEIGHTED TOUCHING / WIND / (OPTIONAL PBL) / TEMPORAL FILL
# ============================================================
def weighted_update_all(grid, ugrid, vgrid, blhgrid, grad,
                         grad_scale, wind_scale, blh_scale,
                         wind_exp, blh_exp, grad_exp, val_weight_exp,
                         adjustable_mask, temporal_priors):
    updated = grid.copy()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not adjustable_mask[i, j]:
                continue

            vals, us, vs, blhs, grads, dists = [], [], [], [], [], []
            for ii in range(max(0, i - 1), min(grid.shape[0], i + 2)):
                for jj in range(max(0, j - 1), min(grid.shape[1], j + 2)):
                    if ii == i and jj == j:
                        continue
                    vals.append(grid[ii, jj])
                    us.append(ugrid[ii, jj])
                    vs.append(vgrid[ii, jj])
                    blhs.append(blhgrid[ii, jj] if blhgrid is not None else np.nan)
                    grads.append(grad[ii, jj])
                    dists.append(np.hypot(ii - i, jj - j))

            if not vals:
                continue

            vals, us, vs = np.array(vals), np.array(us), np.array(vs)
            blhs, grads, dists = np.array(blhs), np.array(grads), np.array(dists)

            wind_term = (np.hypot(us - np.nanmedian(us), vs - np.nanmedian(vs)) / wind_scale) ** wind_exp
            grad_term = (grads / grad_scale) ** grad_exp

            if blhgrid is not None:
                blh_term = (np.abs(blhs - np.nanmedian(blhs)) / blh_scale) ** blh_exp
                exponent = -(wind_term + blh_term + grad_term)
            else:
                exponent = -(wind_term + grad_term)

            w_spatial = (1 / (dists + 0.3)) * np.exp(exponent) * (np.abs(vals) ** val_weight_exp)

            total_temporal_w = 0.0
            t_vals, t_ws = [], []
            for w_t_grid, ctx_val in temporal_priors:
                val = ctx_val[i, j]
                w_t = w_t_grid[i, j]
                if np.isnan(val) or np.isnan(w_t) or w_t <= 0:
                    continue
                t_vals.append(val)
                t_ws.append(w_t)
                total_temporal_w += w_t

            spatial_share = max(1.0 - total_temporal_w, 0.0)
            w_spatial_sum = np.nansum(w_spatial)
            w_spatial_scaled = (w_spatial * (spatial_share / w_spatial_sum)
                                if w_spatial_sum > 0 else w_spatial)

            all_vals = np.array(list(vals) + t_vals)
            all_ws   = np.array(list(w_spatial_scaled) + t_ws)

            valid = ~np.isnan(all_vals) & ~np.isnan(all_ws)
            if not np.any(valid):
                continue

            w_sum = np.sum(all_ws[valid])
            if w_sum == 0 or not np.isfinite(w_sum):
                continue

            updated[i, j] = np.sum(all_ws[valid] * all_vals[valid]) / w_sum

    return updated


import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore", message="Mean of empty slice", category=RuntimeWarning
)

# ============================================================
# USER PARAMETERS
# ============================================================
output_path = r"Z:\Nasa_Project\TEMPO_NO2\NO2_retrievals\resampled_pixels\Pandora_sites\Richmond\Richmond_filled_no2.csv"

start_datetime = pd.Timestamp("2023-08-01 00:00")
end_datetime   = pd.Timestamp("2024-12-31 23:00")

FILL_FLAG_VALUE   = 200
TEMPORAL_WEIGHT   = 0.20
TEMPORAL_DECAY    = 0.5
FILL_FLAG_PENALTY = 0.5


def dt_to_datetime_str(dt):
    """Matches the non-zero-padded month/day, zero-padded hour convention
    used when the resampled TEMPO HDF5 was originally written."""
    return f"{dt.month}/{dt.day}/{dt.year} {dt.strftime('%H')}:00"


# ============================================================
# RUN THE GAP-FILLING PIPELINE FOR ONE FIELD, ONE HOUR
# Returns a small DataFrame (Pixel Number, filled_<column>,
# <field_key>_qc_filled) to be merged into the combined output row,
# or None if there's nothing usable for this field/hour.
# ============================================================
def fill_one_field(dt, raw_datetime, cfg, xi, yi, lon_idx, lat_idx, shape):
    column = cfg["column"]

    g_current = load_hour(raw_datetime, column, lon_idx, lat_idx, shape)
    if g_current is None:
        return None

    grid_raw     = g_current["value"]
    pixel_grid   = g_current["pixel_num"]
    qc_grid      = g_current["qc_flag"]
    neg_override = g_current["neg_override"]
    raw_mask     = ~np.isnan(grid_raw)

    context_hours = []
    for offset in range(1, cfg["max_context_hours"] + 1):
        dt_ctx = dt - pd.Timedelta(hours=offset)
        dt_ctx_str = dt_to_datetime_str(dt_ctx)
        g_ctx = load_hour(dt_ctx_str, column, lon_idx, lat_idx, shape)
        if g_ctx is not None:
            context_hours.append((offset, g_ctx))

    grid_filled_2 = two_pass_interp(grid_raw)

    grid_median = median_filter(grid_filled_2, size=3, mode="nearest")
    grid_median[raw_mask] = grid_raw[raw_mask]

    adjustable_mask = ~raw_mask

    gx = prewitt(grid_median, axis=1)
    gy = prewitt(grid_median, axis=0)
    grad_mag = np.hypot(gx, gy)
    theta = np.arctan2(gy, gx)
    grad_thresh = np.nanpercentile(grad_mag, 80)

    for i in range(grid_median.shape[0]):
        for j in range(grid_median.shape[1]):
            if adjustable_mask[i, j]:
                grid_median[i, j] = edge_direction_update(i, j, grid_median, theta)

    u_feats = cfg["wind_u_features"]
    v_feats = cfg["wind_v_features"]
    grid_u_blend = np.nanmean([g_current[f] for f in u_feats], axis=0)
    grid_v_blend = np.nanmean([g_current[f] for f in v_feats], axis=0)
    grid_pbl = g_current["pbl_height_m"] if cfg["use_pbl_driver"] else None

    temporal_priors = []
    for offset, g_ctx in context_hours:
        sim = met_cosine_similarity(g_current, g_ctx, cfg["sim_features"], cfg["feature_weights"])
        decay = TEMPORAL_DECAY ** (offset - 1)
        w_temporal = TEMPORAL_WEIGHT * sim * decay

        ctx_was_filled = (g_ctx["qc_flag"] != 0) | np.isnan(g_ctx["qc_flag"]) | (g_ctx["value"] < 0)
        w_temporal = np.where(ctx_was_filled, w_temporal * FILL_FLAG_PENALTY, w_temporal)

        temporal_priors.append((w_temporal, g_ctx["value"]))

    grid_median = weighted_update_all(
        grid_median, grid_u_blend, grid_v_blend, grid_pbl, grad_mag,
        grad_thresh, 5.0, 2.0, 1.0, 1.0, 1.0, 4.0,
        adjustable_mask, temporal_priors
    )

    gx = prewitt(grid_median, axis=1)
    gy = prewitt(grid_median, axis=0)
    theta = np.arctan2(gy, gx)
    grad_mag = np.hypot(gx, gy)

    for i in range(grid_median.shape[0]):
        for j in range(grid_median.shape[1]):
            if adjustable_mask[i, j]:
                grid_median[i, j] = edge_direction_update(i, j, grid_median, theta)

    changed_mask = ~np.isclose(grid_median, grid_raw, equal_nan=True)
    fill_flag_grid = np.where(neg_override.astype(bool), 201, FILL_FLAG_VALUE)

    pn_flat  = pixel_grid.flatten()
    val_flat = grid_median.flatten()
    qc_flat  = np.where(changed_mask.flatten(), fill_flag_grid.flatten(), qc_grid.flatten())

    valid_pn = ~np.isnan(pn_flat)
    return pd.DataFrame({
        "Pixel Number": pn_flat[valid_pn].astype(int),
        f"filled_{column}": val_flat[valid_pn],
        f"{column}_qc_filled": qc_flat[valid_pn],
    }).sort_values("Pixel Number").reset_index(drop=True)


# ============================================================
# PROCESS ONE HOUR, BOTH FIELDS -- writes one combined row set
# ============================================================
def process_hour(dt, processed_timestamps):
    raw_datetime = dt_to_datetime_str(dt)
    print(f"\n{'='*60}")
    print(f"Processing {raw_datetime}")
    print(f"{'='*60}")

    if raw_datetime in processed_timestamps:
        print(f"  Already processed -- skipping")
        return

    df_raw = df[df["Datetime"] == raw_datetime].copy()
    if df_raw.empty:
        print(f"  No data for {raw_datetime} -- skipping")
        return

    lons = df_raw["Longitude"].values
    lats = df_raw["Latitude"].values
    xi = np.unique(lons)
    yi = np.unique(lats)
    shape = (len(yi), len(xi))
    lon_idx = {v: i for i, v in enumerate(xi)}
    lat_idx = {v: i for i, v in enumerate(yi)}

    df_to_csv = df_raw.copy()

    for field_key, cfg in FIELD_CONFIGS.items():
        column = cfg["column"]
        result_df = fill_one_field(dt, raw_datetime, cfg, xi, yi, lon_idx, lat_idx, shape)

        if result_df is None:
            print(f"  {cfg['label']}: no usable data")
            df_to_csv[f"filled_{column}"] = np.nan
            df_to_csv[f"{column}_qc_filled"] = np.nan
        else:
            print(f"  {cfg['label']}: {len(result_df)} pixels filled")
            df_to_csv = df_to_csv.merge(
                result_df[["Pixel Number", f"filled_{column}", f"{column}_qc_filled"]],
                on="Pixel Number", how="left",
            )

    write_header = not os.path.exists(output_path)
    df_to_csv.to_csv(output_path, mode='a', header=write_header, index=False)
    processed_timestamps.add(raw_datetime)
    print(f"  Written {len(df_to_csv)} rows to {output_path}")


