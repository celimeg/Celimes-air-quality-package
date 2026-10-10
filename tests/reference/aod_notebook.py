# Verbatim from GOES_AOD_GAP_FILL.ipynb (gap-filling cell), up to the resume/main loop.
# Used only as a reference implementation in tests. Expects `df` in globals.
import os
import gc
import warnings
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, prewitt
import pvlib

warnings.filterwarnings(
    "ignore", message="Mean of empty slice", category=RuntimeWarning
)

# ============================================================
# USER PARAMETERS
# ============================================================
output_csv = r"Z:\Nasa_Project\GOES_ABI\resampled_pixels\Sites\CalTech\CalTech_Gap_filled.csv"

start_date = pd.Timestamp("2020-01-01", tz="UTC")
end_date   = pd.Timestamp("2024-12-31", tz="UTC")

# Only pixels within this radius of the site are written to the CSV.
# Computation still uses the full master grid for neighbor context --
# this only trims what gets saved, covering both the 7.5 km and 27.5 km
# averaging buffers used later with margin to spare.
target_point = (34.136730,-118.126167)  # (lat, lon) -- confirm this matches your site
radius_km = 30.0

OUTPUT_COLUMNS = [
    "Pixel Number", "Latitude", "Longitude", "Datetime",
    "Weighted_AOD", "AOD_DQF0", "Weighted_DQF", "avg_DQF", "pct_DQF0",
    "filled_AOD", "pct_DQF0_filled",
]

PCT_DQF0_THRESH = 10

MAX_CONTEXT_HOURS  = 2
SZA_THRESHOLD      = 70.0
TEMPORAL_WEIGHT    = 0.20
TEMPORAL_DECAY     = 0.5
FILL_FLAG_PENALTY  = 0.5

MET_SIM_FEATURES = [
    "u_wind_850mb_ms", "v_wind_850mb_ms",
    "u_wind_250mb_ms", "v_wind_250mb_ms",
    "temp_2m_K", "rh_2m_pct", "pbl_height_m",
]

FEATURE_WEIGHTS = {
    "u_wind_850mb_ms": 2.0,
    "v_wind_850mb_ms": 2.0,
    "u_wind_250mb_ms": 1.0,
    "v_wind_250mb_ms": 1.0,
    "temp_2m_K":        0.5,
    "rh_2m_pct":        0.5,
    "pbl_height_m":     1.75,
    "wspd850":          2.0,
    "wspd250":          0.5,
}

# ============================================================
# HELPER: MATCH df["Datetime"]'S NON-ZERO-PADDED STRING FORMAT
# (e.g. "8/6/2020 19:00", not "08/06/2020 19:00")
# ============================================================
def to_df_datetime_str(ts):
    ts_naive = ts.tz_localize(None) if ts.tzinfo is not None else ts
    return (ts_naive.strftime("%m/%d/%Y %H:%M")
            .replace("/0", "/")
            .lstrip("0"))

# ============================================================
# HELPER: READ THE LAST ROW OF A LARGE CSV WITHOUT LOADING IT
# (seeks backward from end of file in chunks -- safe on a
# multi-year, multi-GB file since it never reads the whole thing)
# ============================================================
def get_last_datetime(path, datetime_col="Datetime"):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None

    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        buffer = b""
        pos = file_size
        chunk = 4096
        while pos > 0:
            read_size = min(chunk, pos)
            pos -= read_size
            f.seek(pos)
            buffer = f.read(read_size) + buffer
            if buffer.count(b"\n") >= 2:
                break

    lines = [l for l in buffer.splitlines() if l.strip()]
    if len(lines) < 1:
        return None
    last_line = lines[-1].decode("utf-8")

    with open(path, "r") as f:
        header = f.readline().strip().split(",")
    dt_idx = header.index(datetime_col)

    last_values = last_line.split(",")
    return last_values[dt_idx]

# ============================================================
# HELPER: HAVERSINE DISTANCE (matches the AERONET extraction script)
# ============================================================
def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))

# ============================================================
# HELPER: TRIM A HOUR'S ROWS TO THE SAVE RADIUS -- applied at every
# process_hour() return point so the filter lives in one place
# ============================================================
def restrict_to_radius(df_subset):
    if df_subset.empty:
        return df_subset
    dist = haversine_km(
        df_subset["Latitude"].values, df_subset["Longitude"].values,
        target_point[0], target_point[1]
    )
    return df_subset[dist <= radius_km]

# ============================================================
# HELPER: SOLAR ZENITH ANGLE
# ============================================================
def compute_sza(dt, lat, lon):
    times = pd.DatetimeIndex([dt], tz="UTC") if dt.tzinfo is None else pd.DatetimeIndex([dt])
    sp    = pvlib.solarposition.get_solarposition(times, lat, lon)
    return float(sp["zenith"].iloc[0])

# ============================================================
# HELPER: LOAD ONE HOUR FROM df INTO GRIDS (MASTER GRID SHAPE)
# ============================================================
def load_hour(dt, lat_idx, lon_idx, shape, domain_center_lat, domain_center_lon):
    dt_str = to_df_datetime_str(dt)
    sza = compute_sza(dt, domain_center_lat, domain_center_lon)
    if sza > SZA_THRESHOLD:
        return None

    df_h = df[df["Datetime"] == dt_str].copy()
    if df_h.empty:
        return None

    fail = df_h["pct_DQF0"] < PCT_DQF0_THRESH
    df_h.loc[fail, "Weighted_AOD"] = np.nan

    def empty():
        return np.full(shape, np.nan)

    g = {k: empty() for k in [
        "aod", "u_wind_850mb_ms", "v_wind_850mb_ms",
        "u_wind_250mb_ms", "v_wind_250mb_ms",
        "temp_2m_K", "rh_2m_pct", "pbl_height_m",
        "pct_dqf0",
    ]}

    for _, r in df_h.iterrows():
        i = lat_idx.get(r["Latitude"])
        j = lon_idx.get(r["Longitude"])
        if i is None or j is None:
            continue
        g["aod"][i, j]              = r["Weighted_AOD"]
        g["pct_dqf0"][i, j]         = r["pct_DQF0"]
        g["u_wind_850mb_ms"][i, j]  = r["u_wind_850mb_ms"]
        g["v_wind_850mb_ms"][i, j]  = r["v_wind_850mb_ms"]
        g["u_wind_250mb_ms"][i, j]  = r["u_wind_250mb_ms"]
        g["v_wind_250mb_ms"][i, j]  = r["v_wind_250mb_ms"]
        g["temp_2m_K"][i, j]        = r["temp_2m_K"]
        g["rh_2m_pct"][i, j]        = r["rh_2m_pct"]
        g["pbl_height_m"][i, j]     = r["pbl_height_m"]

    if np.sum(~np.isnan(g["aod"])) == 0:
        return None

    return g

# ============================================================
# HELPER: COSINE SIMILARITY WITH FEATURE WEIGHTS + WIND SPEED
# ============================================================
def met_cosine_similarity(g_current, g_context):
    shape = g_current["aod"].shape
    sim   = np.full(shape, 0.0)

    weight_vec = np.array([
        FEATURE_WEIGHTS["u_wind_850mb_ms"],
        FEATURE_WEIGHTS["v_wind_850mb_ms"],
        FEATURE_WEIGHTS["u_wind_250mb_ms"],
        FEATURE_WEIGHTS["v_wind_250mb_ms"],
        FEATURE_WEIGHTS["temp_2m_K"],
        FEATURE_WEIGHTS["rh_2m_pct"],
        FEATURE_WEIGHTS["pbl_height_m"],
        FEATURE_WEIGHTS["wspd850"],
        FEATURE_WEIGHTS["wspd250"],
    ])

    for i in range(shape[0]):
        for j in range(shape[1]):
            vec_cur = np.array([g_current[f][i, j] for f in MET_SIM_FEATURES])
            vec_ctx = np.array([g_context[f][i, j]  for f in MET_SIM_FEATURES])

            if np.any(np.isnan(vec_cur)) or np.any(np.isnan(vec_ctx)):
                continue

            wspd850_cur = np.hypot(g_current["u_wind_850mb_ms"][i, j], g_current["v_wind_850mb_ms"][i, j])
            wspd850_ctx = np.hypot(g_context["u_wind_850mb_ms"][i, j], g_context["v_wind_850mb_ms"][i, j])
            wspd250_cur = np.hypot(g_current["u_wind_250mb_ms"][i, j], g_current["v_wind_250mb_ms"][i, j])
            wspd250_ctx = np.hypot(g_context["u_wind_250mb_ms"][i, j], g_context["v_wind_250mb_ms"][i, j])

            vec_cur = np.append(vec_cur, [wspd850_cur, wspd250_cur])
            vec_ctx = np.append(vec_ctx, [wspd850_ctx, wspd250_ctx])

            vec_cur = vec_cur * weight_vec
            vec_ctx = vec_ctx * weight_vec

            norm_cur = np.linalg.norm(vec_cur)
            norm_ctx = np.linalg.norm(vec_ctx)
            if norm_cur < 1e-12 or norm_ctx < 1e-12:
                continue

            sim[i, j] = np.dot(vec_cur / norm_cur, vec_ctx / norm_ctx)

    return np.clip(sim, 0, 1)

# ============================================================
# HELPER: TWO-PASS ADAPTIVE NEIGHBOR FILL
# ============================================================
MIN_NEIGHBORS = 2
MAX_WINDOW    = 9

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
# HELPER: EDGE DIRECTION UPDATE
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
            d      = np.hypot(vx, vy) + 1e-12
            gvec   = np.array([np.cos(theta[ii, jj]), np.sin(theta[ii, jj])])
            align  = abs(np.dot([vx / d, vy / d], gvec))
            w      = (1 - align) / d
            vals.append(grid[ii, jj])
            weights.append(w)
    if not vals:
        return grid[i, j]
    weights = np.nan_to_num(weights)
    return (np.sum(np.array(vals) * weights) / np.sum(weights)
            if np.sum(weights) else np.median(vals))

# ============================================================
# HELPER: WEIGHTED UPDATE ALL
# ============================================================
def weighted_update_all(grid, ugrid, vgrid, blhgrid, grad,
                        grad_scale, wind_scale, blh_scale,
                        wind_exp, blh_exp, grad_exp, aod_weight_exp,
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
                    blhs.append(blhgrid[ii, jj])
                    grads.append(grad[ii, jj])
                    dists.append(np.hypot(ii - i, jj - j))
            if not vals:
                continue
            vals  = np.array(vals)
            us    = np.array(us)
            vs    = np.array(vs)
            blhs  = np.array(blhs)
            grads = np.array(grads)
            dists = np.array(dists)
            w_spatial = (1 / (dists + 0.3)) * np.exp(
                -((np.hypot(us - np.nanmedian(us), vs - np.nanmedian(vs)) / wind_scale) ** wind_exp +
                  (np.abs(blhs - np.nanmedian(blhs)) / blh_scale) ** blh_exp +
                  (grads / grad_scale) ** grad_exp)
            ) * (vals ** aod_weight_exp)
            total_temporal_w = 0.0
            t_vals, t_ws     = [], []
            for w_t_grid, aod_ctx in temporal_priors:
                aod_val = aod_ctx[i, j]
                w_t     = w_t_grid[i, j]
                if np.isnan(aod_val) or np.isnan(w_t) or w_t <= 0:
                    continue
                t_vals.append(aod_val)
                t_ws.append(w_t)
                total_temporal_w += w_t
            spatial_share = max(1.0 - total_temporal_w, 0.0)
            w_spatial_sum = np.nansum(w_spatial)
            if w_spatial_sum > 0:
                w_spatial_scaled = w_spatial * (spatial_share / w_spatial_sum)
            else:
                w_spatial_scaled = w_spatial
            all_vals = list(vals) + t_vals
            all_ws   = list(w_spatial_scaled) + t_ws
            all_vals = np.array(all_vals)
            all_ws   = np.array(all_ws)
            valid    = ~np.isnan(all_vals) & ~np.isnan(all_ws)
            if not np.any(valid):
                continue
            w_sum = np.sum(all_ws[valid])
            if w_sum == 0 or not np.isfinite(w_sum):
                continue
            updated[i, j] = np.sum(all_ws[valid] * all_vals[valid]) / w_sum
    return updated

# ============================================================
# PROCESS ONE HOUR -> RETURNS A DATAFRAME (never None; if there's
# no data or SZA is too high, returns raw rows with NaN fill columns,
# matching the original per-hour CSV behavior)
# ============================================================
def process_hour(raw_datetime, lat_idx, lon_idx, shape):
    dt_str = to_df_datetime_str(raw_datetime)

    df_raw = df[df["Datetime"] == dt_str].copy()
    if df_raw.empty:
        return restrict_to_radius(df_raw)  # empty frame, nothing to add

    df_raw["filled_AOD"] = np.nan
    df_raw["pct_DQF0_filled"] = np.nan

    sza = compute_sza(raw_datetime, df_raw["Latitude"].mean(), df_raw["Longitude"].mean())
    if sza > SZA_THRESHOLD:
        return restrict_to_radius(df_raw[OUTPUT_COLUMNS])

    df_day = df_raw.copy()
    fail_qc = df_day["pct_DQF0"] < PCT_DQF0_THRESH
    df_day.loc[fail_qc, "Weighted_AOD"] = np.nan

    def empty():
        return np.full(shape, np.nan)

    grid_raw      = empty()
    grid_u850     = empty()
    grid_v850     = empty()
    grid_u250     = empty()
    grid_v250     = empty()
    grid_temp2m   = empty()
    grid_rh       = empty()
    grid_pbl      = empty()
    pct_dqf0_grid = empty()

    for _, r in df_day.iterrows():
        i = lat_idx.get(r["Latitude"])
        j = lon_idx.get(r["Longitude"])
        if i is None or j is None:
            continue
        grid_raw[i, j]      = r["Weighted_AOD"]
        pct_dqf0_grid[i, j] = r["pct_DQF0"]
        grid_u850[i, j]     = r["u_wind_850mb_ms"]
        grid_v850[i, j]     = r["v_wind_850mb_ms"]
        grid_u250[i, j]     = r["u_wind_250mb_ms"]
        grid_v250[i, j]     = r["v_wind_250mb_ms"]
        grid_temp2m[i, j]   = r["temp_2m_K"]
        grid_rh[i, j]       = r["rh_2m_pct"]
        grid_pbl[i, j]      = r["pbl_height_m"]

    if np.sum(~np.isnan(grid_raw)) == 0:
        return restrict_to_radius(df_raw[OUTPUT_COLUMNS])

    raw_mask = ~np.isnan(grid_raw)

    domain_center_lat = df_day["Latitude"].mean()
    domain_center_lon = df_day["Longitude"].mean()

    g_current = {
        "aod":              grid_raw,
        "u_wind_850mb_ms":  grid_u850, "v_wind_850mb_ms": grid_v850,
        "u_wind_250mb_ms":  grid_u250, "v_wind_250mb_ms": grid_v250,
        "temp_2m_K":        grid_temp2m,
        "rh_2m_pct":        grid_rh,
        "pbl_height_m":     grid_pbl,
    }

    context_hours = []
    for offset in range(1, MAX_CONTEXT_HOURS + 1):
        dt_ctx = raw_datetime - pd.Timedelta(hours=offset)
        g_ctx  = load_hour(dt_ctx, lat_idx, lon_idx, shape,
                           domain_center_lat, domain_center_lon)
        if g_ctx is not None:
            context_hours.append((offset, g_ctx))

    grid_filled_2 = two_pass_interp(grid_raw)

    grid_median = median_filter(grid_filled_2, size=3, mode="nearest")
    grid_median[raw_mask] = grid_raw[raw_mask]

    adjustable_mask = (pct_dqf0_grid < PCT_DQF0_THRESH) | np.isnan(pct_dqf0_grid)
    gx          = prewitt(grid_median, axis=1)
    gy          = prewitt(grid_median, axis=0)
    grad_mag    = np.hypot(gx, gy)
    theta       = np.arctan2(gy, gx)
    grad_thresh = np.nanpercentile(grad_mag, 80)

    for i in range(grid_median.shape[0]):
        for j in range(grid_median.shape[1]):
            if adjustable_mask[i, j]:
                grid_median[i, j] = edge_direction_update(i, j, grid_median, theta)

    temporal_priors = []
    for offset, g_ctx in context_hours:
        sim        = met_cosine_similarity(g_current, g_ctx)
        decay      = TEMPORAL_DECAY ** (offset - 1)
        w_temporal = TEMPORAL_WEIGHT * sim * decay
        ctx_was_filled = (g_ctx["pct_dqf0"] < PCT_DQF0_THRESH) | np.isnan(g_ctx["pct_dqf0"])
        w_temporal     = np.where(ctx_was_filled, w_temporal * FILL_FLAG_PENALTY, w_temporal)
        temporal_priors.append((w_temporal, g_ctx["aod"]))

    grid_u_blend = (grid_u850 + grid_u250) / 2.0
    grid_v_blend = (grid_v850 + grid_v250) / 2.0

    grid_median = weighted_update_all(
        grid_median, grid_u_blend, grid_v_blend, grid_pbl, grad_mag,
        grad_thresh, 5.0, 2.0, 1.0, 1.0, 1.0, 4.0,
        adjustable_mask, temporal_priors
    )

    gx          = prewitt(grid_median, axis=1)
    gy          = prewitt(grid_median, axis=0)
    theta       = np.arctan2(gy, gx)
    grad_mag    = np.hypot(gx, gy)
    grad_thresh = np.nanpercentile(grad_mag, 80)
    for i in range(grid_median.shape[0]):
        for j in range(grid_median.shape[1]):
            if adjustable_mask[i, j]:
                grid_median[i, j] = edge_direction_update(i, j, grid_median, theta)

    changed_mask = ~np.isclose(grid_median, grid_raw, equal_nan=True)
    dqf_out = np.where(changed_mask, 200, pct_dqf0_grid)

    # Map filled results back onto df_raw's rows via lat/lon lookup
    filled_vals = []
    dqf_vals = []
    for _, r in df_raw.iterrows():
        i = lat_idx.get(r["Latitude"])
        j = lon_idx.get(r["Longitude"])
        if i is None or j is None:
            filled_vals.append(np.nan)
            dqf_vals.append(np.nan)
        else:
            filled_vals.append(grid_median[i, j])
            dqf_vals.append(dqf_out[i, j])

    df_raw["filled_AOD"] = filled_vals
    df_raw["pct_DQF0_filled"] = dqf_vals

    return restrict_to_radius(df_raw[OUTPUT_COLUMNS])

# ============================================================
# BUILD MASTER GRID ONCE (fixed shape shared by every hour, every day)
# ============================================================
yi_master = np.sort(df["Latitude"].unique())
xi_master = np.sort(df["Longitude"].unique())
master_shape = (len(yi_master), len(xi_master))

lat_idx_master = {v: i for i, v in enumerate(yi_master)}
lon_idx_master = {v: i for i, v in enumerate(xi_master)}

print(f"Master grid: {master_shape[0]} lat x {master_shape[1]} lon")

