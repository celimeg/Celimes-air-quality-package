# Verbatim from saving_goes_abi_aod_to_csv-Copy3.ipynb. Reference for tests;
# the test substitutes the paths and the Windows-only %#m strftime flags.
import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

year = "2022"

base_path = f"Z:/Nasa_Project/GOES_ABI/{year}"
centroid_csv = r"Z:\Nasa_Project\TEMPO_NO2\NO2_retrievals\resampled_pixels\Las_Vegas\TEMPO_LV_footprint.csv"
output_path = r"Z:\Nasa_Project\GOES_ABI\resampled_pixels\Sites\LV"
os.makedirs(output_path, exist_ok=True)

out_csv = os.path.join(output_path, f"LV_ABI_AOD_{year}_tempo_resolution.csv")
first_write = not os.path.exists(out_csv)
dqf_weights = np.array([1.0, 0.5, 0.25, 0.0])

# --------------------------------------------------
# LOAD MASTER GRID
# --------------------------------------------------
grid_df = pd.read_csv(centroid_csv)
grid_df = grid_df.sort_values("Pixel Number")

pixel_ids = grid_df["Pixel Number"].values
lat_all   = grid_df["Latitude"].values
lon_all   = grid_df["Longitude"].values

first_lon = lon_all[0]
n_rows    = np.sum(lon_all == first_lon)
n_cols    = len(pixel_ids) // n_rows

lat_grid       = lat_all.reshape(n_cols, n_rows).T
lon_grid       = lon_all.reshape(n_cols, n_rows).T
pixel_ids_grid = pixel_ids.reshape(n_cols, n_rows).T

lat_step = np.abs(lat_grid[0, 0] - lat_grid[1, 0])
lon_step = np.abs(lon_grid[0, 0] - lon_grid[0, 1])

# lat_edges descending, lon_edges ascending
lat_edges = np.concatenate([[lat_grid[0, 0] + lat_step/2], lat_grid[:, 0] - lat_step/2])
lon_edges = np.concatenate([lon_grid[0, :] - lon_step/2, [lon_grid[0, -1] + lon_step/2]])

# --------------------------------------------------
# PRECOMPUTE ABI PROJECTION
# --------------------------------------------------
print("Precomputing ABI projection grid...")

sample_file = None
for doy in sorted(os.listdir(base_path)):
    day_path = os.path.join(base_path, doy)
    if os.path.isdir(day_path):
        hour_dirs = os.listdir(day_path)
        if hour_dirs:
            files = [f for f in os.listdir(os.path.join(day_path, hour_dirs[0])) if f.endswith(".nc")]
            if files:
                sample_file = os.path.join(day_path, hour_dirs[0], files[0])
                break

ds_sample = xr.open_dataset(sample_file, decode_times=False)
x    = ds_sample["x"].values
y    = ds_sample["y"].values
proj = ds_sample["goes_imager_projection"]

lon0  = proj.attrs["longitude_of_projection_origin"]
H     = proj.attrs["perspective_point_height"] + proj.attrs["semi_major_axis"]
r_eq  = proj.attrs["semi_major_axis"]
r_pol = proj.attrs["semi_minor_axis"]
ds_sample.close()

lambda0 = np.deg2rad(lon0)

def scan_to_latlon(x2d, y2d):
    a = (np.sin(x2d)**2 +
         np.cos(x2d)**2 *
         (np.cos(y2d)**2 + (r_eq**2/r_pol**2)*np.sin(y2d)**2))
    b = -2*H*np.cos(x2d)*np.cos(y2d)
    c = H**2 - r_eq**2
    disc = b**2 - 4*a*c
    r_s  = np.full_like(a, np.nan)
    valid = disc >= 0
    r_s[valid] = (-b[valid] - np.sqrt(disc[valid])) / (2*a[valid])
    s_x = r_s*np.cos(x2d)*np.cos(y2d)
    s_y = -r_s*np.sin(x2d)
    s_z = r_s*np.cos(x2d)*np.sin(y2d)
    lat = np.rad2deg(np.arctan((r_eq**2/r_pol**2)*(s_z/np.sqrt((H-s_x)**2+s_y**2))))
    lon = np.rad2deg(lambda0 - np.arctan(s_y/(H-s_x)))
    return lat, lon

# Pixel centers
x2d_c, y2d_c = np.meshgrid(x, y)
abi_lat, abi_lon = scan_to_latlon(x2d_c, y2d_c)

# Pixel edges
dx = np.diff(x).mean()
dy = np.diff(y).mean()
x_edges = np.concatenate([[x[0] - dx/2], x + dx/2])
y_edges = np.concatenate([[y[0] - dy/2], y + dy/2])

xe2d, ye2d = np.meshgrid(x_edges, y_edges)
lat_edge_grid, lon_edge_grid = scan_to_latlon(xe2d, ye2d)

# Precompute domain pixels
domain_pad = 0.1
in_domain = (
    (abi_lat >= lat_edges[-1] - domain_pad) & (abi_lat <= lat_edges[0] + domain_pad) &
    (abi_lon >= lon_edges[0]  - domain_pad) & (abi_lon <= lon_edges[-1] + domain_pad)
)
domain_rows, domain_cols = np.where(in_domain)
print(f"ABI pixels near study area: {len(domain_rows)}")

# --------------------------------------------------
# RESUME LOGIC
# --------------------------------------------------
last_datetime = None
if os.path.exists(out_csv):
    try:
        existing = pd.read_csv(out_csv, usecols=["Datetime"])
        if len(existing) > 0:
            last_datetime_str = existing["Datetime"].iloc[-1]
            last_datetime = datetime.strptime(last_datetime_str, "%m/%d/%Y %H:00")
            first_write = False
            print(f"Resuming from {last_datetime}", flush=True)
    except:
        print("Could not determine last datetime. Starting fresh.", flush=True)

# --------------------------------------------------
# PROCESS FILES
# --------------------------------------------------
for doy in range(1, 367):

    day_path = os.path.join(base_path, f"{doy:03d}")
    if not os.path.isdir(day_path):
        continue

    for hour in sorted(os.listdir(day_path)):

        hour_path = os.path.join(day_path, hour)
        if not os.path.isdir(hour_path):
            continue

        try:
            hour_datetime = datetime(int(year), 1, 1) + timedelta(days=doy-1, hours=int(hour))

            if last_datetime is not None and hour_datetime <= last_datetime:
                continue

            print(f"Processing DOY {doy:03d} Hour {hour}", flush=True)

            files_in_hour = [f for f in os.listdir(hour_path) if f.endswith(".nc")]
            if not files_in_hour:
                continue

            datetime_str = hour_datetime.strftime("%#m/%#d/%Y %H:00")

            # Accumulators
            sum_aod_w    = np.zeros((n_rows, n_cols))
            sum_w        = np.zeros((n_rows, n_cols))
            sum_dqf_raw  = np.zeros((n_rows, n_cols))
            cnt_dqf      = np.zeros((n_rows, n_cols))
            sum_aod_dqf0 = np.zeros((n_rows, n_cols))
            cnt_aod_dqf0 = np.zeros((n_rows, n_cols))
            cnt_dqf0     = np.zeros((n_rows, n_cols))

            for f in files_in_hour:
                try:
                    ds  = xr.open_dataset(os.path.join(hour_path, f), decode_times=False)
                    AOD = ds["AOD"].values
                    DQF = ds["DQF"].values
                    ds.close()
                except:
                    print(f"Skipping corrupted file: {f}")
                    continue

                # Extract all domain pixels at once
                aod_vals = AOD[domain_rows, domain_cols]
                dqf_vals = DQF[domain_rows, domain_cols]

                # Filter valid pixels
                valid = (
                    ~np.isnan(aod_vals) & (aod_vals >= 0) &
                    ~np.isnan(dqf_vals) & (dqf_vals >= 0) & (dqf_vals <= 3)
                )
                if not np.any(valid):
                    continue

                pi_v   = domain_rows[valid]
                pj_v   = domain_cols[valid]
                aod_v  = aod_vals[valid]
                dqf_v  = dqf_vals[valid].astype(int)
                dqf_wv = dqf_weights[dqf_v]

                # Get ABI pixel edges for all valid pixels at once
                abi_lat_lo = lat_edge_grid[pi_v + 1, pj_v]
                abi_lat_hi = lat_edge_grid[pi_v,     pj_v]
                abi_lon_lo = lon_edge_grid[pi_v,     pj_v]
                abi_lon_hi = lon_edge_grid[pi_v,     pj_v + 1]

                # Drop pixels with NaN edges
                edge_valid = ~(np.isnan(abi_lat_lo) | np.isnan(abi_lat_hi) |
                               np.isnan(abi_lon_lo) | np.isnan(abi_lon_hi))

                pi_v       = pi_v[edge_valid];       pj_v       = pj_v[edge_valid]
                aod_v      = aod_v[edge_valid];      dqf_v      = dqf_v[edge_valid]
                dqf_wv     = dqf_wv[edge_valid]
                abi_lat_lo = abi_lat_lo[edge_valid]; abi_lat_hi = abi_lat_hi[edge_valid]
                abi_lon_lo = abi_lon_lo[edge_valid]; abi_lon_hi = abi_lon_hi[edge_valid]
                abi_area   = (abi_lat_hi - abi_lat_lo) * (abi_lon_hi - abi_lon_lo)

                # Find overlapping grid cells
                lat_lo_idx = np.clip(np.searchsorted(-lat_edges, -abi_lat_hi, side='left'),  0, n_rows)
                lat_hi_idx = np.clip(np.searchsorted(-lat_edges, -abi_lat_lo, side='right'), 0, n_rows)
                lon_lo_idx = np.clip(np.searchsorted(lon_edges, abi_lon_lo, side='right') - 1, 0, n_cols)
                lon_hi_idx = np.clip(np.searchsorted(lon_edges, abi_lon_hi, side='left'),      0, n_cols)

                max_lat_span = int((lat_hi_idx - lat_lo_idx).max()) if len(pi_v) > 0 else 0
                max_lon_span = int((lon_hi_idx - lon_lo_idx).max()) if len(pi_v) > 0 else 0

                for di in range(max_lat_span + 1):
                    for dj in range(max_lon_span + 1):
                        gi = lat_lo_idx + di
                        gj = lon_lo_idx + dj

                        in_range = (gi < lat_hi_idx) & (gi < n_rows) & \
                                   (gj < lon_hi_idx) & (gj < n_cols) & \
                                   (gi >= 0) & (gj >= 0)

                        if not np.any(in_range):
                            continue

                        gi_v     = gi[in_range];        gj_v     = gj[in_range]
                        aod_s    = aod_v[in_range];     dqf_s    = dqf_v[in_range]
                        dqf_ws   = dqf_wv[in_range]
                        lat_lo_s = abi_lat_lo[in_range]; lat_hi_s = abi_lat_hi[in_range]
                        lon_lo_s = abi_lon_lo[in_range]; lon_hi_s = abi_lon_hi[in_range]
                        area_s   = abi_area[in_range]

                        g_lat_hi = lat_edges[gi_v]
                        g_lat_lo = lat_edges[gi_v + 1]
                        g_lon_lo = lon_edges[gj_v]
                        g_lon_hi = lon_edges[gj_v + 1]

                        overlap_lat = np.minimum(lat_hi_s, g_lat_hi) - np.maximum(lat_lo_s, g_lat_lo)
                        overlap_lon = np.minimum(lon_hi_s, g_lon_hi) - np.maximum(lon_lo_s, g_lon_lo)

                        pos = (overlap_lat > 0) & (overlap_lon > 0) & (area_s > 0)
                        if not np.any(pos):
                            continue

                        gi_v         = gi_v[pos];         gj_v         = gj_v[pos]
                        aod_s        = aod_s[pos];        dqf_s        = dqf_s[pos]
                        dqf_ws       = dqf_ws[pos]
                        overlap_frac = (overlap_lat[pos] * overlap_lon[pos]) / area_s[pos]
                        combined_w   = overlap_frac * dqf_ws

                        np.add.at(sum_aod_w,   (gi_v, gj_v), aod_s * combined_w)
                        np.add.at(sum_w,        (gi_v, gj_v), combined_w)
                        np.add.at(sum_dqf_raw,  (gi_v, gj_v), dqf_s)
                        np.add.at(cnt_dqf,      (gi_v, gj_v), 1)

                        dqf0 = dqf_s == 0
                        if np.any(dqf0):
                            np.add.at(sum_aod_dqf0, (gi_v[dqf0], gj_v[dqf0]), aod_s[dqf0] * overlap_frac[dqf0])
                            np.add.at(cnt_aod_dqf0, (gi_v[dqf0], gj_v[dqf0]), overlap_frac[dqf0])
                            np.add.at(cnt_dqf0,     (gi_v[dqf0], gj_v[dqf0]), 1)

            AOD_final = sum_aod_w    / np.where(sum_w == 0,         np.nan, sum_w)
            avg_DQF   = sum_dqf_raw  / np.where(cnt_dqf == 0,       np.nan, cnt_dqf)
            DQF_final = sum_w        / np.where(cnt_dqf == 0,       np.nan, cnt_dqf)
            AOD_DQF0  = sum_aod_dqf0 / np.where(cnt_aod_dqf0 == 0,  np.nan, cnt_aod_dqf0)
            pct_DQF0  = (cnt_dqf0    / np.where(cnt_dqf == 0,       np.nan, cnt_dqf)) * 100

            df = pd.DataFrame({
                "Pixel Number": pixel_ids_grid.flatten(),
                "Latitude":     lat_grid.flatten(),
                "Longitude":    lon_grid.flatten(),
                "Weighted_AOD": AOD_final.flatten(),
                "AOD_DQF0":     AOD_DQF0.flatten(),
                "Weighted_DQF": DQF_final.flatten(),
                "avg_DQF":      avg_DQF.flatten(),
                "pct_DQF0":     pct_DQF0.flatten(),
                "Datetime":     datetime_str
            }).sort_values("Pixel Number")

            df.to_csv(
                out_csv,
                mode="w" if first_write else "a",
                header=first_write,
                index=False
            )
            first_write = False

        except Exception as e:
            print(f"Skipping entire hour DOY {doy:03d} Hour {hour} due to error: {e}")
            continue

print("Finished!")