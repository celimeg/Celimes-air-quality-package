# Verbatim from tempo_to_csv-Copy1.ipynb (yearly HDF5 cell). Reference for tests.
import os
import glob
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from datetime import datetime

# -------------------------------
# PATHS
# -------------------------------
organized_root = r"Z:\Nasa_Project\TEMPO_NO2\NO2_retrievals\L3_data\organized"
footprint_csv  = r"Z:\Nasa_Project\TEMPO_NO2\NO2_retrievals\resampled_pixels\Las_Vegas\TEMPO_LV_footprint.csv"
output_path    = r"Z:\Nasa_Project\TEMPO_NO2\NO2_retrievals\resampled_pixels\Las_Vegas\no2"
os.makedirs(output_path, exist_ok=True)
out_h5 = os.path.join(output_path, "Resampled_TEMPO_NO2_LV.h5")

# -------------------------------
# FIELDS
# -------------------------------
fields = {
    "geolocation/relative_azimuth_angle":                                   "relative_azimuth_angle",
    "geolocation/solar_zenith_angle":                                       "solar_zenith_angle",
    "geolocation/viewing_zenith_angle":                                     "viewing_zenith_angle",
    "product/vertical_column_stratosphere":                                 "vertical_column_stratosphere",
    "product/vertical_column_troposphere":                                  "vertical_column_troposphere",
    "product/vertical_column_troposphere_uncertainty":                      "vertical_column_troposphere_uncertainty",
    "qa_statistics/max_vertical_column_stratosphere_sample":                "max_vertical_column_stratosphere_sample",
    "qa_statistics/max_vertical_column_total_sample":                       "max_vertical_column_total_sample",
    "qa_statistics/max_vertical_column_troposphere_sample":                 "max_vertical_column_troposphere_sample",
    "qa_statistics/max_vertical_column_troposphere_uncertainty_sample":     "max_vertical_column_troposphere_uncertainty_sample",
    "qa_statistics/min_vertical_column_stratosphere_sample":                "min_vertical_column_stratosphere_sample",
    "qa_statistics/min_vertical_column_total_sample":                       "min_vertical_column_total_sample",
    "qa_statistics/min_vertical_column_troposphere_sample":                 "min_vertical_column_troposphere_sample",
    "qa_statistics/min_vertical_column_troposphere_uncertainty_sample":     "min_vertical_column_troposphere_uncertainty_sample",
    "support_data/albedo":                                                  "albedo",
    "support_data/amf_cloud_fraction":                                      "amf_cloud_fraction",
    "support_data/amf_stratosphere":                                        "amf_stratosphere",
    "support_data/amf_total":                                               "amf_total",
    "support_data/amf_troposphere":                                         "amf_troposphere",
    "support_data/eff_cloud_fraction":                                      "eff_cloud_fraction",
    "support_data/fitted_slant_column":                                     "fitted_slant_column",
    "support_data/fitted_slant_column_uncertainty":                         "fitted_slant_column_uncertainty",
    "support_data/terrain_height":                                          "terrain_height",
    "support_data/vertical_column_total":                                   "vertical_column_total",
    "support_data/vertical_column_total_uncertainty":                       "vertical_column_total_uncertainty",
}

# -------------------------------
# LOAD TARGET FOOTPRINT GRID
# -------------------------------
grid_df = pd.read_csv(footprint_csv).sort_values("Pixel Number").reset_index(drop=True)
pixel_ids  = grid_df["Pixel Number"].values
lat_target = grid_df["Latitude"].values
lon_target = grid_df["Longitude"].values
n_pixels   = len(pixel_ids)

# -------------------------------
# PRECOMPUTE NATIVE-GRID INDEX LOOKUP (once)
# TEMPO L3 already delivers a regular lat/lon grid at the same
# spacing as the footprint (it was built directly from these native
# axes), so each target pixel maps to exactly one native TEMPO
# index -- pixels just "clip" onto the grid, no area weighting.
# -------------------------------
sample_file = next(Path(organized_root).rglob("*.nc"), None)
if sample_file is None:
    raise FileNotFoundError(f"No .nc files found under {organized_root}")

with h5py.File(sample_file, "r") as f:
    lon_axis = f["longitude"][:]
    lat_axis = f["latitude"][:]

lon_idx = np.array([np.argmin(np.abs(lon_axis - lon)) for lon in lon_target])
lat_idx = np.array([np.argmin(np.abs(lat_axis - lat)) for lat in lat_target])

# Sanity check -- confirm nearest-index matches are essentially exact.
# If this ever prints a large error, the footprint grid and native
# TEMPO grid have drifted apart and area weighting would be needed.
tol = 1e-4  # degrees
lon_err = np.abs(lon_axis[lon_idx] - lon_target)
lat_err = np.abs(lat_axis[lat_idx] - lat_target)
if np.any(lon_err > tol) or np.any(lat_err > tol):
    print(f"WARNING: footprint pixels not exactly aligned with native TEMPO grid "
          f"(max lon err={lon_err.max():.5f} deg, max lat err={lat_err.max():.5f} deg)")

# -------------------------------
# RESUME LOGIC
# -------------------------------
last_datetime = None
if os.path.exists(out_h5):
    try:
        existing = pd.read_hdf(out_h5, key="no2", columns=["Datetime"])
        if len(existing) > 0:
            last_datetime = pd.to_datetime(existing["Datetime"].iloc[-1], format="%m/%d/%Y %H:00")
            print(f"Resuming from {last_datetime}", flush=True)
    except Exception:
        print("Could not determine last datetime. Starting fresh.", flush=True)

# -------------------------------
# GROUP FILES BY HOUR FOLDER
# (usually 1 file per hour, occasionally 2 -- averaged if so)
# -------------------------------
all_files = sorted(Path(organized_root).rglob("*.nc"))
print(f"Found {len(all_files)} .nc files", flush=True)

hour_groups = {}
for fp in all_files:
    hour_groups.setdefault(fp.parent, []).append(fp)

files_processed = 0
files_skipped   = 0
total_rows      = 0

for hour_path in sorted(hour_groups.keys()):
    nc_files = sorted(hour_groups[hour_path])

    # Parse datetime from folder structure: .../year/month/day/hour/
    parts = hour_path.parts
    try:
        hour  = parts[-1]
        day   = parts[-2]
        month = parts[-3]
        year  = parts[-4]
        hour_datetime = datetime.strptime(f"{year}-{month}-{day} {hour}", "%Y-%m-%d %H")
        datetime_str = hour_datetime.strftime("%#m/%#d/%Y %H:00")
    except Exception as e:
        print(f"Could not parse datetime for {hour_path}, skipping. ({e})")
        files_skipped += len(nc_files)
        continue

    if last_datetime is not None and hour_datetime <= last_datetime:
        continue

    file_data = []  # per-file dict of {col_name: array over pixels}
    file_qf   = []  # per-file qf array over pixels

    for fp in nc_files:
        try:
            with h5py.File(fp, "r") as f:
                qf_raw = f["product/main_data_quality_flag"][:]
                if qf_raw.ndim == 3:
                    qf_raw = qf_raw[0]
                qf_vals = qf_raw[lat_idx, lon_idx].astype(float)
                qf_vals[qf_vals == -9999] = np.nan

                data = {}
                for hdf_key, col_name in fields.items():
                    try:
                        arr = f[hdf_key][:]
                        if arr.ndim == 3:
                            arr = arr[0]
                        data[col_name] = arr[lat_idx, lon_idx].astype(float)
                    except Exception:
                        data[col_name] = np.full(n_pixels, np.nan)
        except Exception as e:
            print(f"Failed to load {fp}, skipping file. ({e})")
            files_skipped += 1
            continue

        file_data.append(data)
        file_qf.append(qf_vals)

    if not file_data:
        continue

    if len(file_data) > 1:
        print(f"{hour_path}: {len(file_data)} files found this hour, averaging.", flush=True)

    # Average across files if more than one this hour (nanmean so a
    # NaN in one file doesn't wipe out a valid value in the other)
    avg_data = {
        col_name: np.nanmean(np.vstack([d[col_name] for d in file_data]), axis=0)
        for col_name in fields.values()
    }
    avg_qf = np.nanmean(np.vstack(file_qf), axis=0)

    df = pd.DataFrame({
        "Pixel Number": pixel_ids,
        "Latitude":     lat_target,
        "Longitude":    lon_target,
        "Datetime":     datetime_str,
        "main_data_quality_flag": avg_qf,
    })
    for col_name, arr in avg_data.items():
        df[col_name] = arr

    # If NO2 value is NaN, set QF to NaN so it gets excluded downstream
    df.loc[df["vertical_column_troposphere"].isna(), "main_data_quality_flag"] = np.nan

    df.to_hdf(
        out_h5,
        key="no2",
        mode="a",
        append=True,
        format="table",
        index=False,
        complib="blosc",
        complevel=5,
        min_itemsize={"Datetime": 20},
    )

    files_processed += len(nc_files)
    total_rows += len(df)
    print(f"[{hour_datetime}] {len(nc_files)} file(s) — {len(df)} pixels written", flush=True)

# -------------------------------
# SUMMARY
# -------------------------------
print(f"\nDone! Output saved to {out_h5}")
print(f"Total files processed: {files_processed}")
print(f"Total files skipped:   {files_skipped}")
print(f"Total rows written:    {total_rows}")