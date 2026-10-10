# Verbatim from creating_goes_footprints.ipynb. Reference for tests.
import glob
import os
import numpy as np
import pandas as pd
import xarray as xr

# =======================================================================
# CONFIG
# =======================================================================
GOES_DIR = r"Z:\Nasa_Project\GOES_ABI\2022\002\14"   # folder containing at least one .nc file
GEOMETRY_CSV = r"Z:\Nasa_Project\GOES_ABI\resampled_pixels\Sites\Fresno\GOES_ABI_footprint_ncar.csv"

TARGET_LAT, TARGET_LON = 40.037530,-105.242770
BOX_HALF_WIDTH_KM = 40.0  # bounding box extends this far from target in each direction (N/S/E/W)


# =======================================================================
# ABI fixed-grid navigation
# =======================================================================
def get_projection_params(ds):
    proj = ds["goes_imager_projection"]
    r_eq = float(proj.semi_major_axis)
    r_pol = float(proj.semi_minor_axis)
    h = float(proj.perspective_point_height)
    lon_origin = float(proj.longitude_of_projection_origin)
    return dict(r_eq=r_eq, r_pol=r_pol, H=h + r_eq, lon_origin=lon_origin)


def fixed_grid_to_latlon(x, y, params):
    r_eq, r_pol, H = params["r_eq"], params["r_pol"], params["H"]
    lam0 = np.deg2rad(params["lon_origin"])

    a = np.sin(x) ** 2 + (np.cos(x) ** 2) * (
        np.cos(y) ** 2 + (r_eq ** 2 / r_pol ** 2) * np.sin(y) ** 2
    )
    b = -2 * H * np.cos(x) * np.cos(y)
    c = H ** 2 - r_eq ** 2
    disc = b ** 2 - 4 * a * c

    with np.errstate(invalid="ignore"):
        rs = (-b - np.sqrt(disc)) / (2 * a)

    sx = rs * np.cos(x) * np.cos(y)
    sy = -rs * np.sin(x)
    sz = rs * np.cos(x) * np.sin(y)

    with np.errstate(invalid="ignore"):
        lat = np.rad2deg(np.arctan((r_eq ** 2 / r_pol ** 2) * (sz / np.sqrt((H - sx) ** 2 + sy ** 2))))
        lon = np.rad2deg(lam0 - np.arctan(sy / (H - sx)))

    off_earth = disc < 0
    lat = np.where(off_earth, np.nan, lat)
    lon = np.where(off_earth, np.nan, lon)
    return lat, lon


def bbox_from_radius(target_lat, target_lon, half_width_km):
    half_lat_deg = half_width_km / 111.0
    half_lon_deg = half_width_km / (111.0 * np.cos(np.radians(target_lat)))
    return (
        target_lat - half_lat_deg, target_lat + half_lat_deg,
        target_lon - half_lon_deg, target_lon + half_lon_deg,
    )


# =======================================================================
# 1. Get native ABI pixel centroids in the bbox, just to measure the
#    real pixel spacing (in true meters). These native centroids are
#    NOT used as the output grid -- the ABI scan grid is rotated
#    relative to true north/east away from the sub-satellite point, so
#    its rows/columns are diagonal in lat/lon space, not straight. We
#    only borrow their spacing.
#
#    ABI pixel footprints are generally NOT square -- spacing differs
#    along the native row direction (roughly east-west) vs. the native
#    column direction (roughly north-south), so each is measured
#    separately rather than collapsed into one isotropic distance.
# =======================================================================
nc_file = sorted(glob.glob(os.path.join(GOES_DIR, "*.nc")))[0]
print(f"Using {nc_file}")

ds = xr.open_dataset(nc_file)
params = get_projection_params(ds)
x = ds["x"].values
y = ds["y"].values
ds.close()

xx, yy = np.meshgrid(x, y)
lat_grid, lon_grid = fixed_grid_to_latlon(xx, yy, params)

lat_min, lat_max, lon_min, lon_max = bbox_from_radius(TARGET_LAT, TARGET_LON, BOX_HALF_WIDTH_KM)
mask = (lat_grid >= lat_min) & (lat_grid <= lat_max) & (lon_grid >= lon_min) & (lon_grid <= lon_max)
rows_m, cols_m = np.where(mask)

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371008.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

# East-west (native row) spacing: distance from (row, col) to (row, col+1)
valid_ew = mask[rows_m, cols_m] & mask[rows_m, np.minimum(cols_m + 1, lat_grid.shape[1] - 1)] & (cols_m + 1 < lat_grid.shape[1])
r_ew, c_ew = rows_m[valid_ew], cols_m[valid_ew]
ew_dist_m = haversine_m(lat_grid[r_ew, c_ew], lon_grid[r_ew, c_ew], lat_grid[r_ew, c_ew + 1], lon_grid[r_ew, c_ew + 1])
lon_pixel_size_m = float(np.nanmean(ew_dist_m))

# North-south (native column) spacing: distance from (row, col) to (row+1, col)
valid_ns = mask[rows_m, cols_m] & mask[np.minimum(rows_m + 1, lat_grid.shape[0] - 1), cols_m] & (rows_m + 1 < lat_grid.shape[0])
r_ns, c_ns = rows_m[valid_ns], cols_m[valid_ns]
ns_dist_m = haversine_m(lat_grid[r_ns, c_ns], lon_grid[r_ns, c_ns], lat_grid[r_ns + 1, c_ns], lon_grid[r_ns + 1, c_ns])
lat_pixel_size_m = float(np.nanmean(ns_dist_m))

print(f"Native ABI pixel spacing in this region: EW (lon) = {lon_pixel_size_m:.1f} m, NS (lat) = {lat_pixel_size_m:.1f} m")

# =======================================================================
# 2. Build a TRUE regular lat/lon grid -- fixed longitude per column,
#    fixed latitude per row, each axis sized to its own real ABI pixel
#    spacing. This is guaranteed rectangular by construction, no skew
#    possible.
# =======================================================================
lat_step_deg = lat_pixel_size_m / 111000.0
lon_step_deg = lon_pixel_size_m / (111000.0 * np.cos(np.radians(TARGET_LAT)))

lat_vals = np.arange(lat_max, lat_min - lat_step_deg, -lat_step_deg)   # north -> south
lon_vals = np.arange(lon_min, lon_max + lon_step_deg, lon_step_deg)    # west -> east

# Trim -- drop any row/column whose full pixel footprint (center +/- half
# spacing) would extend past the bounding box, so no partial pixels remain.
half_lat = lat_step_deg / 2
half_lon = lon_step_deg / 2
lat_vals = lat_vals[(lat_vals - half_lat >= lat_min) & (lat_vals + half_lat <= lat_max)]
lon_vals = lon_vals[(lon_vals - half_lon >= lon_min) & (lon_vals + half_lon <= lon_max)]

if len(lat_vals) == 0 or len(lon_vals) == 0:
    raise ValueError(
        "Bounding box is too small relative to pixel spacing -- no complete pixels fit. "
        "Increase BOX_HALF_WIDTH_KM."
    )

records = []
pixel_number = 1
for lon in lon_vals:            # outer loop: longitude ascending (west -> east)
    for lat in lat_vals:        # inner loop: latitude descending (north -> south)
        records.append((pixel_number, lat, lon))
        pixel_number += 1

out = pd.DataFrame(records, columns=["Pixel Number", "Latitude", "Longitude"])

os.makedirs(os.path.dirname(GEOMETRY_CSV), exist_ok=True)
out.to_csv(GEOMETRY_CSV, index=False)
print(f"Saved {len(out)} pixels ({len(lon_vals)} cols x {len(lat_vals)} rows) to {GEOMETRY_CSV}")