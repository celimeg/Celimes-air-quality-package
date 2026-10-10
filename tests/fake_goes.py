"""Write small synthetic ABI L2 AOD files with real GOES-West navigation."""

import os

import numpy as np
import xarray as xr

from celimes_air_quality_package.footprints import abi_latlon_to_scan

GOES_WEST = dict(r_eq=6378137.0, r_pol=6356752.31414, H=35786023.0 + 6378137.0, lon_origin=-137.2)


def make_abi_grid(lat_min, lat_max, lon_min, lon_max, step=56e-6, margin=0.003):
    xs, ys = abi_latlon_to_scan(np.array([lat_min, lat_min, lat_max, lat_max]),
                                np.array([lon_min, lon_max, lon_min, lon_max]), GOES_WEST)
    # Snap to a fixed-grid-like lattice so every file shares the same x/y
    x0 = np.floor((xs.min() - margin) / step) * step
    x1 = np.ceil((xs.max() + margin) / step) * step
    y0 = np.floor((ys.min() - margin) / step) * step
    y1 = np.ceil((ys.max() + margin) / step) * step
    x = np.arange(x0, x1, step) + step / 2
    y = np.arange(y1, y0, -step) - step / 2        # ABI y decreases with row
    return x, y


def write_abi_file(path, x, y, aod, dqf):
    ds = xr.Dataset(
        {"AOD": (("y", "x"), aod.astype("float32")), "DQF": (("y", "x"), dqf.astype("float32")),
         "goes_imager_projection": ((), np.int32(-2147483647), {
             "semi_major_axis": GOES_WEST["r_eq"], "semi_minor_axis": GOES_WEST["r_pol"],
             "perspective_point_height": GOES_WEST["H"] - GOES_WEST["r_eq"],
             "longitude_of_projection_origin": GOES_WEST["lon_origin"]})},
        coords={"x": ("x", x), "y": ("y", y)},
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.to_netcdf(path)


def make_goes_tree(root, x, y, hours, scans_per_hour=3, seed=0):
    """root/YYYY/DOY/HH/*.nc with random AOD (some NaN/negative) and DQF 0-3."""
    rng = np.random.default_rng(seed)
    for ts in hours:
        for k in range(scans_per_hour):
            aod = rng.gamma(2.0, 0.08, (len(y), len(x)))
            aod[rng.random(aod.shape) < 0.15] = np.nan
            aod[rng.random(aod.shape) < 0.02] = -0.05
            dqf = rng.choice([0, 0, 1, 2, 3], aod.shape).astype(float)
            dqf[rng.random(aod.shape) < 0.03] = np.nan
            d = os.path.join(root, f"{ts:%Y}", f"{ts.dayofyear:03d}", f"{ts:%H}")
            write_abi_file(os.path.join(d, f"OR_ABI-L2-AODC-M6_G18_s{ts:%Y%j%H}{k:02d}0000.nc"), x, y, aod, dqf)
