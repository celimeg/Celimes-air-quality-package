"""Regridding and footprint builders against the original notebook code."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import fake_goes  # noqa: E402
import synthetic as syn  # noqa: E402

from celimes_air_quality_package import (FootprintGrid, build_goes_footprint, regrid_goes_aod,  # noqa: E402
                                         regrid_tempo)
from celimes_air_quality_package.footprints import (abi_latlon_to_scan,  # noqa: E402
                                                    abi_scan_to_latlon)

REF = os.path.join(os.path.dirname(__file__), "reference")


def test_abi_navigation_roundtrip():
    lat = np.array([39.53, 34.1, 45.8, 40.04])
    lon = np.array([-119.81, -118.1, -121.9, -105.24])
    x, y = abi_latlon_to_scan(lat, lon, fake_goes.GOES_WEST)
    la, lo = abi_scan_to_latlon(x, y, fake_goes.GOES_WEST)
    np.testing.assert_allclose(la, lat, atol=1e-7)
    np.testing.assert_allclose(lo, lon, atol=1e-7)


def test_footprint_grid_any_numbering():
    fp = syn.footprint(7, 5)
    shuffled = fp.sample(frac=1, random_state=0).reset_index(drop=True)
    g1, g2 = FootprintGrid.from_footprint(fp), FootprintGrid.from_footprint(shuffled)
    assert g1.shape == (7, 5)
    np.testing.assert_array_equal(g1.pixel_ids, g2.pixel_ids)
    assert g1.pixel_ids[0, 0] == 1            # north-west corner is pixel 1
    assert g1.pixel_ids[1, 0] == 2            # numbering runs down the first column


@pytest.fixture(scope="module")
def goes_site(tmp_path_factory):
    root = tmp_path_factory.mktemp("goes")
    x, y = fake_goes.make_abi_grid(39.25, 39.75, -120.1, -119.5)
    hours = pd.date_range("2022-07-01 18:00", periods=3, freq="h")
    fake_goes.make_goes_tree(str(root), x, y, hours)
    return str(root), hours


def test_goes_footprint_matches_notebook(goes_site, tmp_path):
    root, hours = goes_site
    nc_dir = os.path.join(root, f"{hours[0]:%Y}", f"{hours[0].dayofyear:03d}", f"{hours[0]:%H}")
    ref_csv = tmp_path / "ref_fp.csv"
    src = open(os.path.join(REF, "goes_footprint_notebook.py")).read()
    src = src.replace(src[src.index("GOES_DIR = "):src.index("\n", src.index("GOES_DIR = "))], f"GOES_DIR = r'{nc_dir}'")
    src = src.replace(src[src.index("GEOMETRY_CSV = "):src.index("\n", src.index("GEOMETRY_CSV = "))],
                      f"GEOMETRY_CSV = r'{ref_csv}'")
    src = src.replace(src[src.index("TARGET_LAT, TARGET_LON = "):src.index("\n", src.index("TARGET_LAT, TARGET_LON = "))],
                      "TARGET_LAT, TARGET_LON = 39.5, -119.8")
    src = src.replace("BOX_HALF_WIDTH_KM = 40.0", "BOX_HALF_WIDTH_KM = 15.0")
    exec(src, {})
    ref = pd.read_csv(ref_csv)
    new = build_goes_footprint(nc_dir, center=(39.5, -119.8), half_width_km=15.0)
    assert len(new) == len(ref) > 50
    np.testing.assert_array_equal(new["Pixel Number"], ref["Pixel Number"])
    np.testing.assert_allclose(new[["Latitude", "Longitude"]], ref[["Latitude", "Longitude"]], atol=1e-9)


def test_goes_regrid_matches_notebook(goes_site, tmp_path):
    root, hours = goes_site
    fp = syn.footprint(10, 12, lat0=39.35, lon0=-120.0, dlat=0.027, dlon=0.031)
    fp_csv = tmp_path / "fp.csv"
    fp.to_csv(fp_csv, index=False)

    # Reference notebook (Linux strftime, local paths)
    out_dir = tmp_path / "ref"
    src = open(os.path.join(REF, "goes_regrid_notebook.py")).read()
    src = (src.replace('year = "2022"', f'year = "{hours[0].year}"')
              .replace('base_path = f"Z:/Nasa_Project/GOES_ABI/{year}"', f'base_path = r"{root}/{hours[0].year}"')
              .replace(src[src.index("centroid_csv = "):src.index("\n", src.index("centroid_csv = "))], f"centroid_csv = r'{fp_csv}'")
              .replace(src[src.index("output_path = "):src.index("\n", src.index("output_path = "))], f"output_path = r'{out_dir}'")
              .replace("%#m/%#d/%Y", "%m/%d/%Y"))
    exec(src, {})
    ref = pd.read_csv(next(out_dir.glob("*.csv")))

    out = tmp_path / "new.csv"
    regrid_goes_aod(root, str(fp_csv), str(out), "2022-07-01", "2022-07-01")
    new = pd.read_csv(out)
    assert len(new) == len(ref) == 3 * len(fp)
    a = ref.sort_values(["Datetime", "Pixel Number"]).reset_index(drop=True)
    b = new.sort_values(["Datetime", "Pixel Number"]).reset_index(drop=True)
    for col in ["Weighted_AOD", "AOD_DQF0", "Weighted_DQF", "avg_DQF", "pct_DQF0"]:
        np.testing.assert_allclose(b[col], a[col], rtol=1e-9, equal_nan=True, err_msg=col)
    assert b["Weighted_AOD"].notna().mean() > 0.9

    # resuming does not duplicate hours
    regrid_goes_aod(root, str(fp_csv), str(out), "2022-07-01", "2022-07-01")
    assert len(pd.read_csv(out)) == len(new)


def _fake_tempo_tree(root, lat_axis, lon_axis, hours, seed=0):
    import h5py
    rng = np.random.default_rng(seed)
    for ts in hours:
        d = os.path.join(root, f"{ts:%Y}", f"{ts:%m}", f"{ts:%d}", f"{ts:%H}")
        os.makedirs(d, exist_ok=True)
        with h5py.File(os.path.join(d, f"TEMPO_NO2_L3_V03_{ts:%Y%m%dT%H%M%S}Z_S001.nc"), "w") as f:
            f["latitude"] = lat_axis
            f["longitude"] = lon_axis
            shape = (1, len(lat_axis), len(lon_axis))
            qf = rng.choice([0, 1, 2], shape).astype("int16")
            qf[rng.random(shape) < 0.05] = -9999
            f["product/main_data_quality_flag"] = qf
            trop = 3e15 + 1e15 * rng.random(shape)
            trop[rng.random(shape) < 0.1] = np.nan
            f["product/vertical_column_troposphere"] = trop
            f["product/vertical_column_stratosphere"] = 3e15 + 1e14 * rng.random(shape)
            f["support_data/eff_cloud_fraction"] = rng.random(shape)


def test_tempo_native_matches_notebook(tmp_path):
    lat_axis = np.round(np.arange(39.0, 40.0, 0.02) + 0.01, 6)
    lon_axis = np.round(np.arange(-120.4, -119.4, 0.02) + 0.01, 6)
    hours = pd.date_range("2024-01-03 17:00", periods=3, freq="h")
    root = tmp_path / "tempo"
    _fake_tempo_tree(str(root), lat_axis, lon_axis, hours)
    from celimes_air_quality_package import build_tempo_footprint
    fp = build_tempo_footprint(str(root), center=(39.5, -119.9), half_width_km=10)
    fp_csv = tmp_path / "fp.csv"
    fp.to_csv(fp_csv, index=False)

    ref_dir = tmp_path / "ref"
    src = open(os.path.join(REF, "tempo_native_notebook.py")).read()
    for var, val in [("organized_root", root), ("footprint_csv", fp_csv), ("output_path", ref_dir)]:
        i = src.index(f"{var} ")
        src = src.replace(src[i:src.index("\n", i)], f"{var} = r'{val}'")
    src = src.replace("%#m/%#d/%Y", "%m/%d/%Y")
    exec(src, {})
    ref = pd.read_hdf(ref_dir / "Resampled_TEMPO_NO2_LV.h5", key="no2")

    out = tmp_path / "new.h5"
    regrid_tempo(str(root), str(fp_csv), str(out), method="native")
    new = pd.read_hdf(out, key="no2")
    assert len(new) == len(ref) == 3 * len(fp)
    a = ref.sort_values(["Datetime", "Pixel Number"]).reset_index(drop=True)
    b = new.sort_values(["Datetime", "Pixel Number"]).reset_index(drop=True)
    for col in ["main_data_quality_flag", "vertical_column_troposphere", "vertical_column_stratosphere",
                "eff_cloud_fraction"]:
        np.testing.assert_allclose(b[col], a[col], rtol=1e-12, equal_nan=True, err_msg=col)


def test_tempo_area_mode_conserves_constant_field(tmp_path):
    """Area mode onto a coarser grid: a uniform field must come back uniform (no NaN bias)."""
    import h5py
    lat_axis = np.round(np.arange(39.0, 40.0, 0.02) + 0.01, 6)
    lon_axis = np.round(np.arange(-120.4, -119.4, 0.02) + 0.01, 6)
    root = tmp_path / "tempo"
    ts = pd.Timestamp("2024-01-03 18:00")
    d = root / f"{ts:%Y}" / f"{ts:%m}" / f"{ts:%d}" / f"{ts:%H}"
    d.mkdir(parents=True)
    shape = (1, len(lat_axis), len(lon_axis))
    trop = np.full(shape, 5e15)
    trop[:, ::3, ::2] = np.nan                      # missing retrievals must not drag the mean down
    with h5py.File(d / "TEMPO_NO2_L3_V03_20240103T180000Z_S001.nc", "w") as f:
        f["latitude"], f["longitude"] = lat_axis, lon_axis
        f["product/main_data_quality_flag"] = np.zeros(shape, "int16")
        f["product/vertical_column_troposphere"] = trop
    fp = syn.footprint(8, 9, lat0=39.3, lon0=-120.1, dlat=0.027, dlon=0.031)
    out = tmp_path / "area.h5"
    regrid_tempo(str(root), fp, str(out), method="area")
    res = pd.read_hdf(out, key="no2")
    np.testing.assert_allclose(res["vertical_column_troposphere"], 5e15, rtol=1e-12)
    np.testing.assert_allclose(res["vertical_column_troposphere_weighted"], 5e15, rtol=1e-12)
    assert (res["pixel_count"] > 0).all()
