"""The package's gap filling must reproduce the original notebook code.

The reference modules in tests/reference/ are the notebook cells copied
verbatim. Both implementations run on the same synthetic site and the
filled values and flags are compared pixel by pixel.
"""

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import synthetic as syn  # noqa: E402

from celimes_air_quality_package.gap_filling import goes_aod_gap_filling, tempo_no2_gap_filling  # noqa: E402

REF = os.path.join(os.path.dirname(__file__), "reference")


def _nb_datetime_str(ts):
    """The notebook's to_df_datetime_str: '6/5/2023 09:00'."""
    return ts.strftime("%m/%d/%Y %H:%M").replace("/0", "/").lstrip("0")


def _stub_plotting():
    for name in ["cartopy", "cartopy.crs", "cartopy.io", "cartopy.io.shapereader"]:
        sys.modules.setdefault(name, types.ModuleType(name))


@pytest.fixture(scope="module", params=[(12, 10, 0), (9, 15, 7), (20, 18, 42)],
                ids=["12x10", "9x15", "20x18"])
def site(request):
    n_rows, n_cols, seed = request.param
    fp = syn.footprint(n_rows, n_cols)
    fp.attrs["seed"] = seed
    hours = pd.date_range("2023-06-15 13:00", "2023-06-16 05:00", freq="h")  # dawn, day, dusk, night in Reno
    return fp, hours


def test_aod_matches_notebook(site, tmp_path):
    fp, hours = site
    merged = syn.aod_table(fp, hours, seed=fp.attrs['seed']).merge(
        syn.hrrr_table(fp, hours, seed=fp.attrs['seed'] + 1).drop(columns=["Latitude", "Longitude"]),
        on=["Pixel Number", "Datetime"])

    # --- reference: notebook code on string datetimes ---
    nb_df = merged.copy()
    nb_df["Datetime"] = nb_df["Datetime"].map(_nb_datetime_str)
    g = {"df": nb_df}
    exec(open(os.path.join(REF, "aod_notebook.py")).read(), g)
    g["target_point"], g["radius_km"] = (39.4, -119.85), 1e6
    ref = pd.concat([g["process_hour"](t.tz_localize("UTC"), g["lat_idx_master"], g["lon_idx_master"],
                                       g["master_shape"]) for t in hours], ignore_index=True)
    ref["Datetime"] = pd.to_datetime(ref["Datetime"], format="%m/%d/%Y %H:%M")

    # --- package ---
    out = tmp_path / "aod.csv"
    goes_aod_gap_filling(merged, str(out), hours[0], hours[-1])
    new = pd.read_csv(out, parse_dates=["Datetime"])

    a = ref.sort_values(["Datetime", "Pixel Number"]).reset_index(drop=True)
    b = new.sort_values(["Datetime", "Pixel Number"]).reset_index(drop=True)
    assert len(a) == len(b)
    assert (a["Pixel Number"].to_numpy() == b["Pixel Number"].to_numpy()).all()
    n_filled = int((a["pct_DQF0_filled"] == 200).sum())
    assert n_filled > 50, "test data should exercise the filling"
    np.testing.assert_allclose(b["filled_AOD"], a["filled_AOD"], rtol=1e-9, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(b["pct_DQF0_filled"], a["pct_DQF0_filled"], rtol=1e-12, equal_nan=True)
    # night hours (SZA > 70) are written without filling
    night = b["Datetime"] >= pd.Timestamp("2023-06-16 04:00")
    assert night.any() and b.loc[night, "filled_AOD"].isna().all()


def test_no2_matches_notebook(site, tmp_path):
    _stub_plotting()
    fp, hours = site
    hrrr = syn.hrrr_table(fp, hours, seed=fp.attrs['seed'] + 1)
    merged = syn.tempo_table(fp, hours, seed=fp.attrs['seed'] + 2).merge(hrrr.drop(columns=["Latitude", "Longitude"]),
                                              on=["Pixel Number", "Datetime"])

    # --- reference ---
    ref_out = tmp_path / "ref_no2.csv"
    g = {"df": merged.copy(), "hrrr_df": hrrr.copy(), "tempo": fp.copy()}
    src = open(os.path.join(REF, "no2_notebook.py")).read()
    exec(src, g)
    g["group_stats"], g["global_stats"] = g["compute_group_stats"](hrrr, merged)
    g["output_path"] = str(ref_out)
    done = set()
    for t in hours:
        g["process_hour"](t, done)
    ref = pd.read_csv(ref_out)
    from celimes_air_quality_package.utils import parse_datetime
    ref["Datetime"] = parse_datetime(ref["Datetime"])  # notebook writes 00z as a bare date

    # --- package ---
    out = tmp_path / "no2.csv"
    tempo_no2_gap_filling(merged, str(out), hours[0], hours[-1], hrrr_data=hrrr)
    new = pd.read_csv(out, parse_dates=["Datetime"])

    a = ref.sort_values(["Datetime", "Pixel Number"]).reset_index(drop=True)
    b = new.sort_values(["Datetime", "Pixel Number"]).reset_index(drop=True)
    assert len(a) == len(b)
    for col in ["vertical_column_troposphere", "vertical_column_stratosphere"]:
        assert int(a[f"{col}_qc_filled"].isin([200, 201]).sum()) > 50
        np.testing.assert_allclose(b[f"filled_{col}"], a[f"filled_{col}"], rtol=1e-9, equal_nan=True)
        np.testing.assert_allclose(b[f"{col}_qc_filled"], a[f"{col}_qc_filled"], rtol=1e-12, equal_nan=True)
