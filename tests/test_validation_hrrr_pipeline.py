"""Validation statistics, HRRR store round trip, helpers, and an end-to-end run."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import fake_goes  # noqa: E402
import synthetic as syn  # noqa: E402

import celimes_air_quality_package as cap  # noqa: E402
from celimes_air_quality_package.hrrr import _append_day, create_hrrr_store, last_stored_time  # noqa: E402
from celimes_air_quality_package.utils import parse_datetime  # noqa: E402
from celimes_air_quality_package.validation import aeronet_aod550, stats_by_aod_bin  # noqa: E402


def test_parse_datetime_mixed_formats():
    s = pd.Series(["8/6/2020 19:00", "2020-08-06 19:00:00", "2020-08-07", "08/06/2020 05:00"])
    out = parse_datetime(s)
    assert list(out) == [pd.Timestamp("2020-08-06 19:00"), pd.Timestamp("2020-08-06 19:00"),
                         pd.Timestamp("2020-08-07 00:00"), pd.Timestamp("2020-08-06 05:00")]


def test_goes_bucket_cutovers():
    assert cap.goes_bucket("2022-12-31 23:00") == "noaa-goes17"
    assert cap.goes_bucket("2023-01-04 17:00") == "noaa-goes17"
    assert cap.goes_bucket("2023-01-04 18:00") == "noaa-goes18"
    assert cap.goes_bucket("2025-04-07 14:00", "east") == "noaa-goes16"
    assert cap.goes_bucket("2025-04-07 15:00", "east") == "noaa-goes19"


def test_compute_stats_matches_sklearn():
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    rng = np.random.default_rng(0)
    ae = rng.gamma(2, 0.1, 500)
    goes = 0.9 * ae + 0.02 + rng.normal(0, 0.05, 500)
    s = cap.compute_stats(pd.DataFrame({"AOD_AERONET": ae, "AOD_GOES": goes}))
    m = LinearRegression().fit(ae.reshape(-1, 1), goes)
    assert np.isclose(s["r2"], m.score(ae.reshape(-1, 1), goes))
    assert np.isclose(s["rmse"], np.sqrt(mean_squared_error(goes, m.predict(ae.reshape(-1, 1)))))
    ee = (goes >= ae - (0.05 + 0.15 * ae)) & (goes <= ae + (0.05 + 0.15 * ae))
    assert np.isclose(s["pct_ee"], ee.mean() * 100)
    assert np.isclose(s["rmse_1to1"], np.sqrt(np.mean((goes - ae) ** 2)))
    assert len(stats_by_aod_bin(pd.DataFrame({"AOD_AERONET": ae, "AOD_GOES": goes}))) == 4


def test_aeronet_550_matches_notebook_rowwise():
    df = pd.DataFrame({"AOD_440nm": [0.2, -999, 0.3, 0.1], "AOD_500nm": [0.18, 0.15, -999, -999],
                       "AOD_675nm": [0.12, 0.1, 0.2, -999]})

    def nb_440(r):  # notebook functions, verbatim logic
        if r["AOD_675nm"] != -999 and r["AOD_440nm"] != -999:
            a = -np.log(r["AOD_675nm"] / r["AOD_440nm"]) / np.log(675 / 440)
            return r["AOD_675nm"] * (550 / 675) ** -a
        return np.nan

    def nb_500(r):
        if r["AOD_675nm"] != -999 and r["AOD_500nm"] != -999:
            a = -np.log(r["AOD_675nm"] / r["AOD_500nm"]) / np.log(675 / 500)
            return r["AOD_500nm"] * (550 / 500) ** -a
        elif r["AOD_675nm"] != -999:
            return r["AOD_675nm"]
        return np.nan

    out = aeronet_aod550(df)
    np.testing.assert_allclose(out["estimated_AOD_550nm"], df.apply(nb_440, axis=1), equal_nan=True)
    np.testing.assert_allclose(out["estimated_AOD_550nm_from_500nm"], df.apply(nb_500, axis=1), equal_nan=True)


def test_hrrr_store_roundtrip(tmp_path):
    fp = syn.footprint(4, 3)
    path = str(tmp_path / "h.nc")
    create_hrrr_store(path, fp, cap.HRRR_VARIABLES)
    day = pd.date_range("2023-08-01", periods=24, freq="h")
    data = {n: np.random.default_rng(1).normal(size=(24, len(fp))).astype("float32")
            for n, _, _ in cap.HRRR_VARIABLES}
    _append_day(path, list(day), data)
    assert last_stored_time(path) == pd.Timestamp("2023-08-01 23:00")
    df = cap.load_hrrr(path)
    assert len(df) == 24 * len(fp)
    row = df[(df["Datetime"] == day[5]) & (df["Pixel Number"] == fp["Pixel Number"][7])]
    assert np.isclose(row["pbl_height_m"].iloc[0], data["pbl_height_m"][5, 7])


def _write_hrrr_nc(path, fp, hours):
    tab = syn.hrrr_table(fp, hours)
    create_hrrr_store(path, fp, [(c, "", "") for c in syn.HRRR_COLS])
    for day, g in tab.groupby(tab["Datetime"].dt.normalize()):
        ts = sorted(g["Datetime"].unique())
        data = {c: np.stack([g[g["Datetime"] == t].set_index("Pixel Number").loc[fp["Pixel Number"], c].to_numpy()
                             for t in ts]) for c in syn.HRRR_COLS}
        _append_day(path, [pd.Timestamp(t) for t in ts], data)


def test_end_to_end_goes(tmp_path):
    """Synthetic ABI files -> footprint -> regrid -> HRRR -> merge -> gap fill -> site series."""
    x, y = fake_goes.make_abi_grid(39.25, 39.75, -120.1, -119.5)
    hours = pd.date_range("2022-07-01 17:00", periods=4, freq="h")
    goes_dir = tmp_path / "goes"
    fake_goes.make_goes_tree(str(goes_dir), x, y, hours, scans_per_hour=2, seed=3)
    nc = next(goes_dir.rglob("*.nc"))
    fp = cap.build_goes_footprint(str(nc), center=(39.5, -119.8), half_width_km=12)

    reg = tmp_path / "reg.csv"
    cap.regrid_goes_aod(str(goes_dir), fp, str(reg), "2022-07-01", "2022-07-01")
    hrrr_nc = tmp_path / "hrrr.nc"
    _write_hrrr_nc(str(hrrr_nc), fp, hours)

    merged, _ = cap.combine_goes_hrrr(str(reg), str(hrrr_nc))
    assert len(merged) == len(hours) * len(fp)
    out = tmp_path / "filled.csv"
    cap.goes_aod_gap_filling(merged, str(out), "2022-07-01", "2022-07-01", pct_dqf0_thresh=40)
    filled = pd.read_csv(out)
    assert filled["filled_AOD"].notna().mean() > filled["Weighted_AOD"].where(filled["pct_DQF0"] >= 40).notna().mean()
    assert (filled["pct_DQF0_filled"] == 200).any()

    # resume: a second call writes nothing new
    cap.goes_aod_gap_filling(merged, str(out), "2022-07-01", "2022-07-01", pct_dqf0_thresh=40)
    assert len(pd.read_csv(out)) == len(filled)

    site = cap.extract_filled_at_site(str(out), 39.5, -119.8, radius_km=7.5, use="all")
    assert len(site) == len(hours) and site["mean_AOD"].notna().all()
    raw = cap.extract_raw_goes_at_site(str(goes_dir), 39.5, -119.8, "2022-07-01", "2022-07-01")
    assert len(raw) == len(hours) and (raw["percent_valid"] > 0).all()
