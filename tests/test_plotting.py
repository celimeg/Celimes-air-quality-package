import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import synthetic as syn  # noqa: E402

from celimes_air_quality_package import plotting  # noqa: E402


def test_plots_return_figures_and_only_save_when_asked(tmp_path):
    rng = np.random.default_rng(0)
    t = pd.date_range("2021-01-01", periods=1300, freq="7h")
    ae = rng.gamma(2, 0.15, len(t))
    m = pd.DataFrame({"Datetime": t, "AOD_AERONET": ae, "AOD_GOES": ae * 0.9 + rng.normal(0, 0.05, len(t))})
    os.chdir(tmp_path)
    assert plotting.plot_validation_scatter(m, m) is not None
    assert len(plotting.plot_seasonal_scatter(m)) == 4
    assert plotting.plot_metric_by_aod_bin(m, "rmse") is not None
    fp = syn.footprint(6, 5)
    assert plotting.plot_footprint(fp, highlight_pixels=[1, 2, 3], target=(39.38, -119.94)) is not None
    hour = fp.assign(Datetime="2021-01-01 00:00:00", filled_AOD=rng.random(len(fp)))
    assert plotting.plot_hour(hour, "filled_AOD", "2021-01-01 00:00") is not None
    assert list(tmp_path.iterdir()) == []          # nothing written without save_path
    plotting.plot_metric_by_aod_bin(m, "r2", save_path=str(tmp_path / "x.png"), dpi=50)
    assert (tmp_path / "x.png").exists()
