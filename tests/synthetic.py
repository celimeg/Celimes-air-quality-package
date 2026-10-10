"""Synthetic site data shaped like the real regridded GOES/TEMPO + HRRR tables."""

import numpy as np
import pandas as pd

HRRR_COLS = ["u_wind_850mb_ms", "v_wind_850mb_ms", "u_wind_250mb_ms", "v_wind_250mb_ms",
             "temp_2m_K", "rh_2m_pct", "pbl_height_m", "shear_u_0_1km_per_s", "shear_v_0_1km_per_s"]


def footprint(n_rows=12, n_cols=10, lat0=39.30, lon0=-120.00, dlat=0.031, dlon=0.032):
    """Pixel numbering like the notebooks: lon ascending outer, lat descending inner."""
    lats = lat0 + dlat * np.arange(n_rows)[::-1]
    lons = lon0 + dlon * np.arange(n_cols)
    rows = []
    n = 1
    for lo in lons:
        for la in lats:
            rows.append((n, round(la, 6), round(lo, 6)))
            n += 1
    return pd.DataFrame(rows, columns=["Pixel Number", "Latitude", "Longitude"])


def _smooth(rng, shape, scale=1.0):
    z = rng.normal(size=shape)
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(z, 2) * scale


def hrrr_table(fp, hours, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    n = len(fp)
    for k, t in enumerate(hours):
        d = {"Pixel Number": fp["Pixel Number"].to_numpy(), "Datetime": t}
        base = {"u_wind_850mb_ms": 5, "v_wind_850mb_ms": -2, "u_wind_250mb_ms": 25, "v_wind_250mb_ms": 8,
                "temp_2m_K": 295, "rh_2m_pct": 30, "pbl_height_m": 1500,
                "shear_u_0_1km_per_s": 0.004, "shear_v_0_1km_per_s": -0.002}
        spread = {"pbl_height_m": 300, "temp_2m_K": 3, "rh_2m_pct": 8}
        for c in HRRR_COLS:
            # HRRR is ~3 km, sampled by nearest neighbour: pairs of pixels share values
            v = base[c] + spread.get(c, 2 if "wind" in c else 0.001) * rng.normal(size=n // 2 + 1)
            d[c] = np.repeat(v, 2)[:n] + 0.1 * k
        out.append(pd.DataFrame(d))
    df = pd.concat(out, ignore_index=True)
    df = df.merge(fp, on="Pixel Number")
    return df


def aod_table(fp, hours, seed=0, gap_frac=0.35):
    rng = np.random.default_rng(seed)
    nr = fp["Latitude"].nunique()
    nc = fp["Longitude"].nunique()
    lats = np.sort(fp["Latitude"].unique())
    lons = np.sort(fp["Longitude"].unique())
    r = np.searchsorted(lats, fp["Latitude"])
    c = np.searchsorted(lons, fp["Longitude"])
    out = []
    field = 0.15 + np.abs(_smooth(rng, (nr, nc), 0.4))
    for k, t in enumerate(hours):
        field = np.abs(field + _smooth(rng, (nr, nc), 0.03))
        aod = field[r, c]
        pct = rng.uniform(10, 100, len(fp))
        gaps = rng.random(len(fp)) < gap_frac
        # a contiguous cloud blob too
        blob = (np.hypot(r - (3 + k % 4), c - 5) < 2.5)
        pct[gaps | blob] = rng.uniform(0, 9.9, (gaps | blob).sum())
        pct[rng.random(len(fp)) < 0.05] = np.nan
        out.append(pd.DataFrame({
            "Pixel Number": fp["Pixel Number"], "Latitude": fp["Latitude"], "Longitude": fp["Longitude"],
            "Datetime": t, "Weighted_AOD": np.where(np.isnan(pct), np.nan, aod),
            "AOD_DQF0": aod, "Weighted_DQF": 0.8, "avg_DQF": 0.5, "pct_DQF0": pct,
        }))
    return pd.concat(out, ignore_index=True)


def tempo_table(fp, hours, seed=3):
    rng = np.random.default_rng(seed)
    nr = fp["Latitude"].nunique()
    nc = fp["Longitude"].nunique()
    lats = np.sort(fp["Latitude"].unique())
    lons = np.sort(fp["Longitude"].unique())
    r = np.searchsorted(lats, fp["Latitude"])
    c = np.searchsorted(lons, fp["Longitude"])
    out = []
    trop = 3e15 + 2e15 * np.abs(_smooth(rng, (nr, nc), 3))
    strat = 3e15 + 1e14 * _smooth(rng, (nr, nc), 1)
    for k, t in enumerate(hours):
        qf = rng.choice([0, 0, 0, 1, 2], len(fp)).astype(float)
        qf[np.hypot(r - 5, c - (2 + k % 5)) < 2.2] = 1
        qf[rng.random(len(fp)) < 0.04] = np.nan
        tv = trop[r, c] * (1 + 0.05 * k) + 1e14 * rng.normal(size=len(fp))
        tv[rng.random(len(fp)) < 0.06] = -2e14          # QC 0 but negative
        sv = strat[r, c] + 5e13 * rng.normal(size=len(fp))
        out.append(pd.DataFrame({
            "Pixel Number": fp["Pixel Number"], "Latitude": fp["Latitude"], "Longitude": fp["Longitude"],
            "Datetime": t, "main_data_quality_flag": qf,
            "vertical_column_troposphere": tv, "vertical_column_stratosphere": sv,
            "eff_cloud_fraction": rng.uniform(0, 0.4, len(fp)),
        }))
    return pd.concat(out, ignore_index=True)
