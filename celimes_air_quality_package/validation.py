"""Validate GOES AOD (raw and gap-filled) against AERONET.

Workflow for one site::

    ae      = load_aeronet_lev20(r"...\\20200101_20241231_NCAR.lev20")
    raw     = extract_raw_goes_at_site(goes_dir, 40.0375, -105.2428, "2020-01-01", "2024-12-31")
    filled  = extract_filled_at_site(r"...\\NCAR_Gap_filled.csv", 40.0375, -105.2428)
    res     = validate_site(raw, filled, ae, threshold=25)
    res["stats_raw"], res["stats_filled"]

Each site series has one row per hour: Datetime, mean_AOD, percent_valid
(percent of pixels inside the radius that were usable that hour).
"""

import glob
import os

import numpy as np
import pandas as pd

from .footprints import abi_projection, abi_scan_to_latlon
from .utils import haversine_km, parse_datetime, read_table

AERONET_COLUMNS = [
    "Date(dd:mm:yyyy)", "Time(hh:mm:ss)", "Day_of_Year",
    "Site_Latitude(Degrees)", "Site_Longitude(Degrees)", "Site_Elevation(m)",
    "AOD_675nm", "AOD_500nm", "AOD_440nm",
    "Triplet_Variability_675", "Triplet_Variability_500", "Triplet_Variability_440",
    "440-675_Angstrom_Exponent", "Solar_Zenith_Angle(Degrees)", "Optical_Air_Mass",
    "Sensor_Temperature(Degrees_C)", "Precipitable_Water(cm)",
    "Triplet_Variability_Precipitable_Water(cm)", "Data_Quality_Level",
    "AERONET_Instrument_Number", "AERONET_Site_Name",
]
_HOURLY_COLUMNS = [
    "Day_of_Year", "AOD_675nm", "AOD_500nm", "AOD_440nm",
    "Triplet_Variability_675", "Triplet_Variability_500", "Triplet_Variability_440",
    "Solar_Zenith_Angle(Degrees)", "Optical_Air_Mass", "Sensor_Temperature(Degrees_C)",
    "Precipitable_Water(cm)", "Triplet_Variability_Precipitable_Water(cm)",
    "estimated_AOD_550nm", "estimated_AOD_550nm_from_500nm",
]


# ------------------------------------------------------------------
# AERONET
# ------------------------------------------------------------------

def angstrom_exponent(aod1, aod2, wl1, wl2):
    return -np.log(aod1 / aod2) / np.log(wl1 / wl2)


def aeronet_aod550(df):
    """Add estimated_AOD_550nm (from 440/675) and estimated_AOD_550nm_from_500nm (500/675).

    Missing bands are -999 in AERONET files. The 500 nm estimate falls back
    to AOD_675nm when only 675 is available.
    """
    a440, a500, a675 = (df[c].astype(float) for c in ("AOD_440nm", "AOD_500nm", "AOD_675nm"))
    ok675, ok440, ok500 = a675 != -999, a440 != -999, a500 != -999
    with np.errstate(invalid="ignore", divide="ignore"):
        e1 = a675 * (550 / 675) ** -angstrom_exponent(a675, a440, 675, 440)
        e2 = a500 * (550 / 500) ** -angstrom_exponent(a675, a500, 675, 500)
    df = df.copy()
    df["estimated_AOD_550nm"] = np.where(ok675 & ok440, e1, np.nan)
    df["estimated_AOD_550nm_from_500nm"] = np.where(ok675 & ok500, e2, np.where(ok675, a675, np.nan))
    return df


def load_aeronet_lev20(path, min_valid_per_hour=2):
    """Read an AERONET V3 .lev20 file and return hourly means (UTC, floored to the hour).

    Hours with fewer than min_valid_per_hour valid AOD550 estimates are dropped.
    """
    data = pd.read_csv(path, skiprows=6, low_memory=False)
    data = data[[c for c in AERONET_COLUMNS if c in data.columns]].copy()
    data["Datetime"] = pd.to_datetime(data["Date(dd:mm:yyyy)"] + " " + data["Time(hh:mm:ss)"],
                                      format="%d:%m:%Y %H:%M:%S")
    data = aeronet_aod550(data).replace(-999, np.nan)
    data["Hour"] = data["Datetime"].dt.floor("h")
    cols = [c for c in _HOURLY_COLUMNS if c in data.columns]
    g = data.groupby("Hour")
    hourly = g[cols].mean()
    n_valid = g["estimated_AOD_550nm"].count()
    hourly = hourly[n_valid >= min_valid_per_hour].reset_index().rename(columns={"Hour": "Datetime"})
    return hourly.dropna(subset=["estimated_AOD_550nm"]).reset_index(drop=True)


def load_aeronet_hourly_csv(path):
    """Read an already-averaged hourly AERONET CSV (Datetime, estimated_AOD_550nm, ...)."""
    df = pd.read_csv(path)
    df["Datetime"] = parse_datetime(df["Datetime"])
    return df.replace(-999, np.nan).dropna(subset=["estimated_AOD_550nm"])


# ------------------------------------------------------------------
# GOES series at a site
# ------------------------------------------------------------------

def extract_raw_goes_at_site(goes_dir, lat, lon, start, end, radius_km=7.5, out_csv=None):
    """Hourly mean of native (not regridded) DQF=0 ABI AOD within radius_km of a site.

    goes_dir is laid out goes_dir/YYYY/DOY/HH/*.nc. All scans in the hour are
    pooled. percent_valid = DQF=0 retrievals / all pixel-scans inside the
    radius x 100. If out_csv is given, rows are appended and a rerun resumes.
    """
    import xarray as xr

    done = set()
    if out_csv and os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(parse_datetime(prev[_dtcol(prev)]))
    nav = {}
    rows = []
    hours = pd.date_range(pd.Timestamp(start).normalize(),
                          pd.Timestamp(end).normalize() + pd.Timedelta(hours=23), freq="h")
    for ts in hours:
        if ts in done:
            continue
        files = sorted(glob.glob(os.path.join(goes_dir, f"{ts:%Y}", f"{ts.dayofyear:03d}", f"{ts:%H}", "*.nc")))
        if not files:
            continue
        vals, total = [], 0
        for f in files:
            try:
                with xr.open_dataset(f) as ds:
                    if "AOD" not in ds or "DQF" not in ds:
                        continue
                    key = (ds.sizes["y"], ds.sizes["x"],
                           float(ds["goes_imager_projection"].attrs["longitude_of_projection_origin"]))
                    if key not in nav:
                        la, lo = abi_scan_to_latlon(*np.meshgrid(ds["x"].values, ds["y"].values), abi_projection(ds))
                        with np.errstate(invalid="ignore"):
                            box = (np.abs(la - lat) <= 0.1) & (np.abs(lo - lon) <= 0.1)
                        r, c = np.where(box)
                        inside = haversine_km(la[r, c], lo[r, c], lat, lon) <= radius_km
                        nav[key] = (r[inside], c[inside])
                    r, c = nav[key]
                    aod = ds["AOD"].values[r, c]
                    dqf = ds["DQF"].values[r, c]
            except OSError:
                continue
            total += len(r)
            vals.append(aod[dqf == 0])
        if total == 0:
            continue
        v = np.concatenate(vals) if vals else np.array([])
        row = {"Datetime": ts,
               "mean_AOD": np.nanmean(v) if v.size else np.nan,
               "percent_valid": v.size / total * 100}
        rows.append(row)
        if out_csv:
            pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False,
                                       date_format="%Y-%m-%d %H:%M:%S")
    out = pd.DataFrame(rows, columns=["Datetime", "mean_AOD", "percent_valid"])
    if out_csv and done:
        out = load_site_series(out_csv)
    return out


def extract_filled_at_site(filled, lat, lon, radius_km=7.5, use="filled", fill_flag=200,
                           value_col="filled_AOD", flag_col="pct_DQF0_filled", legacy_box=False):
    """Hourly mean of gap-filled AOD within radius_km of a site.

    Parameters
    ----------
    filled : str or DataFrame
        Output of goes_aod_gap_filling.
    use : "filled" or "all"
        "filled" (the original notebook's choice) averages only pixels that
        were gap filled (flag == fill_flag), so it validates the filled
        values on their own. "all" averages every pixel with a value, i.e.
        the complete product a user would get.
    legacy_box : bool
        The original notebook pre-filtered with a box of radius_km/111
        degrees in both latitude and longitude, which clips the circle in
        longitude (about +/-5.7 km instead of 7.5 km at 40N). True
        reproduces that; False uses the full circle.
    """
    df = read_table(filled) if isinstance(filled, (str, os.PathLike)) else filled.copy()
    df["Datetime"] = parse_datetime(df["Datetime"])
    if legacy_box:
        d = radius_km / 111.0
        df = df[(df["Latitude"].between(lat - d, lat + d)) & (df["Longitude"].between(lon - d, lon + d))]
    df = df[haversine_km(df["Latitude"].to_numpy(), df["Longitude"].to_numpy(), lat, lon) <= radius_km]
    df = df.assign(hour=df["Datetime"].dt.floor("h"))
    if use == "filled":
        good = (df[flag_col] == fill_flag) & df[value_col].notna()
    elif use == "all":
        good = df[value_col].notna()
    else:
        raise ValueError("use must be 'filled' or 'all'")
    g = df.assign(_v=df[value_col].where(good), _ok=good).groupby("hour")
    out = pd.DataFrame({"mean_AOD": g["_v"].mean(), "percent_valid": g["_ok"].mean() * 100})
    return out.reset_index().rename(columns={"hour": "Datetime"})


def _dtcol(df):
    return "Datetime" if "Datetime" in df.columns else "datetime"


def load_site_series(path):
    """Read a saved hourly site series (accepts 'datetime' or 'Datetime')."""
    df = pd.read_csv(path)
    df = df.rename(columns={"datetime": "Datetime"})
    df["Datetime"] = parse_datetime(df["Datetime"])
    return df.replace(-999, np.nan)


# ------------------------------------------------------------------
# Collocation and statistics
# ------------------------------------------------------------------

def collocate(goes_df, ae_df):
    """Match hourly GOES and AERONET on Datetime -> Datetime, AOD_GOES, AOD_AERONET."""
    g = goes_df[["Datetime", "mean_AOD"]].dropna(subset=["mean_AOD"])
    a = ae_df.groupby("Datetime", as_index=False)["estimated_AOD_550nm"].mean()
    m = g.merge(a, on="Datetime", how="inner")
    return m.rename(columns={"mean_AOD": "AOD_GOES", "estimated_AOD_550nm": "AOD_AERONET"})


def compute_stats(df):
    """Agreement statistics for collocated pairs.

    r2, rmse     from an ordinary least-squares fit of GOES on AERONET
                 (rmse is the residual RMSE about that fitted line, as in
                 the original notebooks)
    rmse_1to1    RMSE of GOES - AERONET (about the 1:1 line)
    bias         mean(GOES - AERONET)
    pct_ee       percent within the expected error envelope +/-(0.05 + 0.15 AOD_AERONET)
    """
    ae = df["AOD_AERONET"].to_numpy(float)
    goes = df["AOD_GOES"].to_numpy(float)
    n = len(ae)
    if n < 2:
        return dict(r2=np.nan, rmse=np.nan, rmse_1to1=np.nan, bias=np.nan, pct_ee=np.nan, pct=np.nan,
                    n=n, slope=np.nan, intercept=np.nan, mean_goes=np.nan, std_goes=np.nan,
                    mean_aeronet=np.nan, std_aeronet=np.nan)
    slope, intercept = np.polyfit(ae, goes, 1)
    pred = slope * ae + intercept
    ss_res = np.sum((goes - pred) ** 2)
    ss_tot = np.sum((goes - goes.mean()) ** 2)
    ee = 0.05 + 0.15 * ae
    pct_ee = float(np.mean(np.abs(goes - ae) <= ee) * 100)
    return dict(r2=1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                rmse=float(np.sqrt(ss_res / n)),
                rmse_1to1=float(np.sqrt(np.mean((goes - ae) ** 2))),
                bias=float(np.mean(goes - ae)), pct_ee=pct_ee, pct=pct_ee, n=n,
                slope=float(slope), intercept=float(intercept),
                mean_goes=float(goes.mean()), std_goes=float(goes.std()),
                mean_aeronet=float(ae.mean()), std_aeronet=float(ae.std()))


def validate_site(raw_df, filled_df, ae_df, threshold=25):
    """Filter both series to percent_valid > threshold, collocate, and compute stats."""
    out = {}
    for name, df in (("raw", raw_df), ("filled", filled_df)):
        sub = df[(df["percent_valid"] > threshold) & df["mean_AOD"].notna()]
        m = collocate(sub, ae_df)
        out[f"{name}_merged"] = m
        out[f"stats_{name}"] = compute_stats(m)
    return out


def threshold_sensitivity(raw_df, filled_df, ae_df, thresholds=(25, 50, 75), site=None):
    """One row per threshold with raw and filled R2, RMSE, EE% and N side by side."""
    rows = []
    for t in thresholds:
        r = validate_site(raw_df, filled_df, ae_df, t)
        sr, sf = r["stats_raw"], r["stats_filled"]
        rows.append({"Site": site, "Threshold (%)": t,
                     "Raw R2": sr["r2"], "Filled R2": sf["r2"],
                     "Raw RMSE": sr["rmse"], "Filled RMSE": sf["rmse"],
                     "Raw RMSE 1:1": sr["rmse_1to1"], "Filled RMSE 1:1": sf["rmse_1to1"],
                     "Raw EE%": sr["pct_ee"], "Filled EE%": sf["pct_ee"],
                     "Raw N": sr["n"], "Filled N": sf["n"]})
    return pd.DataFrame(rows).round({"Raw R2": 3, "Filled R2": 3, "Raw RMSE": 3, "Filled RMSE": 3,
                                     "Raw RMSE 1:1": 3, "Filled RMSE 1:1": 3, "Raw EE%": 1, "Filled EE%": 1})


def season_of(dt):
    """DJF/MAM/JJA/SON from datetimes (by month)."""
    m = pd.to_datetime(dt).dt.month
    return m.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                  6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})


def stats_by_aod_bin(merged, bins=(0, 1, 2, 3, 5)):
    """compute_stats within AERONET AOD bins (left-closed)."""
    labels = [f"{a}-{b}" for a, b in zip(bins[:-1], bins[1:])]
    cut = pd.cut(merged["AOD_AERONET"], bins=list(bins), labels=labels, right=False)
    rows = []
    for lab in labels:
        s = compute_stats(merged[cut == lab])
        rows.append({"Bin": lab, **s})
    return pd.DataFrame(rows)
