"""Load regridded satellite tables and merge them with meteorology.

Everything comes back as a flat DataFrame keyed by ``Pixel Number`` and a
naive-UTC ``Datetime``, which is what the gap-filling functions take.
"""

import os

import pandas as pd

from .footprints import load_footprint
from .hrrr import load_hrrr
from .utils import parse_datetime, read_table


def _filter_time(df, start, end):
    if start is not None:
        df = df[df["Datetime"] >= pd.Timestamp(start)]
    if end is not None:
        e = pd.Timestamp(end)
        if e == e.normalize():
            e += pd.Timedelta(hours=23)
        df = df[df["Datetime"] <= e]
    return df


def load_goes_regridded(path, footprint=None, start=None, end=None):
    """Load regridded GOES AOD.

    Accepts the per-site CSV/HDF5 tables written by regrid_goes_aod (or the
    older notebook scripts, whose Datetime strings look like "8/6/2020 19:00"),
    or a gridded NetCDF with dims (time, lat, lon). For the gridded form,
    ``footprint`` is needed to attach Pixel Number by matching lat/lon.
    """
    if path.lower().endswith(".nc"):
        if footprint is None:
            raise ValueError("A gridded GOES NetCDF needs footprint= to attach Pixel Number")
        import xarray as xr

        with xr.open_dataset(path) as ds:
            df = ds.to_dataframe().reset_index()
        df = df.rename(columns={"time": "Datetime", "lat": "Latitude", "lon": "Longitude"})
        fp = load_footprint(footprint)
        for c in ("Latitude", "Longitude"):
            df[c] = df[c].round(5)
            fp[c] = fp[c].round(5)
        df = df.merge(fp, on=["Latitude", "Longitude"], how="left")
        n_bad = df["Pixel Number"].isna().sum()
        if n_bad:
            print(f"WARNING: {n_bad} rows did not match a footprint pixel (lat/lon precision?)")
        df = df.dropna(subset=["Pixel Number"])
    else:
        df = read_table(path)
    df["Datetime"] = parse_datetime(df["Datetime"])
    df["Pixel Number"] = df["Pixel Number"].astype(int)
    return _filter_time(df, start, end).reset_index(drop=True)


def load_tempo_regridded(path, key=None, start=None, end=None):
    """Load a regridded TEMPO table (HDF5 written by regrid_tempo or the notebooks, or CSV)."""
    df = read_table(path, key=key)
    df["Datetime"] = parse_datetime(df["Datetime"])
    df["Pixel Number"] = df["Pixel Number"].astype(int)
    return _filter_time(df, start, end).reset_index(drop=True)


def merge_with_met(sat_df, met_df, how="inner"):
    """Join satellite rows to meteorology on Datetime + Pixel Number.

    Latitude/Longitude are taken from the satellite table; any other
    column present in both gets a ``_met`` suffix on the meteorology side.
    """
    met = met_df.drop(columns=[c for c in ("Latitude", "Longitude") if c in met_df], errors="ignore")
    sat = sat_df.copy()
    sat["Pixel Number"] = sat["Pixel Number"].astype(int)
    met["Pixel Number"] = met["Pixel Number"].astype(int)
    merged = sat.merge(met, on=["Datetime", "Pixel Number"], how=how, suffixes=("", "_met"))
    print(f"Satellite rows: {len(sat):,}  Met rows: {len(met):,}  Merged: {len(merged):,}")
    return merged


def combine_goes_hrrr(goes_path, hrrr_nc, footprint=None, start=None, end=None):
    """Load regridded GOES AOD and HRRR for a site and merge them (inner join).

    Returns (merged, hrrr_df). hrrr_df is the full pre-merge HRRR table,
    which the NO2 filler uses for its similarity statistics.
    """
    goes = load_goes_regridded(goes_path, footprint, start, end)
    met = load_hrrr(hrrr_nc, start, end)
    return merge_with_met(goes, met), met


def combine_tempo_hrrr(tempo_path, hrrr_nc, key=None, start=None, end=None):
    """Load regridded TEMPO and HRRR for a site and merge them (inner join).

    Returns (merged, hrrr_df).
    """
    tempo = load_tempo_regridded(tempo_path, key, start, end)
    met = load_hrrr(hrrr_nc, start, end)
    return merge_with_met(tempo, met), met


def load_many(paths, loader, **kw):
    """Concatenate several files with the same loader (e.g. one HRRR file per year)."""
    frames = [loader(p, **kw) for p in paths if os.path.exists(p)]
    if not frames:
        raise FileNotFoundError(f"None of these exist: {paths}")
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["Pixel Number", "Datetime"]).reset_index(drop=True)
