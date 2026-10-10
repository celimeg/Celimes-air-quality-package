"""Shared helpers: datetime parsing, distances, solar geometry, table I/O."""

import os

import numpy as np
import pandas as pd

# Datetime formats seen across the pipeline's existing outputs. All are UTC.
#   "8/6/2020 19:00"        regridded GOES CSVs, TEMPO HDF5 tables
#   "08/06/2020 19:00"      AERONET raw extraction CSVs
#   "2020-08-06 19:00:00"   pandas default / gap-filled outputs
_KNOWN_FORMATS = ("%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")


def parse_datetime(values):
    """Parse a Series/array of datetime strings or datetimes to naive UTC Timestamps.

    Tries each known format in turn and only falls back to pandas inference
    for leftovers, so a column mixing "2020-08-06 19:00:00" and a bare
    "2020-08-06" (midnight rows written by pandas) parses correctly.
    Timezone-aware input is converted to UTC and made naive.
    """
    s = pd.Series(values) if not isinstance(values, pd.Series) else values
    if pd.api.types.is_datetime64_any_dtype(s):
        out = pd.to_datetime(s)
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_convert("UTC").dt.tz_localize(None)
        return out

    s_str = s.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    remaining = out.isna() & s.notna()
    for fmt in _KNOWN_FORMATS:
        if not remaining.any():
            break
        parsed = pd.to_datetime(s_str[remaining], format=fmt, errors="coerce")
        out.loc[parsed.index] = out.loc[parsed.index].fillna(parsed)
        remaining = out.isna() & s.notna()
    if remaining.any():
        parsed = pd.to_datetime(s_str[remaining], errors="coerce", utc=True)
        out.loc[parsed.index] = parsed.dt.tz_localize(None)
    return out


def to_utc_naive(ts):
    """Return a single timestamp as a naive UTC pd.Timestamp."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (R = 6371 km)."""
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def bbox_from_radius(target_lat, target_lon, half_width_km):
    """(lat_min, lat_max, lon_min, lon_max) for a box extending half_width_km each way."""
    half_lat = half_width_km / 111.0
    half_lon = half_width_km / (111.0 * np.cos(np.radians(target_lat)))
    return (target_lat - half_lat, target_lat + half_lat,
            target_lon - half_lon, target_lon + half_lon)


def solar_zenith(dt, lat, lon):
    """Solar zenith angle (degrees) at one UTC time and location (uses pvlib)."""
    import pvlib

    ts = pd.Timestamp(dt)
    idx = pd.DatetimeIndex([ts]) if ts.tzinfo is not None else pd.DatetimeIndex([ts], tz="UTC")
    sp = pvlib.solarposition.get_solarposition(idx, lat, lon)
    return float(sp["zenith"].iloc[0])


# ------------------------------------------------------------------
# Table I/O: CSV, HDF5 (pandas table), Parquet
# ------------------------------------------------------------------

def read_table(path, key=None, columns=None):
    """Read a CSV, HDF5 table (.h5/.hdf5) or Parquet file into a DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".h5", ".hdf5", ".hdf"):
        if key is None:
            with pd.HDFStore(path, mode="r") as store:
                keys = store.keys()
            if len(keys) != 1:
                raise ValueError(f"{path} has keys {keys}; pass key= to choose one")
            key = keys[0]
        return pd.read_hdf(path, key=key, columns=columns)
    if ext == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns, low_memory=False)


def append_table(df, path, key="data", min_itemsize=None):
    """Append rows to a CSV or HDF5 table, creating it (with header) if needed."""
    ext = os.path.splitext(path)[1].lower()
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    if ext in (".h5", ".hdf5", ".hdf"):
        df.to_hdf(path, key=key, mode="a", append=True, format="table", index=False,
                  complib="blosc", complevel=5, min_itemsize=min_itemsize)
    elif ext == ".csv":
        # Always write the full time. pandas otherwise writes a bare date when
        # every row in the chunk is at midnight (e.g. the 00 UTC hour), which
        # then needs special-case parsing downstream.
        df.to_csv(path, mode="a", header=not os.path.exists(path), index=False,
                  date_format="%Y-%m-%d %H:%M:%S")
    else:
        raise ValueError(f"append_table supports .csv and .h5, not {ext!r}")


def existing_datetimes(path, column="Datetime", key="data"):
    """Set of Timestamps already written to an output table (for resuming)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    try:
        col = read_table(path, key=key if path.lower().endswith((".h5", ".hdf5", ".hdf")) else None,
                         columns=[column])[column]
    except (KeyError, ValueError):
        return set()
    return set(parse_datetime(col).dropna().unique())
