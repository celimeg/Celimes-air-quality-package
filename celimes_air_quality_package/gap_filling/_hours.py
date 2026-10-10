"""Fast per-hour access to a merged satellite + meteorology table."""

import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from ..utils import parse_datetime, solar_zenith


class HourIndex:
    """Groups a merged table by hour once and puts columns onto a fixed grid.

    The grid uses ascending latitude (row 0 = south) and ascending
    longitude, matching the notebooks' np.unique ordering.
    """

    def __init__(self, df, decimals=6):
        df = df.copy()
        df["Datetime"] = parse_datetime(df["Datetime"])
        self.df = df.reset_index(drop=True)
        lat_r = np.round(self.df["Latitude"].to_numpy(dtype=float), decimals)
        lon_r = np.round(self.df["Longitude"].to_numpy(dtype=float), decimals)
        self.lats = np.unique(lat_r)
        self.lons = np.unique(lon_r)
        self.shape = (len(self.lats), len(self.lons))
        self.row = np.searchsorted(self.lats, lat_r)
        self.col = np.searchsorted(self.lons, lon_r)
        self.groups = {pd.Timestamp(k): v for k, v in self.df.groupby("Datetime").indices.items()}
        self._sza = {}

    def rows(self, ts):
        """Integer row positions of this hour (empty array if none)."""
        return self.groups.get(pd.Timestamp(ts), np.array([], dtype=int))

    def grid(self, idx, column, values=None):
        """Put one column (or the given values) for rows idx onto the grid; NaN elsewhere."""
        g = np.full(self.shape, np.nan)
        v = self.df[column].to_numpy(dtype=float)[idx] if values is None else values
        g[self.row[idx], self.col[idx]] = v
        return g

    def sza(self, ts, lat, lon):
        key = (pd.Timestamp(ts), round(lat, 6), round(lon, 6))
        if key not in self._sza:
            self._sza[key] = solar_zenith(ts, lat, lon)
        return self._sza[key]


class LRU(OrderedDict):
    def __init__(self, maxsize=6):
        super().__init__()
        self.maxsize = maxsize

    def get_or(self, key, fn):
        if key in self:
            self.move_to_end(key)
            return self[key]
        val = fn()
        self[key] = val
        if len(self) > self.maxsize:
            self.popitem(last=False)
        return val


def last_written_datetime(path):
    """Last Datetime in an existing CSV output (reads only the end of the file)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
    if "Datetime" not in header:
        return None
    col = header.index("Datetime")
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        buf = b""
        while pos > 0 and buf.count(b"\n") < 3:
            step = min(65536, pos)
            pos -= step
            f.seek(pos)
            buf = f.read(step) + buf
    lines = [l for l in buf.decode("utf-8", "ignore").splitlines() if l.strip()]
    if len(lines) < 2 and pos == 0 and len(lines) <= 1:
        return None
    last = lines[-1].split(",")
    try:
        return parse_datetime(pd.Series([last[col]])).iloc[0]
    except IndexError:
        return None
