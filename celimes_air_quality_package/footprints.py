"""Pixel footprints: regular lat/lon grids that every product is regridded onto.

A footprint is a table with columns ``Pixel Number, Latitude, Longitude``
(pixel centres). Builders here follow the conventions in the original
notebooks:

* GOES ABI: a true regular lat/lon grid whose row and column spacing match
  the real ABI pixel spacing measured separately in the north-south and
  east-west directions near the site (ABI pixels are not square).
* TEMPO L3: the native L3 lat/lon axes, cropped to the box.
* Pixel numbering: longitude ascending in the outer loop, latitude
  descending in the inner loop (pixel 1 is the north-west corner and
  numbering runs down each column).
* Only pixels whose whole footprint lies inside the box are kept.

``FootprintGrid`` turns any footprint (any numbering order) into the 2-D
grid geometry the regridders and gap-filler need.
"""

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import bbox_from_radius, haversine_km

FOOTPRINT_COLUMNS = ["Pixel Number", "Latitude", "Longitude"]


# ------------------------------------------------------------------
# ABI fixed-grid navigation (GOES-R PUG Vol. 3, section 5.1.2.8)
# ------------------------------------------------------------------

def abi_projection(ds):
    """Projection parameters from an open ABI dataset (xarray)."""
    proj = ds["goes_imager_projection"]
    r_eq = float(proj.attrs["semi_major_axis"])
    r_pol = float(proj.attrs["semi_minor_axis"])
    h = float(proj.attrs["perspective_point_height"])
    return dict(r_eq=r_eq, r_pol=r_pol, H=h + r_eq,
                lon_origin=float(proj.attrs["longitude_of_projection_origin"]))


def abi_scan_to_latlon(x, y, params):
    """ABI scan angles (radians) to geodetic lat/lon (degrees). Off-earth -> NaN."""
    r_eq, r_pol, H = params["r_eq"], params["r_pol"], params["H"]
    lam0 = np.deg2rad(params["lon_origin"])
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    a = np.sin(x) ** 2 + np.cos(x) ** 2 * (np.cos(y) ** 2 + (r_eq ** 2 / r_pol ** 2) * np.sin(y) ** 2)
    b = -2 * H * np.cos(x) * np.cos(y)
    c = H ** 2 - r_eq ** 2
    disc = b ** 2 - 4 * a * c
    with np.errstate(invalid="ignore"):
        rs = (-b - np.sqrt(disc)) / (2 * a)
        sx = rs * np.cos(x) * np.cos(y)
        sy = -rs * np.sin(x)
        sz = rs * np.cos(x) * np.sin(y)
        lat = np.rad2deg(np.arctan((r_eq ** 2 / r_pol ** 2) * (sz / np.sqrt((H - sx) ** 2 + sy ** 2))))
        lon = np.rad2deg(lam0 - np.arctan(sy / (H - sx)))
    off = disc < 0
    return np.where(off, np.nan, lat), np.where(off, np.nan, lon)


def abi_latlon_to_scan(lat, lon, params):
    """Geodetic lat/lon (degrees) to ABI scan angles (radians). Inverse of the above."""
    r_eq, r_pol, H = params["r_eq"], params["r_pol"], params["H"]
    lam0 = np.deg2rad(params["lon_origin"])
    phi = np.deg2rad(np.asarray(lat, dtype=float))
    lam = np.deg2rad(np.asarray(lon, dtype=float))
    e = np.sqrt((r_eq ** 2 - r_pol ** 2) / r_eq ** 2)
    phi_c = np.arctan((r_pol ** 2 / r_eq ** 2) * np.tan(phi))
    rc = r_pol / np.sqrt(1 - e ** 2 * np.cos(phi_c) ** 2)
    sx = H - rc * np.cos(phi_c) * np.cos(lam - lam0)
    sy = -rc * np.cos(phi_c) * np.sin(lam - lam0)
    sz = rc * np.sin(phi_c)
    y = np.arctan(sz / sx)
    x = np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2))
    return x, y


def _resolve_box(center, half_width_km, bbox):
    """Return (lat_min, lat_max, lon_min, lon_max, ref_lat) from either input style."""
    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = bbox
        return lat_min, lat_max, lon_min, lon_max, (lat_min + lat_max) / 2
    if center is None or half_width_km is None:
        raise ValueError("Give either center=(lat, lon) with half_width_km, or bbox=(lat_min, lat_max, lon_min, lon_max)")
    lat_min, lat_max, lon_min, lon_max = bbox_from_radius(center[0], center[1], half_width_km)
    return lat_min, lat_max, lon_min, lon_max, center[0]


def _number_pixels(lat_vals, lon_vals):
    """Pixel numbering convention: lon ascending (outer), lat descending (inner)."""
    lat_vals = np.sort(np.asarray(lat_vals))[::-1]
    lon_vals = np.sort(np.asarray(lon_vals))
    lon_g, lat_g = np.meshgrid(lon_vals, lat_vals, indexing="ij")  # [col, row]
    n = lat_g.size
    return pd.DataFrame({
        "Pixel Number": np.arange(1, n + 1),
        "Latitude": lat_g.ravel(),
        "Longitude": lon_g.ravel(),
    })


def _first_nc(path):
    if os.path.isfile(path):
        return path
    files = sorted(glob.glob(os.path.join(path, "**", "*.nc"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .nc files under {path}")
    return files[0]


# ------------------------------------------------------------------
# Footprint builders
# ------------------------------------------------------------------

def build_goes_footprint(goes_nc, center=None, half_width_km=None, bbox=None, out_csv=None):
    """Build a regular lat/lon footprint matched to the local GOES ABI pixel spacing.

    Parameters
    ----------
    goes_nc : str
        Any ABI L2 .nc file (or a folder containing one), used only for its
        navigation and to measure the real pixel spacing near the site.
    center, half_width_km : (lat, lon), float
        Box centred on a site, extending half_width_km north/south/east/west.
    bbox : (lat_min, lat_max, lon_min, lon_max)
        Alternative to center/half_width_km.
    out_csv : str, optional
        Where to save the footprint.

    Returns
    -------
    DataFrame with Pixel Number, Latitude, Longitude.
    """
    import xarray as xr

    lat_min, lat_max, lon_min, lon_max, ref_lat = _resolve_box(center, half_width_km, bbox)

    nc = _first_nc(goes_nc)
    with xr.open_dataset(nc) as ds:
        params = abi_projection(ds)
        x = ds["x"].values
        y = ds["y"].values

    xx, yy = np.meshgrid(x, y)
    lat_grid, lon_grid = abi_scan_to_latlon(xx, yy, params)
    mask = (lat_grid >= lat_min) & (lat_grid <= lat_max) & (lon_grid >= lon_min) & (lon_grid <= lon_max)
    rows, cols = np.where(mask)
    if rows.size == 0:
        raise ValueError("No ABI pixels inside the box. Check the coordinates and the file's coverage.")

    nr, nc_ = lat_grid.shape
    # East-west spacing: (row, col) -> (row, col+1); north-south: (row, col) -> (row+1, col)
    ok_ew = (cols + 1 < nc_)
    ok_ew[ok_ew] &= mask[rows[ok_ew], cols[ok_ew] + 1]
    ok_ns = (rows + 1 < nr)
    ok_ns[ok_ns] &= mask[rows[ok_ns] + 1, cols[ok_ns]]
    r, c = rows[ok_ew], cols[ok_ew]
    lon_px_m = float(np.nanmean(haversine_km(lat_grid[r, c], lon_grid[r, c], lat_grid[r, c + 1], lon_grid[r, c + 1]))) * 1000
    r, c = rows[ok_ns], cols[ok_ns]
    lat_px_m = float(np.nanmean(haversine_km(lat_grid[r, c], lon_grid[r, c], lat_grid[r + 1, c], lon_grid[r + 1, c]))) * 1000
    print(f"Native ABI pixel spacing here: EW = {lon_px_m:.1f} m, NS = {lat_px_m:.1f} m")

    lat_step = lat_px_m / 111000.0
    lon_step = lon_px_m / (111000.0 * np.cos(np.radians(ref_lat)))
    lat_vals = np.arange(lat_max, lat_min - lat_step, -lat_step)
    lon_vals = np.arange(lon_min, lon_max + lon_step, lon_step)
    lat_vals = lat_vals[(lat_vals - lat_step / 2 >= lat_min) & (lat_vals + lat_step / 2 <= lat_max)]
    lon_vals = lon_vals[(lon_vals - lon_step / 2 >= lon_min) & (lon_vals + lon_step / 2 <= lon_max)]
    if len(lat_vals) == 0 or len(lon_vals) == 0:
        raise ValueError("Box is too small for the pixel spacing; no complete pixels fit.")

    out = _number_pixels(lat_vals, lon_vals)
    _save(out, out_csv)
    print(f"GOES footprint: {len(out)} pixels ({len(lon_vals)} cols x {len(lat_vals)} rows)")
    return out


def build_tempo_footprint(tempo_nc, center=None, half_width_km=None, bbox=None, out_csv=None):
    """Footprint on the native TEMPO L3 grid (1-D lat/lon axes), cropped to the box."""
    import h5py

    lat_min, lat_max, lon_min, lon_max, _ = _resolve_box(center, half_width_km, bbox)
    with h5py.File(_first_nc(tempo_nc), "r") as f:
        lon_axis = f["longitude"][:]
        lat_axis = f["latitude"][:]
    half_lon = abs(float(np.mean(np.diff(lon_axis)))) / 2
    half_lat = abs(float(np.mean(np.diff(lat_axis)))) / 2
    lon_sel = lon_axis[(lon_axis - half_lon >= lon_min) & (lon_axis + half_lon <= lon_max)]
    lat_sel = lat_axis[(lat_axis - half_lat >= lat_min) & (lat_axis + half_lat <= lat_max)]
    if len(lat_sel) == 0 or len(lon_sel) == 0:
        raise ValueError("Box is too small for the TEMPO grid spacing; no complete pixels fit.")
    out = _number_pixels(lat_sel, lon_sel)
    _save(out, out_csv)
    print(f"TEMPO footprint: {len(out)} pixels ({len(lon_sel)} cols x {len(lat_sel)} rows)")
    return out


def _save(df, out_csv):
    if out_csv:
        d = os.path.dirname(out_csv)
        if d:
            os.makedirs(d, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Saved footprint to {out_csv}")


def load_footprint(footprint):
    """Load a footprint from a CSV path or DataFrame; accepts 'Pixel_Number' too."""
    df = pd.read_csv(footprint) if isinstance(footprint, (str, os.PathLike)) else footprint.copy()
    if "Pixel_Number" in df.columns and "Pixel Number" not in df.columns:
        df = df.rename(columns={"Pixel_Number": "Pixel Number"})
    missing = set(FOOTPRINT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Footprint is missing columns {sorted(missing)}")
    df = df[FOOTPRINT_COLUMNS].drop_duplicates()
    df["Pixel Number"] = df["Pixel Number"].astype(int)
    return df.sort_values("Pixel Number").reset_index(drop=True)


# ------------------------------------------------------------------
# Grid geometry
# ------------------------------------------------------------------

@dataclass
class FootprintGrid:
    """2-D geometry of a regular footprint, independent of pixel-numbering order.

    Row 0 is the northern-most latitude, column 0 the western-most longitude.
    """
    footprint: pd.DataFrame
    lats: np.ndarray          # row centres, descending
    lons: np.ndarray          # column centres, ascending
    rows: np.ndarray          # row index of each footprint pixel (footprint order)
    cols: np.ndarray          # column index of each footprint pixel
    pixel_ids: np.ndarray     # (n_rows, n_cols) Pixel Number, -1 where no pixel
    lat_step: float
    lon_step: float

    @classmethod
    def from_footprint(cls, footprint, decimals=6):
        fp = load_footprint(footprint)
        lat_r = np.round(fp["Latitude"].to_numpy(), decimals)
        lon_r = np.round(fp["Longitude"].to_numpy(), decimals)
        lats = np.unique(lat_r)[::-1]
        lons = np.unique(lon_r)
        rows = np.searchsorted(-lats, -lat_r)
        cols = np.searchsorted(lons, lon_r)
        pid = np.full((len(lats), len(lons)), -1, dtype=int)
        pid[rows, cols] = fp["Pixel Number"].to_numpy()
        lat_step = float(np.median(np.abs(np.diff(lats)))) if len(lats) > 1 else np.nan
        lon_step = float(np.median(np.abs(np.diff(lons)))) if len(lons) > 1 else np.nan
        return cls(fp, lats, lons, rows, cols, pid, lat_step, lon_step)

    @property
    def shape(self):
        return (len(self.lats), len(self.lons))

    @property
    def lat_edges(self):
        """Row edges, descending (n_rows + 1)."""
        return np.concatenate([[self.lats[0] + self.lat_step / 2], self.lats - self.lat_step / 2])

    @property
    def lon_edges(self):
        """Column edges, ascending (n_cols + 1)."""
        return np.concatenate([self.lons - self.lon_step / 2, [self.lons[-1] + self.lon_step / 2]])

    @property
    def lat_grid(self):
        return np.repeat(self.lats[:, None], len(self.lons), axis=1)

    @property
    def lon_grid(self):
        return np.repeat(self.lons[None, :], len(self.lats), axis=0)

    def index_of(self, pixel_numbers):
        """(rows, cols) for an array of Pixel Numbers; -1 where not in the footprint."""
        lut = pd.Series(np.arange(len(self.footprint)), index=self.footprint["Pixel Number"].to_numpy())
        pos = lut.reindex(np.asarray(pixel_numbers).astype(int)).to_numpy()
        ok = ~np.isnan(pos)
        r = np.full(len(pos), -1)
        c = np.full(len(pos), -1)
        r[ok] = self.rows[pos[ok].astype(int)]
        c[ok] = self.cols[pos[ok].astype(int)]
        return r, c

    def to_frame(self, grids):
        """Flatten {name: (n_rows, n_cols) array} to a DataFrame in footprint order."""
        out = self.footprint.copy()
        for name, g in grids.items():
            out[name] = np.asarray(g)[self.rows, self.cols]
        return out


def pixels_within_radius(footprint, lat, lon, radius_km):
    """Footprint rows whose centre is within radius_km of (lat, lon)."""
    fp = load_footprint(footprint)
    d = haversine_km(fp["Latitude"].to_numpy(), fp["Longitude"].to_numpy(), lat, lon)
    return fp[d <= radius_km].assign(distance_km=d[d <= radius_km])
