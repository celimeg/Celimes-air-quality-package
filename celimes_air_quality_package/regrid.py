"""Regrid satellite and reanalysis products onto a footprint.

* regrid_goes_aod   GOES ABI AOD, area-overlap and DQF weighted, hourly
* regrid_tempo      TEMPO L3, either clipped to a native-TEMPO footprint or
                    area-weighted onto any other footprint (e.g. a GOES grid)
* regrid_modis_aod  MODIS AOD granules, area-weighted binning
* regrid_era5       ERA5, sub-pixel averaged onto a footprint

Outputs are appended hour by hour to a CSV or HDF5 table, and every
function resumes from where an earlier run stopped.
"""

import glob
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .footprints import FootprintGrid, abi_projection, abi_scan_to_latlon, load_footprint
from .utils import append_table, existing_datetimes

logger = logging.getLogger(__name__)


# ==================================================================
# Shared: fractional overlap of rectangular source pixels with the grid
# ==================================================================

def _overlaps(lat_edges, lon_edges, lat_lo, lat_hi, lon_lo, lon_hi):
    """All (source pixel, grid cell) overlaps.

    lat_edges must be descending, lon_edges ascending. Returns arrays
    (src, gi, gj, frac) where frac = overlap area / source pixel area
    (in degree^2), only for positive overlaps.
    """
    n_rows, n_cols = len(lat_edges) - 1, len(lon_edges) - 1
    area = (lat_hi - lat_lo) * (lon_hi - lon_lo)
    lat_lo_idx = np.clip(np.searchsorted(-lat_edges, -lat_hi, side="left"), 0, n_rows)
    lat_hi_idx = np.clip(np.searchsorted(-lat_edges, -lat_lo, side="right"), 0, n_rows)
    lon_lo_idx = np.clip(np.searchsorted(lon_edges, lon_lo, side="right") - 1, 0, n_cols)
    lon_hi_idx = np.clip(np.searchsorted(lon_edges, lon_hi, side="left"), 0, n_cols)

    src_all, gi_all, gj_all, frac_all = [], [], [], []
    if len(area) == 0:
        return (np.array([], int),) * 3 + (np.array([]),)
    max_di = int((lat_hi_idx - lat_lo_idx).max())
    max_dj = int((lon_hi_idx - lon_lo_idx).max())
    src_idx = np.arange(len(area))
    for di in range(max_di + 1):
        for dj in range(max_dj + 1):
            gi = lat_lo_idx + di
            gj = lon_lo_idx + dj
            ok = (gi < lat_hi_idx) & (gi < n_rows) & (gj < lon_hi_idx) & (gj < n_cols) & (gi >= 0) & (gj >= 0)
            if not ok.any():
                continue
            gi_v, gj_v, s = gi[ok], gj[ok], src_idx[ok]
            ov_lat = np.minimum(lat_hi[ok], lat_edges[gi_v]) - np.maximum(lat_lo[ok], lat_edges[gi_v + 1])
            ov_lon = np.minimum(lon_hi[ok], lon_edges[gj_v + 1]) - np.maximum(lon_lo[ok], lon_edges[gj_v])
            pos = (ov_lat > 0) & (ov_lon > 0) & (area[ok] > 0)
            if not pos.any():
                continue
            src_all.append(s[pos])
            gi_all.append(gi_v[pos])
            gj_all.append(gj_v[pos])
            frac_all.append(ov_lat[pos] * ov_lon[pos] / area[ok][pos])
    if not src_all:
        return (np.array([], int),) * 3 + (np.array([]),)
    return (np.concatenate(src_all), np.concatenate(gi_all),
            np.concatenate(gj_all), np.concatenate(frac_all))


def _hours(start, end):
    """Hourly UTC timestamps from start date 00z to end date 23z (inclusive)."""
    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize() + pd.Timedelta(hours=23)
    return pd.date_range(s, e, freq="h")


# ==================================================================
# GOES ABI AOD
# ==================================================================

class _AbiMapping:
    """Precomputed ABI-pixel to footprint-cell overlap weights for one ABI grid."""

    def __init__(self, nc_path, grid, domain_pad=0.1):
        import xarray as xr

        with xr.open_dataset(nc_path, decode_times=False) as ds:
            params = abi_projection(ds)
            x = ds["x"].values
            y = ds["y"].values
        self.shape = (len(y), len(x))
        self.key = (round(params["lon_origin"], 3), self.shape)

        abi_lat, abi_lon = abi_scan_to_latlon(*np.meshgrid(x, y), params)
        dx, dy = np.diff(x).mean(), np.diff(y).mean()
        xe = np.concatenate([[x[0] - dx / 2], x + dx / 2])
        ye = np.concatenate([[y[0] - dy / 2], y + dy / 2])
        lat_e, lon_e = abi_scan_to_latlon(*np.meshgrid(xe, ye), params)

        le, oe = grid.lat_edges, grid.lon_edges
        with np.errstate(invalid="ignore"):
            dom = ((abi_lat >= le[-1] - domain_pad) & (abi_lat <= le[0] + domain_pad) &
                   (abi_lon >= oe[0] - domain_pad) & (abi_lon <= oe[-1] + domain_pad))
        self.rows, self.cols = np.where(dom)
        r, c = self.rows, self.cols
        lat_lo, lat_hi = lat_e[r + 1, c], lat_e[r, c]
        lon_lo, lon_hi = lon_e[r, c], lon_e[r, c + 1]
        edge_ok = ~(np.isnan(lat_lo) | np.isnan(lat_hi) | np.isnan(lon_lo) | np.isnan(lon_hi))
        keep = np.where(edge_ok)[0]
        src, gi, gj, frac = _overlaps(le, oe, lat_lo[keep], lat_hi[keep], lon_lo[keep], lon_hi[keep])
        self.src = keep[src]          # index into self.rows/self.cols
        self.gi, self.gj, self.frac = gi, gj, frac
        print(f"  ABI pixels near footprint: {len(self.rows)}  (overlaps: {len(self.src)})")


def regrid_goes_aod(goes_dir, footprint, out_path, start, end,
                    dqf_weights=(1.0, 0.5, 0.25, 0.0), overwrite=False):
    """Regrid GOES ABI AOD onto a footprint, one row per pixel per hour.

    Each ABI pixel contributes to every footprint cell it overlaps, weighted
    by overlap fraction x DQF weight. All scans within the hour are pooled.

    Parameters
    ----------
    goes_dir : str
        Root of downloaded files laid out as goes_dir/YYYY/DOY/HH/*.nc.
    footprint : str or DataFrame
        Target footprint (Pixel Number, Latitude, Longitude).
    out_path : str
        Output .csv or .h5 (key "aod"). Existing hours are skipped.
    start, end : str
        Inclusive dates, "YYYY-MM-DD".
    dqf_weights : 4-tuple
        Weights for DQF 0, 1, 2, 3.

    Output columns
    --------------
    Pixel Number, Latitude, Longitude, Datetime (UTC),
    Weighted_AOD   overlap x DQF weighted mean AOD
    AOD_DQF0       overlap-weighted mean of DQF=0 retrievals only
    Weighted_DQF   mean combined weight per contributing retrieval
    avg_DQF        mean raw DQF value
    pct_DQF0       percent of contributing retrievals with DQF=0
    """
    dqf_w = np.asarray(dqf_weights, dtype=float)
    grid = FootprintGrid.from_footprint(footprint)
    nr, nc = grid.shape

    if overwrite and os.path.exists(out_path):
        os.remove(out_path)
    done = existing_datetimes(out_path, key="aod")
    mappings = {}

    def mapping_for(nc_file):
        import xarray as xr
        with xr.open_dataset(nc_file, decode_times=False) as ds:
            key = (round(float(ds["goes_imager_projection"].attrs["longitude_of_projection_origin"]), 3),
                   (ds.sizes["y"], ds.sizes["x"]))
        if key not in mappings:
            print(f"Building ABI overlap map for {key} ...")
            mappings[key] = _AbiMapping(nc_file, grid)
        return mappings[key]

    import xarray as xr
    n_written = 0
    for ts in _hours(start, end):
        if ts in done:
            continue
        hour_dir = os.path.join(goes_dir, f"{ts:%Y}", f"{ts.dayofyear:03d}", f"{ts:%H}")
        files = sorted(glob.glob(os.path.join(hour_dir, "*.nc")))
        if not files:
            continue

        acc = {k: np.zeros(nr * nc) for k in ("aod_w", "w", "dqf_raw", "cnt", "aod0", "w0", "cnt0")}
        for f in files:
            try:
                m = mapping_for(f)
                with xr.open_dataset(f, decode_times=False) as ds:
                    aod = ds["AOD"].values[m.rows, m.cols]
                    dqf = ds["DQF"].values[m.rows, m.cols]
            except (OSError, KeyError, ValueError) as exc:
                print(f"  Skipping unreadable file {os.path.basename(f)}: {exc}")
                continue
            valid = ~np.isnan(aod) & (aod >= 0) & ~np.isnan(dqf) & (dqf >= 0) & (dqf <= 3)
            sel = valid[m.src]
            if not sel.any():
                continue
            s = m.src[sel]
            cell = m.gi[sel] * nc + m.gj[sel]
            frac = m.frac[sel]
            a = aod[s]
            d = dqf[s].astype(int)
            w = frac * dqf_w[d]
            np.add.at(acc["aod_w"], cell, a * w)
            np.add.at(acc["w"], cell, w)
            np.add.at(acc["dqf_raw"], cell, d)
            np.add.at(acc["cnt"], cell, 1)
            z = d == 0
            np.add.at(acc["aod0"], cell[z], a[z] * frac[z])
            np.add.at(acc["w0"], cell[z], frac[z])
            np.add.at(acc["cnt0"], cell[z], 1)

        def ratio(n, dnm):
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(dnm == 0, np.nan, n / np.where(dnm == 0, 1, dnm)).reshape(nr, nc)

        out = grid.to_frame({
            "Weighted_AOD": ratio(acc["aod_w"], acc["w"]),
            "AOD_DQF0": ratio(acc["aod0"], acc["w0"]),
            "Weighted_DQF": ratio(acc["w"], acc["cnt"]),
            "avg_DQF": ratio(acc["dqf_raw"], acc["cnt"]),
            "pct_DQF0": ratio(acc["cnt0"], acc["cnt"]) * 100,
        })
        out.insert(3, "Datetime", ts)
        append_table(out, out_path, key="aod")
        n_written += 1
        print(f"  {ts:%Y-%m-%d %H}:00  {len(files)} scans")
    print(f"Done: {n_written} new hours written to {out_path}")


# ==================================================================
# TEMPO L3
# ==================================================================

TEMPO_FIELDS = {
    "geolocation/relative_azimuth_angle": "relative_azimuth_angle",
    "geolocation/solar_zenith_angle": "solar_zenith_angle",
    "geolocation/viewing_zenith_angle": "viewing_zenith_angle",
    "product/vertical_column_stratosphere": "vertical_column_stratosphere",
    "product/vertical_column_troposphere": "vertical_column_troposphere",
    "product/vertical_column_troposphere_uncertainty": "vertical_column_troposphere_uncertainty",
    "qa_statistics/max_vertical_column_stratosphere_sample": "max_vertical_column_stratosphere_sample",
    "qa_statistics/max_vertical_column_total_sample": "max_vertical_column_total_sample",
    "qa_statistics/max_vertical_column_troposphere_sample": "max_vertical_column_troposphere_sample",
    "qa_statistics/max_vertical_column_troposphere_uncertainty_sample": "max_vertical_column_troposphere_uncertainty_sample",
    "qa_statistics/min_vertical_column_stratosphere_sample": "min_vertical_column_stratosphere_sample",
    "qa_statistics/min_vertical_column_total_sample": "min_vertical_column_total_sample",
    "qa_statistics/min_vertical_column_troposphere_sample": "min_vertical_column_troposphere_sample",
    "qa_statistics/min_vertical_column_troposphere_uncertainty_sample": "min_vertical_column_troposphere_uncertainty_sample",
    "qa_statistics/num_vertical_column_stratosphere_samples": "num_vertical_column_stratosphere_samples",
    "qa_statistics/num_vertical_column_total_samples": "num_vertical_column_total_samples",
    "qa_statistics/num_vertical_column_troposphere_samples": "num_vertical_column_troposphere_samples",
    "qa_statistics/num_vertical_column_troposphere_uncertainty_samples": "num_vertical_column_troposphere_uncertainty_samples",
    "support_data/albedo": "albedo",
    "support_data/amf_cloud_fraction": "amf_cloud_fraction",
    "support_data/amf_cloud_pressure": "amf_cloud_pressure",
    "support_data/amf_stratosphere": "amf_stratosphere",
    "support_data/amf_total": "amf_total",
    "support_data/amf_troposphere": "amf_troposphere",
    "support_data/eff_cloud_fraction": "eff_cloud_fraction",
    "support_data/fitted_slant_column": "fitted_slant_column",
    "support_data/fitted_slant_column_uncertainty": "fitted_slant_column_uncertainty",
    "support_data/snow_ice_fraction": "snow_ice_fraction",
    "support_data/surface_pressure": "surface_pressure",
    "support_data/terrain_height": "terrain_height",
    "support_data/tropopause_pressure": "tropopause_pressure",
    "support_data/vertical_column_total": "vertical_column_total",
    "support_data/vertical_column_total_uncertainty": "vertical_column_total_uncertainty",
}
TEMPO_QF_KEY = "product/main_data_quality_flag"
TEMPO_QF_WEIGHTS = {0: 1.0, 1: 0.5, 2: 0.0}
TEMPO_QF_WEIGHTED_FIELDS = ["vertical_column_troposphere", "vertical_column_stratosphere",
                            "vertical_column_troposphere_uncertainty", "vertical_column_total",
                            "vertical_column_total_uncertainty"]


def _tempo_hour_groups(tempo_dir):
    """{hour Timestamp: [files]} from a tempo_dir/YYYY/MM/DD/HH/*.nc layout."""
    groups = {}
    for fp in sorted(Path(tempo_dir).rglob("*.nc")):
        try:
            y, m, d, h = fp.parent.parts[-4:]
            ts = pd.Timestamp(datetime.strptime(f"{y}-{m}-{d} {h}", "%Y-%m-%d %H"))
        except ValueError:
            logger.warning("Can't read the hour from the folder of %s; skipping", fp)
            continue
        groups.setdefault(ts, []).append(fp)
    return groups


def _read_window(f, key, i0, i1, j0, j1):
    ds = f[key]
    arr = ds[0, i0:i1, j0:j1] if ds.ndim == 3 else ds[i0:i1, j0:j1]
    arr = arr.astype(float)
    fill = ds.attrs.get("_FillValue")
    if fill is not None:
        arr[arr == np.asarray(fill).ravel()[0]] = np.nan
    return arr


def regrid_tempo(tempo_dir, footprint, out_path, start=None, end=None, method="auto",
                 fields=None, key="no2", tolerance_deg=1e-4):
    """Regrid TEMPO L3 granules onto a footprint, one row per pixel per hour.

    Parameters
    ----------
    tempo_dir : str
        Root laid out as tempo_dir/YYYY/MM/DD/HH/*.nc (download_tempo's layout).
    footprint : str or DataFrame
    out_path : str
        Output .h5 (table key ``key``) or .csv. Existing hours are skipped.
    start, end : str, optional
        Inclusive "YYYY-MM-DD" limits.
    method : "auto", "native" or "area"
        "native": each footprint pixel takes the TEMPO L3 cell at the same
        centre (footprint built with build_tempo_footprint). Multiple granules
        in one hour are averaged with nanmean. Output columns match the
        original per-site TEMPO HDF5 files.
        "area": area-weighted onto any footprint (e.g. a GOES grid). Adds
        QF-weighted columns (``<field>_weighted`` with QF weights 1/0.5/0),
        ``main_data_quality_flag_weighted`` and pixel counts per QF.
        "auto" picks native if every footprint centre is within tolerance_deg
        of a TEMPO cell centre.
    """
    import h5py

    fields = fields or TEMPO_FIELDS
    grid = FootprintGrid.from_footprint(footprint)
    fp = grid.footprint
    groups = _tempo_hour_groups(tempo_dir)
    if start:
        groups = {t: v for t, v in groups.items() if t >= pd.Timestamp(start)}
    if end:
        groups = {t: v for t, v in groups.items() if t <= pd.Timestamp(end) + pd.Timedelta(hours=23)}
    if not groups:
        print("No TEMPO files found for that range.")
        return
    done = existing_datetimes(out_path, key=key)

    with h5py.File(groups[min(groups)][0], "r") as f:
        lon_axis = f["longitude"][:]
        lat_axis = f["latitude"][:]

    lon_i = np.abs(lon_axis[None, :] - fp["Longitude"].to_numpy()[:, None]).argmin(axis=1)
    lat_i = np.abs(lat_axis[None, :] - fp["Latitude"].to_numpy()[:, None]).argmin(axis=1)
    aligned = (np.abs(lon_axis[lon_i] - fp["Longitude"].to_numpy()).max() <= tolerance_deg and
               np.abs(lat_axis[lat_i] - fp["Latitude"].to_numpy()).max() <= tolerance_deg)
    if method == "auto":
        method = "native" if aligned else "area"
    elif method == "native" and not aligned:
        print("WARNING: footprint is not aligned with the TEMPO grid; nearest cells will be used.")
    print(f"TEMPO regrid method: {method}")

    if method == "native":
        i0, i1 = lat_i.min(), lat_i.max() + 1
        j0, j1 = lon_i.min(), lon_i.max() + 1
        li, lj = lat_i - i0, lon_i - j0
    else:
        dlat = abs(float(np.median(np.diff(lat_axis))))
        dlon = abs(float(np.median(np.diff(lon_axis))))
        le, oe = grid.lat_edges, grid.lon_edges
        ii = np.where((lat_axis + dlat / 2 > le[-1]) & (lat_axis - dlat / 2 < le[0]))[0]
        jj = np.where((lon_axis + dlon / 2 > oe[0]) & (lon_axis - dlon / 2 < oe[-1]))[0]
        i0, i1, j0, j1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1
        lat2d, lon2d = np.meshgrid(lat_axis[i0:i1], lon_axis[j0:j1], indexing="ij")
        src, gi, gj, frac = _overlaps(le, oe, (lat2d - dlat / 2).ravel(), (lat2d + dlat / 2).ravel(),
                                      (lon2d - dlon / 2).ravel(), (lon2d + dlon / 2).ravel())
        # Weight by overlap area (frac x source area); the source area is constant
        cell = gi * grid.shape[1] + gj
        n_cells = grid.shape[0] * grid.shape[1]

    n_written = 0
    for ts in sorted(groups):
        if ts in done:
            continue
        per_file = []
        for path in groups[ts]:
            try:
                with h5py.File(path, "r") as f:
                    qf = _read_window(f, TEMPO_QF_KEY, i0, i1, j0, j1)
                    qf[qf == -9999] = np.nan
                    data = {}
                    for k, name in fields.items():
                        try:
                            data[name] = _read_window(f, k, i0, i1, j0, j1)
                        except KeyError:
                            data[name] = np.full(qf.shape, np.nan)
                per_file.append((qf, data))
            except OSError as exc:
                print(f"  Skipping unreadable {path.name}: {exc}")
        if not per_file:
            continue

        if method == "native":
            out = fp.copy()
            out["Datetime"] = ts
            out["main_data_quality_flag"] = np.nanmean(np.vstack([q[li, lj] for q, _ in per_file]), axis=0) \
                if len(per_file) > 1 else per_file[0][0][li, lj]
            for name in fields.values():
                stack = np.vstack([d[name][li, lj] for _, d in per_file])
                with np.errstate(invalid="ignore"), _quiet_nanmean():
                    out[name] = np.nanmean(stack, axis=0)
            if "vertical_column_troposphere" in out:
                out.loc[out["vertical_column_troposphere"].isna(), "main_data_quality_flag"] = np.nan
        else:
            out = _tempo_area_hour(per_file, fields, src, cell, frac, n_cells, grid)
            out.insert(3, "Datetime", ts)

        append_table(out, out_path, key=key)
        n_written += 1
        print(f"  {ts:%Y-%m-%d %H}:00  {len(per_file)} granule(s)")
    print(f"Done: {n_written} new hours written to {out_path}")


class _quiet_nanmean:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc):
        self._w.__exit__(*exc)


def _tempo_area_hour(per_file, fields, src, cell, frac, n_cells, grid):
    """Area-weighted (and QF-weighted) TEMPO values for one hour.

    Each field's mean uses only the overlap weight of cells where that
    field is not NaN, so missing retrievals don't pull the mean toward zero.
    """
    sums = {n: np.zeros(n_cells) for n in fields.values()}
    wts = {n: np.zeros(n_cells) for n in fields.values()}
    qsum = {n: np.zeros(n_cells) for n in TEMPO_QF_WEIGHTED_FIELDS}
    qwt = {n: np.zeros(n_cells) for n in TEMPO_QF_WEIGHTED_FIELDS}
    qf_sum, qf_wsum, qf_plain, qf_plain_w = (np.zeros(n_cells) for _ in range(4))
    counts = {k: np.zeros(n_cells) for k in ("all", 0, 1, 2)}

    for qf, data in per_file:
        q = qf.ravel()[src]
        qw = np.zeros_like(q)
        for code, wgt in TEMPO_QF_WEIGHTS.items():
            qw[q == code] = wgt
        np.add.at(counts["all"], cell, 1)
        for k in (0, 1, 2):
            np.add.at(counts[k], cell[q == k], 1)
        okq = np.isfinite(q)
        np.add.at(qf_plain, cell[okq], q[okq] * frac[okq])
        np.add.at(qf_plain_w, cell[okq], frac[okq])
        cw = frac * qw
        np.add.at(qf_sum, cell[okq], q[okq] * cw[okq])
        np.add.at(qf_wsum, cell[okq], cw[okq])
        for name in fields.values():
            v = data[name].ravel()[src]
            ok = np.isfinite(v)
            np.add.at(sums[name], cell[ok], v[ok] * frac[ok])
            np.add.at(wts[name], cell[ok], frac[ok])
            if name in qsum:
                np.add.at(qsum[name], cell[ok], v[ok] * cw[ok])
                np.add.at(qwt[name], cell[ok], cw[ok])

    def ratio(n, d):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(d > 0, n / np.where(d > 0, d, 1), np.nan).reshape(grid.shape)

    grids = {"pixel_count": counts["all"].reshape(grid.shape),
             "pixel_count_qf0": counts[0].reshape(grid.shape),
             "pixel_count_qf1": counts[1].reshape(grid.shape),
             "pixel_count_qf2": counts[2].reshape(grid.shape),
             "main_data_quality_flag": ratio(qf_plain, qf_plain_w)}
    for name in fields.values():
        grids[name] = ratio(sums[name], wts[name])
    for name in TEMPO_QF_WEIGHTED_FIELDS:
        if name in qsum:
            grids[f"{name}_weighted"] = ratio(qsum[name], qwt[name])
    grids["main_data_quality_flag_weighted"] = ratio(qf_sum, qf_wsum)
    return grid.to_frame(grids)


# ==================================================================
# MODIS AOD (unchanged method from the original package)
# ==================================================================

_MODIS_RES_KM = {"MOD04_L2": 10.0, "MYD04_L2": 10.0, "MOD04_3K": 3.0, "MYD04_3K": 3.0}
_MODIS_FIELDS = ["Scan_Start_Time", "Solar_Zenith", "Solar_Azimuth", "Sensor_Zenith", "Sensor_Azimuth",
                 "Optical_Depth_Land_And_Ocean", "Corrected_Optical_Depth_Land", "Mass_Concentration_Land",
                 "Mean_Reflectance_Land", "Aerosol_Cloud_Fraction_Land", "Land_Ocean_Quality_Flag",
                 "Scattering_Angle", "Glint_Angle"]
_MODIS_MULTILAYER = {"Corrected_Optical_Depth_Land", "Mean_Reflectance_Land"}


def build_modis_footprint(lat_min, lat_max, lon_min, lon_max, resolution_km, out_csv=None):
    """Regular footprint at a fixed resolution (3 km for *_3K, 10 km for *_L2)."""
    from .footprints import _number_pixels, _save
    mean_lat = (lat_min + lat_max) / 2
    lat_step = resolution_km / 111.0
    lon_step = resolution_km / (111.0 * np.cos(np.deg2rad(mean_lat)))
    lat_c = np.arange(lat_max - lat_step / 2, lat_min + lat_step / 2 - 1e-9, -lat_step)
    lon_c = np.arange(lon_min + lon_step / 2, lon_max - lon_step / 2 + 1e-9, lon_step)
    lat_c = lat_c[(lat_c - lat_step / 2 >= lat_min - 1e-9) & (lat_c + lat_step / 2 <= lat_max + 1e-9)]
    lon_c = lon_c[(lon_c - lon_step / 2 >= lon_min - 1e-9) & (lon_c + lon_step / 2 <= lon_max + 1e-9)]
    if len(lat_c) == 0 or len(lon_c) == 0:
        raise ValueError(f"Box too small for a {resolution_km} km grid")
    out = _number_pixels(lat_c, lon_c)
    _save(out, out_csv)
    return out


def _hdf_field(hdf, field):
    sds = hdf.select(field)
    data = sds[:].astype(np.float64)
    attrs = sds.attributes(full=1)
    fill = attrs.get("_FillValue", [np.nan])[0]
    scale = attrs.get("scale_factor", [1.0])[0]
    offset = attrs.get("add_offset", [0.0])[0]
    data[data == fill] = np.nan
    return (data - offset) * scale


def regrid_modis_aod(data_dir, product, start, end, footprint, out_csv=None):
    """Area-weighted daily MODIS AOD on a footprint (requires pyhdf).

    Granules are read from data_dir/modis/<product>/*.hdf (download_modis_aod's
    layout). Use build_modis_footprint to make a footprint at the product's
    native resolution.
    """
    from pyhdf.SD import SD, SDC

    grid = FootprintGrid.from_footprint(footprint)
    lat_grid, lon_grid = grid.lat_grid, grid.lon_grid
    le, oe = grid.lat_edges, grid.lon_edges
    lat_min, lat_max, lon_min, lon_max = le[-1], le[0], oe[0], oe[-1]
    hdf_dir = os.path.join(data_dir, "modis", product)
    out_csv = out_csv or os.path.join(hdf_dir, "regridded", f"{start}_to_{end}_regridded.csv")
    if os.path.exists(out_csv):
        os.remove(out_csv)

    day = datetime.strptime(start, "%Y-%m-%d")
    last = datetime.strptime(end, "%Y-%m-%d")
    while day <= last:
        doy = day.timetuple().tm_yday
        files = sorted(glob.glob(os.path.join(hdf_dir, f"*.A{day.year}{doy:03d}.*.hdf")))
        cols = {}
        lats, lons, areas = [], [], []
        for path in files:
            try:
                hdf = SD(path, SDC.READ)
                lat2d = hdf.select("Latitude")[:, :]
                lon2d = hdf.select("Longitude")[:, :]
            except Exception as exc:  # pyhdf raises HDF4Error
                logger.warning("Could not read %s: %s", path, exc)
                continue
            try:
                area = 9.0 / np.cos(np.deg2rad(_hdf_field(hdf, "Sensor_Zenith")))
            except Exception:
                area = np.ones_like(lat2d, dtype=float)
            mask = (lat2d >= lat_min) & (lat2d <= lat_max) & (lon2d >= lon_min) & (lon2d <= lon_max)
            if not mask.any():
                hdf.end()
                continue
            lats.append(lat2d[mask]); lons.append(lon2d[mask]); areas.append(area[mask])
            for field in _MODIS_FIELDS:
                names = [f"{field}_{i}" for i in range(3)] if field in _MODIS_MULTILAYER else [field]
                try:
                    data = _hdf_field(hdf, field)
                    layers = [data[i][mask] for i in range(3)] if field in _MODIS_MULTILAYER else [data[mask]]
                except Exception:
                    layers = [np.full(int(mask.sum()), np.nan)] * len(names)
                for n, l in zip(names, layers):
                    cols.setdefault(n, []).append(l)
            hdf.end()
        if lats:
            lat_a, lon_a, area_a = map(np.concatenate, (lats, lons, areas))
            grids = {}
            for name, parts in cols.items():
                vals = np.concatenate(parts)
                ok = np.isfinite(vals) & np.isfinite(area_a)
                fs, fa = np.zeros(grid.shape), np.zeros(grid.shape)
                for la, lo, v, ar in zip(lat_a[ok], lon_a[ok], vals[ok], area_a[ok]):
                    r = np.sqrt(ar / np.pi)
                    inside = (np.abs(lat_grid - la) <= r / 111.0) & \
                             (np.abs(lon_grid - lo) <= r / (111.0 * np.cos(np.deg2rad(la))))
                    fs[inside] += v * ar
                    fa[inside] += ar
                with np.errstate(invalid="ignore", divide="ignore"):
                    g = fs / np.where(fa == 0, np.nan, fa)
                grids[name] = np.round(g, 3) if "Optical_Depth" in name else g
            out = grid.to_frame(grids)
            out.insert(3, "Date", day.strftime("%Y-%m-%d"))
            append_table(out, out_csv)
        day += timedelta(days=1)
    print(f"Saved {out_csv}")


# ==================================================================
# ERA5 (unchanged method from the original package)
# ==================================================================

ERA5_VARS = ["blh", "cape", "alnip", "aluvp", "alnid", "aluvd", "sp", "u10", "v10", "t2m", "d2m", "u100", "v100"]


def regrid_era5(era5_nc, footprint, out_path, start, end, sub_n=4):
    """Average ERA5 onto a footprint using sub_n x sub_n sub-points per pixel."""
    import xarray as xr

    fp = load_footprint(footprint)
    grid = FootprintGrid.from_footprint(fp)
    lat_step = grid.lat_step
    n_pix, n_sub = len(fp), sub_n * sub_n
    offs = np.linspace(-lat_step / 2, lat_step / 2, sub_n, endpoint=False) + lat_step / (2 * sub_n)
    all_lat = np.empty(n_pix * n_sub)
    all_lon = np.empty(n_pix * n_sub)
    for i, (la, lo) in enumerate(zip(fp["Latitude"], fp["Longitude"])):
        lon_step_local = lat_step * np.cos(np.deg2rad(la))
        lo_offs = np.linspace(-lon_step_local / 2, lon_step_local / 2, sub_n, endpoint=False) + lon_step_local / (2 * sub_n)
        slat, slon = np.meshgrid(la + offs, lo + lo_offs, indexing="ij")
        all_lat[i * n_sub:(i + 1) * n_sub] = slat.ravel()
        all_lon[i * n_sub:(i + 1) * n_sub] = slon.ravel()

    with xr.open_dataset(era5_nc) as ds:
        tname = "valid_time" if "valid_time" in ds.dims else "time"
        avail = [v for v in ERA5_VARS if v in ds.data_vars]
        times = pd.to_datetime(ds[tname].values)
        s, e = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23)
        for t_idx, t0 in enumerate(times):
            if not (s <= t0 <= e):
                continue
            out = fp.copy()
            out["Datetime"] = t0
            for v in avail:
                vals = ds[v].isel({tname: t_idx}).sel(
                    latitude=xr.DataArray(all_lat, dims="p"),
                    longitude=xr.DataArray(all_lon, dims="p"), method="nearest").values
                out[v] = vals.reshape(n_pix, n_sub).mean(axis=1)
            append_table(out, out_path)
    print(f"Saved {out_path}")
