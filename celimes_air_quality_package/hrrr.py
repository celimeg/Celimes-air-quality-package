"""Sample HRRR analyses at footprint pixel centres from the hrrrzarr store.

Reads s3://hrrrzarr anonymously (no download/decode of GRIB2), pulls only
the zarr chunks containing the pixels, and appends one day at a time to a
single resumable NetCDF with dims (time, pixel).

Typical use::

    from celimes_air_quality_package import hrrr
    hrrr.run_hrrr_supervised(footprint=r"...\\Reno_footprint.csv",
                             out_nc=r"Z:\\Nasa_Project\\HRRR\\Reno\\Reno_goes_hrrr_2020.nc",
                             start="2020-01-01", end="2020-12-31")
    df = hrrr.load_hrrr(r"Z:\\Nasa_Project\\HRRR\\Reno\\Reno_goes_hrrr_2020.nc")

run_hrrr_supervised runs the sampler in fresh subprocesses of chunk_days
days each, so memory never accumulates across a multi-year run. Ctrl+C is
safe at any point; rerunning resumes from the last complete day.

Command line (what the supervisor calls)::

    python -m celimes_air_quality_package.hrrr --footprint F.csv --out O.nc \\
        --start 2020-01-01 --end 2020-12-31 [--variables default|cloud] [--max-days 20]
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

import numpy as np
import pandas as pd

from .footprints import load_footprint

# (output name, hrrrzarr level, hrrrzarr variable), checked against
# https://mesowest.utah.edu/html/hrrr/zarr_documentation/output/zarr_variables.csv
HRRR_VARIABLES = [
    ("cape_surface_Jkg", "surface", "CAPE"),
    ("shear_u_0_1km_per_s", "0_1000m_above_ground", "VUCSH"),
    ("shear_v_0_1km_per_s", "0_1000m_above_ground", "VVCSH"),
    ("shear_u_0_6km_per_s", "0_6000m_above_ground", "VUCSH"),
    ("shear_v_0_6km_per_s", "0_6000m_above_ground", "VVCSH"),
    ("u_wind_10m_ms", "10m_above_ground", "UGRD"),
    ("v_wind_10m_ms", "10m_above_ground", "VGRD"),
    ("u_wind_80m_ms", "80m_above_ground", "UGRD"),
    ("v_wind_80m_ms", "80m_above_ground", "VGRD"),
    ("temp_2m_K", "2m_above_ground", "TMP"),
    ("skin_temp_K", "surface", "TMP"),
    ("dewpoint_2m_K", "2m_above_ground", "DPT"),
    ("rh_2m_pct", "2m_above_ground", "RH"),
    ("surface_pressure_Pa", "surface", "PRES"),
    ("u_wind_850mb_ms", "850mb", "UGRD"),
    ("v_wind_850mb_ms", "850mb", "VGRD"),
    ("u_wind_250mb_ms", "250mb", "UGRD"),
    ("v_wind_250mb_ms", "250mb", "VGRD"),
    ("latent_heat_flux_Wm2", "surface", "LHTFL"),
    ("sensible_heat_flux_Wm2", "surface", "SHTFL"),
    ("pbl_height_m", "surface", "HPBL"),
    ("smoke_near_surface_ugm3", "8m_above_ground", "MASSDEN"),
    ("smoke_column_kgm2", "entire_atmosphere_single_layer", "COLMD"),
]

# Total cloud cover. Level is "entire_atmosphere" (not "..._single_layer").
HRRR_CLOUD_COVER = [("total_cloud_cover_pct", "entire_atmosphere", "TCDC")]

VARIABLE_PRESETS = {"default": HRRR_VARIABLES, "cloud": HRRR_CLOUD_COVER}

SMOKE_FIELDS = ("smoke_near_surface_ugm3", "smoke_column_kgm2")
HRRRV4_DATE = pd.Timestamp("2020-12-02")          # smoke fields start here
MASSDEN_UNIT_FIX_DATE = pd.Timestamp("2021-12-21")  # MASSDEN unit label change

TIME_UNITS = "hours since 1970-01-01 00:00:00"
TIME_CALENDAR = "standard"


def _fix_smoke_units(values, timestamp):
    # Same rule as the original sampler.
    return values if timestamp < MASSDEN_UNIT_FIX_DATE else values * 1e9


def _resolve_variables(variables):
    if variables is None:
        return HRRR_VARIABLES
    if isinstance(variables, str):
        return VARIABLE_PRESETS[variables]
    return [tuple(v) for v in variables]


# ------------------------------------------------------------------
# Sampler
# ------------------------------------------------------------------

class _Sampler:
    def __init__(self, footprint, max_workers=16):
        import s3fs

        self.pixels = load_footprint(footprint)
        self.n_pix = len(self.pixels)
        # One shared S3FileSystem per process: creating one per thread leaks
        # an event-loop thread each time and slows multi-year runs to a crawl.
        self.fs = s3fs.S3FileSystem(anon=True)
        self.max_workers = max_workers
        self.iy = self.ix = None

    def build_index(self):
        import s3fs
        import xarray as xr
        from scipy.spatial import cKDTree

        print("Building nearest-neighbour index from hrrrzarr grid/HRRR_chunk_index.zarr ...")
        idx = xr.open_zarr(s3fs.S3Map("s3://hrrrzarr/grid/HRRR_chunk_index.zarr", s3=self.fs),
                           consolidated=False)
        glat = idx["latitude"].values
        glon = idx["longitude"].values % 360
        tree = cKDTree(np.column_stack([glat.ravel(), glon.ravel()]))
        _, flat = tree.query(np.column_stack([self.pixels["Latitude"].to_numpy(),
                                              self.pixels["Longitude"].to_numpy() % 360]))
        self.iy, self.ix = np.unravel_index(flat, glat.shape)
        print(f"Index built: {self.n_pix} pixels")

    def extract(self, date_str, hour, level, var):
        """One variable, one analysis hour, at the pixel points.

        The store path repeats the level ({level}/{var}/{level}); the array
        lives one group deeper than {level}/{var}.
        """
        import s3fs
        import xarray as xr

        path = f"hrrrzarr/sfc/{date_str}/{date_str}_{hour:02d}z_anl.zarr/{level}/{var}/{level}"
        try:
            ds = xr.open_zarr(s3fs.S3Map(root=path, s3=self.fs, check=False), consolidated=False)
            da = ds[var].squeeze()
            y_dim, x_dim = da.dims[-2], da.dims[-1]
            vals = da.isel({y_dim: xr.DataArray(self.iy, dims="points"),
                            x_dim: xr.DataArray(self.ix, dims="points")}).values.astype(np.float32)
            return vals, None
        except Exception as exc:  # missing groups raise several different types
            return None, str(exc)


def create_hrrr_store(path, pixels, variables):
    """Create an empty (time, pixel) NetCDF store for the sampler."""
    import netCDF4 as nc4

    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    ds = nc4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("pixel", len(pixels))
    ds.createDimension("time", None)
    t = ds.createVariable("time", "i8", ("time",), zlib=True)
    t.units, t.calendar = TIME_UNITS, TIME_CALENDAR
    ds.createVariable("pixel_number", "i4", ("pixel",))[:] = pixels["Pixel Number"].to_numpy()
    ds.createVariable("latitude", "f4", ("pixel",))[:] = pixels["Latitude"].to_numpy()
    ds.createVariable("longitude", "f4", ("pixel",))[:] = pixels["Longitude"].to_numpy()
    for name, _, _ in variables:
        ds.createVariable(name, "f4", ("time", "pixel"), zlib=True, complevel=4,
                          chunksizes=(1, len(pixels)), fill_value=np.nan)
    ds.close()


def _append_day(path, times, data):
    """Write data variables first and 'time' last, so an interrupted day is
    invisible to the resume check and gets redone cleanly."""
    import netCDF4 as nc4

    ds = nc4.Dataset(path, "a")
    start = ds.dimensions["time"].size
    n = len(times)
    for name, arr in data.items():
        ds.variables[name][start:start + n, :] = arr
    ds.variables["time"][start:start + n] = nc4.date2num(list(times), units=TIME_UNITS, calendar=TIME_CALENDAR)
    ds.sync()
    ds.close()


def last_stored_time(path):
    """Last timestamp in a sampler NetCDF, or None."""
    import netCDF4 as nc4

    if not os.path.exists(path):
        return None
    with nc4.Dataset(path, "r") as ds:
        n = ds.dimensions["time"].size
        if n == 0:
            return None
        t = nc4.num2date(ds.variables["time"][n - 1], units=TIME_UNITS, calendar=TIME_CALENDAR)
    return pd.Timestamp(str(t))


def sample_hrrr(footprint, out_nc, start, end, variables=None, max_workers=16, max_days=None):
    """Sample HRRR at footprint pixel centres for every hour of start..end (inclusive).

    Parameters
    ----------
    footprint : str or DataFrame
    out_nc : str
        NetCDF store (created if missing, resumed if present).
    variables : "default", "cloud", or list of (name, level, var)
    max_days : int, optional
        Stop after this many days (used by run_hrrr_supervised).
    """
    variables = _resolve_variables(variables)
    s = _Sampler(footprint, max_workers)
    s.build_index()
    if not os.path.exists(out_nc):
        create_hrrr_store(out_nc, s.pixels, variables)
        print(f"Created {out_nc}")

    dates = pd.date_range(start, end, freq="1D")
    last = last_stored_time(out_nc)
    if last is not None:
        dates = dates[dates > last.normalize()]
        print(f"Resuming after {last.date()}: {len(dates)} days left")
    if max_days:
        dates = dates[:max_days]

    durations = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, day in enumerate(dates, 1):
            t0 = time.time()
            date_str = day.strftime("%Y%m%d")
            hours = pd.date_range(day, day + pd.Timedelta(hours=23), freq="1h")
            data = {n: np.full((24, s.n_pix), np.nan, dtype=np.float32) for n, _, _ in variables}
            futs = {pool.submit(s.extract, date_str, ts.hour, lvl, var): (h, name, ts)
                    for h, ts in enumerate(hours) for name, lvl, var in variables}
            failures = []
            for fut in as_completed(futs):
                h, name, ts = futs[fut]
                vals, err = fut.result()
                if err is not None:
                    if not (name in SMOKE_FIELDS and ts < HRRRV4_DATE):
                        failures.append((ts, name, err))
                    continue
                if name == "smoke_near_surface_ugm3":
                    vals = _fix_smoke_units(vals, ts)
                data[name][h, :] = vals
            if failures:
                print(f"  {len(failures)} field failures on {date_str}, e.g. "
                      f"{failures[0][1]} @ {failures[0][0]}: {failures[0][2][:120]}")
            _append_day(out_nc, list(hours), data)
            durations.append(time.time() - t0)
            avg = sum(durations) / len(durations)
            eta = timedelta(seconds=int(avg * (len(dates) - i)))
            print(f"[{i}/{len(dates)}] {day.date()} {durations[-1]:.1f}s | avg {avg:.1f}s/day | ETA {eta}")
            gc.collect()


def run_hrrr_supervised(footprint, out_nc, start, end, variables="default", chunk_days=20,
                        max_workers=16, max_consecutive_failures=3, pause_s=5):
    """Run sample_hrrr in fresh subprocesses of chunk_days days until end is reached.

    footprint must be a CSV path here (it is passed to the subprocess).
    """
    if not isinstance(footprint, (str, os.PathLike)):
        raise TypeError("run_hrrr_supervised needs the footprint as a CSV path")
    end_ts = pd.Timestamp(end)
    var_arg = variables if isinstance(variables, str) else json.dumps([list(v) for v in variables])
    failures, chunk = 0, 0
    while True:
        last = last_stored_time(out_nc)
        if last is not None and last.normalize() >= end_ts:
            print(f"Done: last stored day {last.date()} reached {end_ts.date()}")
            return
        chunk += 1
        print(f"\n=== chunk {chunk}: up to {chunk_days} days "
              f"({'after ' + str(last.date()) if last is not None else 'starting fresh'}) ===")
        cmd = [sys.executable, "-m", "celimes_air_quality_package.hrrr", "--footprint", str(footprint),
               "--out", str(out_nc), "--start", str(start), "--end", str(end), "--variables", var_arg,
               "--max-days", str(chunk_days), "--workers", str(max_workers)]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            failures += 1
            print(f"Chunk exited with code {rc} ({failures}/{max_consecutive_failures})")
            if failures >= max_consecutive_failures:
                raise RuntimeError("Sampler kept failing; see the output above.")
            time.sleep(30)
            continue
        failures = 0
        new_last = last_stored_time(out_nc)
        if last is not None and new_last is not None and new_last <= last:
            print("Warning: chunk finished without appending any days.")
        time.sleep(pause_s)


def verify_hrrr_variable(footprint, level, var, date="20230615", hour=18):
    """Fetch one hour of one field to check a level/var string before a long run."""
    s = _Sampler(footprint)
    s.build_index()
    vals, err = s.extract(date, hour, level, var)
    if err:
        raise RuntimeError(f"{level}/{var} failed: {err}")
    print(f"OK: {len(vals)} pixels, first values {vals[:5]}")
    return vals


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def load_hrrr(nc_path, start=None, end=None, variables=None, footprint=None):
    """Load a sampler NetCDF as a flat DataFrame.

    Columns: Pixel Number, Latitude, Longitude, Datetime (naive UTC), then
    the variables. start/end are inclusive dates. Files written without a
    pixel_number variable (the old cloud-cover sampler) take pixel numbers
    from ``footprint`` (sorted by Pixel Number, as the sampler wrote them).
    """
    import xarray as xr

    with xr.open_dataset(nc_path) as ds:
        if start or end:
            s = pd.Timestamp(start) if start else None
            e = pd.Timestamp(end) + pd.Timedelta(hours=23) if end else None
            ds = ds.sel(time=slice(s, e))
        data_vars = [v for v in ds.data_vars if ds[v].dims == ("time", "pixel")]
        if variables:
            data_vars = [v for v in data_vars if v in variables]
        if "pixel_number" in ds:
            pix = ds["pixel_number"].values.astype(int)
        elif footprint is not None:
            pix = load_footprint(footprint)["Pixel Number"].to_numpy()
        else:
            print("No pixel_number in file and no footprint given; numbering pixels 1..N")
            pix = np.arange(1, ds.sizes["pixel"] + 1)
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        times = pd.to_datetime(ds["time"].values)
        n_t, n_p = len(times), len(pix)
        out = pd.DataFrame({
            "Pixel Number": np.tile(pix, n_t),
            "Latitude": np.tile(lat, n_t),
            "Longitude": np.tile(lon, n_t),
            "Datetime": np.repeat(times.values, n_p),
        })
        for v in data_vars:
            out[v] = ds[v].values.reshape(-1)
    return out


def _main(argv=None):
    p = argparse.ArgumentParser(description="Sample HRRR (hrrrzarr) at footprint pixel centres")
    p.add_argument("--footprint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--variables", default="default", help="'default', 'cloud', or a JSON list")
    p.add_argument("--max-days", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args(argv)
    variables = a.variables if a.variables in VARIABLE_PRESETS else json.loads(a.variables)
    sample_hrrr(a.footprint, a.out, a.start, a.end, variables, a.workers, a.max_days)


if __name__ == "__main__":
    _main()
