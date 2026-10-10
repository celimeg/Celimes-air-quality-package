"""Downloaders: GOES ABI (AOD, cloud mask), TEMPO L3, MODIS, ERA5, AirNow, EPA AQS.

All downloads skip files that already exist, so an interrupted run can be
restarted with the same call. Partial files are written as ``*.part`` and
renamed only when complete, so a crash never leaves a truncated file that
would later be skipped as "already downloaded".
"""

import calendar
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import requests
from tqdm import tqdm

from .credentials import (check_cds_credentials, check_earthdata_credentials, env_or_arg,
                          write_cdsapirc)

logger = logging.getLogger(__name__)


def _days(start, end):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    out = []
    while s <= e:
        out.append(s)
        s += timedelta(days=1)
    return out


def _stream_to_file(response, out_path, chunk=1024 * 1024):
    tmp = out_path + ".part"
    with open(tmp, "wb") as f:
        for block in response.iter_content(chunk_size=chunk):
            f.write(block)
    os.replace(tmp, out_path)


# ==================================================================
# GOES ABI
# ==================================================================

# GOES-West: GOES-17 -> GOES-18 operational cutover 2023-01-04 18:00 UTC.
# GOES-East: GOES-16 -> GOES-19 operational cutover 2025-04-07 15:00 UTC.
GOES_CUTOVERS = {
    "west": (pd.Timestamp("2023-01-04 18:00"), "noaa-goes17", "noaa-goes18"),
    "east": (pd.Timestamp("2025-04-07 15:00"), "noaa-goes16", "noaa-goes19"),
}


def goes_bucket(when, satellite="west"):
    """NOAA Open Data bucket serving the operational GOES-West/East at a UTC time."""
    cutover, before, after = GOES_CUTOVERS[satellite]
    return after if pd.Timestamp(when) >= cutover else before


def download_goes(start, end, output_dir, product="ABI-L2-AODC", satellite="west",
                  hours=range(24), bucket=None):
    """Download GOES ABI files from AWS (anonymous) for a date range.

    Files are saved as ``output_dir/YYYY/DOY/HH/<file>.nc``, the layout the
    regridders expect.

    Parameters
    ----------
    start, end : str
        Inclusive dates, "YYYY-MM-DD".
    output_dir : str
        Root folder for this product, e.g. r"Z:\\Nasa_Project\\GOES_ABI".
    product : str
        "ABI-L2-AODC" (CONUS AOD, default), "ABI-L2-ACMC" (clear-sky mask), etc.
    satellite : "west" or "east"
        Picks the operational satellite for each hour (handles the
        GOES-17/18 and GOES-16/19 cutovers). Ignored if bucket is given.
    hours : iterable of int
        UTC hours to fetch.
    bucket : str, optional
        Force a specific bucket, e.g. "noaa-goes18".
    """
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    n_new = 0
    for day in tqdm(_days(start, end), unit="day", desc=product):
        doy = day.timetuple().tm_yday
        for hour in hours:
            when = day + timedelta(hours=hour)
            b = bucket or goes_bucket(when, satellite)
            prefix = f"{b}/{product}/{day.year}/{doy:03d}/{hour:02d}/"
            try:
                files = fs.ls(prefix)
            except FileNotFoundError:
                continue
            outdir = os.path.join(output_dir, str(day.year), f"{doy:03d}", f"{hour:02d}")
            os.makedirs(outdir, exist_ok=True)
            for f in files:
                local = os.path.join(outdir, f.split("/")[-1])
                if os.path.exists(local):
                    continue
                fs.get(f, local + ".part")
                os.replace(local + ".part", local)
                n_new += 1
    print(f"Done: {n_new} new {product} files under {output_dir}")


def download_goes_aod(start, end, output_dir, satellite="west", hours=range(24)):
    """GOES ABI CONUS AOD (ABI-L2-AODC). See download_goes."""
    download_goes(start, end, output_dir, "ABI-L2-AODC", satellite, hours)


def download_goes_cloud_mask(start, end, output_dir, satellite="west", hours=range(24)):
    """GOES ABI CONUS clear-sky mask (ABI-L2-ACMC). See download_goes."""
    download_goes(start, end, output_dir, "ABI-L2-ACMC", satellite, hours)


# ==================================================================
# TEMPO (NASA Earthdata / CMR)
# ==================================================================

CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"


def _tempo_granule_time(title):
    """Start time from a TEMPO granule name, e.g. TEMPO_NO2_L3_V03_20230802T151249Z_S001."""
    for part in title.split("_"):
        if len(part) == 16 and part[8] == "T" and part.endswith("Z"):
            return datetime.strptime(part, "%Y%m%dT%H%M%SZ")
    raise ValueError(f"No timestamp in granule name {title!r}")


def download_tempo(start, end, output_dir, product="TEMPO_NO2_L3", version="V03",
                   bbox=None, page_size=2000):
    """Download TEMPO granules from NASA Earthdata.

    Uses ~/.netrc for login (run get_earthdata_credentials() once). Files are
    organized as ``output_dir/YYYY/MM/DD/HH/<granule>.nc``, the layout used
    by regrid_tempo.

    Parameters
    ----------
    start, end : str
        Inclusive dates, "YYYY-MM-DD".
    product, version : str
        CMR short name and version, e.g. "TEMPO_NO2_L3", "V03".
    bbox : (lat_min, lat_max, lon_min, lon_max), optional
        Only granules intersecting this box. TEMPO L3 granules cover the full
        field of regard, so this mostly matters for L2.
    """
    check_earthdata_credentials()
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    params = {
        "short_name": product,
        "version": version,
        "temporal": f"{s:%Y-%m-%dT00:00:00Z},{e:%Y-%m-%dT23:59:59Z}",
        "page_size": page_size,
    }
    if bbox is not None:
        lat_min, lat_max, lon_min, lon_max = bbox
        params["bounding_box"] = f"{lon_min},{lat_min},{lon_max},{lat_max}"

    session = requests.Session()  # trust_env=True: picks up ~/.netrc, including on redirect to URS
    granules, search_after = [], None
    while True:
        headers = {"CMR-Search-After": search_after} if search_after else {}
        r = session.get(CMR_GRANULES, params=params, headers=headers, timeout=60)
        r.raise_for_status()
        page = r.json()["feed"]["entry"]
        granules.extend(page)
        search_after = r.headers.get("CMR-Search-After")
        if not page or not search_after:
            break
    print(f"Found {len(granules)} {product} {version} granules")

    n_new = n_skip = 0
    for g in tqdm(granules, unit="granule"):
        title = g["title"]
        try:
            t = _tempo_granule_time(title)
        except ValueError as exc:
            logger.warning(str(exc))
            continue
        url = next((l["href"] for l in g.get("links", [])
                    if l.get("rel", "").endswith("/data#") and l["href"].startswith("http")), None)
        if url is None:
            logger.warning("No download link for %s", title)
            continue
        name = title if title.endswith(".nc") else title + ".nc"
        outdir = os.path.join(output_dir, f"{t:%Y}", f"{t:%m}", f"{t:%d}", f"{t:%H}")
        out = os.path.join(outdir, name)
        if os.path.exists(out):
            n_skip += 1
            continue
        os.makedirs(outdir, exist_ok=True)
        try:
            with session.get(url, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                _stream_to_file(resp, out)
            n_new += 1
        except requests.RequestException as exc:
            logger.warning("Failed %s: %s", title, exc)
    print(f"Done: {n_new} downloaded, {n_skip} already existed")


# ==================================================================
# MODIS (LAADS DAAC)
# ==================================================================

MODIS_AOD_PRODUCTS = {"MOD04_L2", "MOD04_3K", "MYD04_L2", "MYD04_3K"}


def download_modis_aod(data_dir, product, start, end, xmin, ymin, xmax, ymax,
                       token=None, collection=61):
    """Download MODIS AOD granules overlapping a box from LAADS DAAC.

    Files go to ``data_dir/modis/<product>/``. token defaults to $LAADS_TOKEN
    (get one at https://ladsweb.modaps.eosdis.nasa.gov/profiles/).
    """
    if product not in MODIS_AOD_PRODUCTS:
        raise ValueError(f"product must be one of {sorted(MODIS_AOD_PRODUCTS)}")
    token = env_or_arg(token, "LAADS_TOKEN", "LAADS DAAC token")
    out_dir = os.path.join(data_dir, "modis", product)
    os.makedirs(out_dir, exist_ok=True)
    base = f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/{collection}/{product}"
    headers = {"Authorization": f"Bearer {token}"}

    n_new = n_skip = n_out = 0
    for day in tqdm(_days(start, end), unit="day"):
        doy = day.timetuple().tm_yday
        try:
            r = requests.get(f"{base}/{day.year}/{doy:03d}.json", headers=headers, timeout=30)
            r.raise_for_status()
            contents = r.json().get("content", [])
        except requests.RequestException as exc:
            logger.warning("Listing failed for %s: %s", day.date(), exc)
            continue
        for meta in contents:
            if meta.get("size", 0) <= 0:
                continue
            bb = meta.get("bbox")
            if bb and (bb["east"] < xmin or bb["west"] > xmax or bb["north"] < ymin or bb["south"] > ymax):
                n_out += 1
                continue
            out = os.path.join(out_dir, meta["name"])
            if os.path.exists(out):
                n_skip += 1
                continue
            try:
                with requests.get(f"{base}/{day.year}/{doy:03d}/{meta['name']}", headers=headers,
                                  timeout=120, stream=True) as r2:
                    r2.raise_for_status()
                    _stream_to_file(r2, out)
                n_new += 1
            except requests.RequestException as exc:
                logger.warning("Failed %s: %s", meta["name"], exc)
    print(f"Done: {n_new} downloaded, {n_skip} existed, {n_out} outside the box. Files in {out_dir}")


# ==================================================================
# ERA5 (Copernicus CDS)
# ==================================================================

ERA5_VARIABLES = [
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "100m_u_component_of_wind", "100m_v_component_of_wind",
    "2m_temperature", "2m_dewpoint_temperature", "surface_pressure",
    "boundary_layer_height", "convective_available_potential_energy",
    "near_ir_albedo_for_diffuse_radiation", "near_ir_albedo_for_direct_radiation",
    "uv_visible_albedo_for_diffuse_radiation", "uv_visible_albedo_for_direct_radiation",
]


def download_era5(data_dir, lat_min, lat_max, lon_min, lon_max, start, end,
                  times=None, api_key=None, variables=None):
    """Download ERA5 single-level reanalysis for a box, merged into one NetCDF.

    Output: ``data_dir/era5/<start>_to_<end>_era5.nc``. api_key defaults to
    $CDS_API_KEY, then an existing ~/.cdsapirc.
    """
    import cdsapi
    import xarray as xr

    api_key = api_key or os.environ.get("CDS_API_KEY")
    if api_key:
        write_cdsapirc(api_key)
    else:
        check_cds_credentials()

    times = times or [f"{h:02d}:00" for h in range(24)]
    era5_dir = os.path.join(data_dir, "era5")
    tmp_dir = os.path.join(era5_dir, "_temp_era5_monthly")
    os.makedirs(tmp_dir, exist_ok=True)
    final = os.path.join(era5_dir, f"{start}_to_{end}_era5.nc")

    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    months, cur = [], s.replace(day=1)
    while cur <= e:
        months.append((cur.year, cur.month))
        cur = cur.replace(year=cur.year + (cur.month == 12), month=cur.month % 12 + 1)

    client = cdsapi.Client()
    month_files = []
    for year, month in months:
        days = [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)
                if s <= datetime(year, month, d) <= e]
        f = os.path.join(tmp_dir, f"era5_{year}{month:02d}.nc")
        if not os.path.exists(f):
            print(f"Requesting ERA5 {year}-{month:02d} ({len(days)} days)...")
            try:
                client.retrieve("reanalysis-era5-single-levels", {
                    "product_type": ["reanalysis"], "variable": variables or ERA5_VARIABLES,
                    "year": [str(year)], "month": [f"{month:02d}"], "day": days, "time": times,
                    "data_format": "netcdf", "download_format": "unarchived",
                    "area": [lat_max, lon_min, lat_min, lon_max],
                }).download(f)
            except Exception as exc:  # cdsapi raises plain Exceptions
                print(f"  Failed {year}-{month:02d}: {exc}")
                if os.path.exists(f):
                    os.remove(f)
                continue
        month_files.append(f)

    if not month_files:
        raise FileNotFoundError("No ERA5 files downloaded; check credentials, box and dates.")
    with xr.open_mfdataset(month_files, combine="by_coords") as ds:
        ds.to_netcdf(final)
    for f in month_files:
        os.remove(f)
    os.rmdir(tmp_dir)
    print(f"Saved {final}")
    return final


# ==================================================================
# AirNow
# ==================================================================

_AIRNOW_LABELS = {"PM25": "PM2.5", "PM10": "PM10", "CO": "CO", "NO2": "NO2", "SO2": "SO2", "O3": "OZONE"}
_AIRNOW_HEADERS = ["Latitude", "Longitude", "Datetime", "Pollutant", "Concentration", "Units",
                   "AQI", "Monitor_Type", "Site_Name", "Site_Agency", "AQS_ID", "Full_AQS_ID"]


def download_airnow_data(data_dir, pollutants, start, end, xmin, ymin, xmax, ymax,
                         api_key=None, timeout=60):
    """Download AirNow hourly data for a box, pivoted to one column per pollutant.

    start/end are inclusive "YYYY-MM-DD". Datetimes are UTC. api_key defaults
    to $AIRNOW_API_KEY. Saved to ``data_dir/airnow/<start>_to_<end>_airnow.csv``.
    """
    api_key = env_or_arg(api_key, "AIRNOW_API_KEY", "AirNow API key")
    out_dir = os.path.join(data_dir, "airnow")
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, f"{start}_to_{end}_airnow.csv")

    xm, ym = (xmin + xmax) / 2, (ymin + ymax) / 2
    boxes = [f"{xmin},{ymin},{xm},{ym}", f"{xm},{ymin},{xmax},{ym}",
             f"{xmin},{ym},{xm},{ymax}", f"{xm},{ym},{xmax},{ymax}"]
    chunks = []
    for day in tqdm(_days(start, end), unit="day"):
        for b in boxes:
            params = {
                "startDate": f"{day:%Y-%m-%d}T00", "endDate": f"{day:%Y-%m-%d}T23",
                "parameters": ",".join(pollutants), "BBOX": b, "dataType": "B",
                "format": "text/csv", "verbose": 1, "monitorType": 0,
                "includerawconcentrations": 0, "API_KEY": api_key,
            }
            try:
                r = requests.get("https://www.airnowapi.org/aq/data/", params=params, timeout=timeout)
                r.raise_for_status()
            except requests.RequestException as exc:
                logger.warning("AirNow request failed: %s", exc)
                continue
            if r.text.strip():
                from io import StringIO
                chunks.append(pd.read_csv(StringIO(r.text), header=None, names=_AIRNOW_HEADERS))
    if not chunks:
        raise FileNotFoundError("No AirNow data returned; check key, box and dates.")

    df = pd.concat(chunks, ignore_index=True)
    df["Concentration"] = pd.to_numeric(df["Concentration"], errors="coerce")
    df = df.drop_duplicates(subset=["Latitude", "Longitude", "Datetime", "Site_Name", "AQS_ID", "Pollutant"])
    wide = df.pivot_table(index=["Latitude", "Longitude", "Datetime", "Site_Name", "Site_Agency",
                                 "AQS_ID", "Full_AQS_ID"],
                          columns="Pollutant", values="Concentration", aggfunc="first").reset_index()
    wide.columns.name = None
    for p in pollutants:
        col = _AIRNOW_LABELS.get(p, p)
        if col not in wide.columns:
            wide[col] = float("nan")
    wide.to_csv(final, index=False)
    print(f"Saved {final}")
    return wide


# ==================================================================
# EPA AQS (AirData API)
# ==================================================================

_AQS_PARAM_MAP = {
    "Ozone": "Ozone", "Carbon monoxide": "CO", "PM2.5 - Local Conditions": "PM2.5",
    "Nitrogen dioxide (NO2)": "NO2", "PM10 Total 0-10um STP": "PM10", "Sulfur dioxide": "SO2",
}


def download_aqs_data(data_dir, params, start, end, minlat, minlon, maxlat, maxlon,
                      email=None, key=None, timeout=60):
    """Download EPA AQS sample data for a box, pivoted to one column per pollutant.

    params are AQS parameter codes, e.g. ["88101", "42602"]
    (88101 PM2.5, 42101 CO, 44201 O3, 42602 NO2, 81102 PM10, 42401 SO2).
    Output has both ``Datetime`` (local standard time, as AQS reports it)
    and ``Datetime_GMT``. Ozone is converted to ppb. email/key default to
    $AQS_EMAIL / $AQS_KEY. Saved to ``data_dir/aqs/<start>_to_<end>_aqs.csv``.
    """
    email = env_or_arg(email, "AQS_EMAIL", "AQS email")
    key = env_or_arg(key, "AQS_KEY", "AQS key")
    out_dir = os.path.join(data_dir, "aqs")
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, f"{start}_to_{end}_aqs.csv")

    frames = []
    for day in tqdm(_days(start, end), unit="day"):
        d = f"{day:%Y%m%d}"
        q = {"email": email, "key": key, "param": ",".join(params), "bdate": d, "edate": d,
             "minlat": minlat, "maxlat": maxlat, "minlon": minlon, "maxlon": maxlon}
        try:
            r = requests.get("https://aqs.epa.gov/data/api/sampleData/byBox", params=q, timeout=timeout)
            r.raise_for_status()
            data = r.json().get("Data", [])
        except (requests.RequestException, ValueError) as exc:
            logger.warning("AQS request failed for %s: %s", d, exc)
            continue
        if data:
            frames.append(pd.DataFrame(data))
    if not frames:
        raise FileNotFoundError("No AQS data returned; check credentials, box and dates.")

    df = pd.concat(frames, ignore_index=True)
    for col, width in [("state_code", 2), ("county_code", 3), ("site_number", 4)]:
        df[col] = df[col].astype(int).astype(str).str.zfill(width)
    df["AQS_ID"] = df["state_code"] + df["county_code"] + df["site_number"]
    df["Datetime"] = pd.to_datetime(df["date_local"] + " " + df["time_local"], errors="coerce")
    df["Datetime_GMT"] = pd.to_datetime(df["date_gmt"] + " " + df["time_gmt"], errors="coerce")
    df = df.dropna(subset=["Datetime"])
    df["parameter"] = df["parameter"].map(_AQS_PARAM_MAP)
    df = df.dropna(subset=["parameter"])
    df.loc[df["parameter"] == "Ozone", "sample_measurement"] *= 1000

    wide = df.pivot_table(index=["latitude", "longitude", "AQS_ID", "Datetime", "Datetime_GMT"],
                          columns="parameter", values="sample_measurement", aggfunc="first").reset_index()
    wide.columns.name = None
    wide.to_csv(final, index=False)
    print(f"Saved {final}")
    return wide


# Old name kept for existing notebooks
download_airdata_data = download_aqs_data
