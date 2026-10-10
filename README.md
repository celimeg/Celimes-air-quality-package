# celimes_air_quality_package

Download, regrid, gap-fill and validate GOES-R ABI aerosol optical depth and TEMPO NO2, with HRRR meteorology.

```
download  ->  footprints  ->  regrid  ->  hrrr  ->  io (merge)  ->  gap_filling  ->  validation / plotting
```

## Install

```bash
pip install -e .            # from this folder
pip install -e ".[all]"     # + ERA5 (cdsapi), MODIS (pyhdf) and pytest
```

## Credentials

The package never stores keys in code. Each downloader takes the key as an argument, and if you leave it out it reads an environment variable:

| Service | Environment variable(s) | Used by |
|---|---|---|
| AirNow | `AIRNOW_API_KEY` | `download_airnow_data` |
| EPA AQS | `AQS_EMAIL`, `AQS_KEY` | `download_aqs_data` |
| LAADS DAAC | `LAADS_TOKEN` | `download_modis_aod` |
| Copernicus CDS | `CDS_API_KEY` (or `~/.cdsapirc`) | `download_era5` |
| NASA Earthdata | `~/.netrc` (run `get_earthdata_credentials()` once) | `download_tempo` |

GOES and HRRR come from public AWS buckets and need no login.

## Quick start: one site, GOES AOD

```python
import celimes_air_quality_package as cap

# 1. Download GOES-West CONUS AOD (picks GOES-17 or GOES-18 automatically by date)
cap.download_goes_aod("2023-08-01", "2023-08-31", r"Z:\Nasa_Project\GOES_ABI")

# 2. Footprint: regular lat/lon grid matched to the local ABI pixel spacing
fp = cap.build_goes_footprint(r"Z:\Nasa_Project\GOES_ABI\2023\213\18",
                              center=(39.5413, -119.8136), half_width_km=40,
                              out_csv=r"...\Reno\GOES_ABI_footprint_reno.csv")

# 3. Regrid (area-overlap x DQF weighted, all scans in each hour pooled)
cap.regrid_goes_aod(r"Z:\Nasa_Project\GOES_ABI", fp, r"...\Reno\Reno_AOD_regridded.csv",
                    "2023-08-01", "2023-08-31")

# 4. HRRR at the same pixels (chunked subprocesses, resumable)
cap.run_hrrr_supervised(r"...\Reno\GOES_ABI_footprint_reno.csv", r"Z:\Nasa_Project\HRRR\Reno\Reno_goes_hrrr_2023.nc",
                        "2023-01-01", "2023-12-31")

# 5. Merge and gap fill
merged, hrrr_df = cap.combine_goes_hrrr(r"...\Reno\Reno_AOD_regridded.csv",
                                        r"Z:\Nasa_Project\HRRR\Reno\Reno_goes_hrrr_2023.nc")
cap.goes_aod_gap_filling(merged, r"...\Reno\Reno_Gap_filled.csv", "2023-08-01", "2023-08-31")

# 6. Validate against AERONET
ae = cap.load_aeronet_lev20(r"Z:\Nasa_Project\AERONET_AOD\sites\Reno\20200101_20241231_Reno.lev20")
raw = cap.extract_raw_goes_at_site(r"Z:\Nasa_Project\GOES_ABI", 39.5413, -119.8136, "2023-08-01", "2023-08-31")
filled = cap.extract_filled_at_site(r"...\Reno\Reno_Gap_filled.csv", 39.5413, -119.8136)
res = cap.validate_site(raw, filled, ae, threshold=25)

from celimes_air_quality_package import plotting
plotting.plot_validation_scatter(res["raw_merged"], res["filled_merged"])
```

## Quick start: TEMPO NO2

```python
cap.download_tempo("2024-01-01", "2024-01-31", r"Z:\Nasa_Project\TEMPO_NO2\NO2_retrievals\L3_data\organized")
fp = cap.build_tempo_footprint(r"...\organized\2024\01\03\19", center=(37.913, -122.336), half_width_km=30)
cap.regrid_tempo(r"...\organized", fp, r"...\Richmond\Resampled_TEMPO_NO2_Richmond.h5")   # native grid
merged, hrrr_df = cap.combine_tempo_hrrr(r"...\Richmond\Resampled_TEMPO_NO2_Richmond.h5", r"...\Richmond_Tempo_hrrr.nc")
cap.tempo_no2_gap_filling(merged, r"...\Richmond\Richmond_filled_no2.csv",
                          "2023-08-01", "2024-12-31", hrrr_data=hrrr_df)
```

`regrid_tempo(..., method="area")` puts TEMPO onto any other footprint, such as a GOES grid.

## Conventions

* **Times** are UTC everywhere and stored as naive timestamps. Outputs write `YYYY-MM-DD HH:MM:SS`; readers also accept the older `8/6/2020 19:00` strings.
* **Footprints** are `Pixel Number, Latitude, Longitude` (pixel centres). Builders number pixels with longitude ascending in the outer loop and latitude descending in the inner loop, so pixel 1 is the north-west corner. `FootprintGrid` handles any numbering order.
* **Resuming**: every long-running function appends hour by hour (HRRR: day by day) and skips what is already written, so rerunning the same call continues where it stopped.
* **Figures** are returned, not saved. Pass `save_path=` to write a file.

## Gap-filling outputs

| Product | Value column | Flag column | Flag meaning |
|---|---|---|---|
| AOD | `filled_AOD` | `pct_DQF0_filled` | original `pct_DQF0`, or **200** where the value was filled or changed |
| NO2 | `filled_vertical_column_troposphere` / `_stratosphere` | `<column>_qc_filled` | original QC, **200** where filled, **201** where the original had QC 0 but a negative column |

AOD gaps are pixels with `pct_DQF0` below `pct_dqf0_thresh` (default 10). NO2 anchors are pixels with QC 0 and a non-negative column.

Hours with solar zenith above 70 degrees at the domain centre are written unfilled (AOD only).

Defaults reproduce the HRRR versions of `GOES_AOD_GAP_FILL.ipynb` and `TEMPO_NO2_GAP_FILL.ipynb`. For the original ERA5 configuration, use `goes_aod_gap_filling(..., **cap.AOD_ERA5_CONFIG)`.

## Where behaviour differs from the original notebooks

The gap filling itself is unchanged; `tests/test_matches_notebooks.py` checks this against the notebook code. The other differences are:

1. **TEMPO area regridding** (`method="area"`) divides each field by the overlap area of the non-missing retrievals only. The notebook divided by the total overlap area, which pulled means toward zero wherever some TEMPO cells were missing. QC fill values (-9999) are treated as missing, not averaged in. Native-grid regridding is unchanged.
2. **Filled AOD at a site** (`extract_filled_at_site`):
   * `use="filled"` (default) matches the notebook and averages only pixels flagged 200. `use="all"` averages every pixel with a value.
   * The notebook's longitude pre-filter clipped the 7.5 km circle to about +/-5.7 km east-west at 40N. The full circle is now used; pass `legacy_box=True` to reproduce the old numbers.
3. **Statistics**: `rmse` is still the residual RMSE about the fitted regression line, as in the notebooks. `rmse_1to1` (GOES minus AERONET) is added alongside it.
4. **GOES download** handles the GOES-17 to GOES-18 switch at 2023-01-04 18:00 UTC, and the GOES-East switch from 16 to 19 at 2025-04-07 15:00 UTC (`satellite="east"`).
5. **AQS downloads** keep a `Datetime_GMT` column next to the local `Datetime`.

## Tests

```bash
pytest
```

The tests build synthetic GOES/TEMPO/HRRR data and compare the package to verbatim copies of the notebook code in `tests/reference/`. Covered:

* AOD and NO2 gap filling, at three grid sizes
* GOES regridding
* TEMPO native regridding
* GOES footprint construction
* an end-to-end pipeline run
