"""celimes_air_quality_package

Satellite air-quality pipeline: download GOES ABI / TEMPO / MODIS / ERA5 /
AirNow / AQS data, build pixel footprints, regrid onto them, sample HRRR,
gap-fill GOES AOD and TEMPO NO2, and validate against AERONET.

Submodules: download, footprints, regrid, hrrr, io, gap_filling,
validation, plotting, credentials, utils.
"""

__version__ = "0.2.0"

from .credentials import get_earthdata_credentials, get_era5_credentials
from .download import (download_airdata_data, download_airnow_data, download_aqs_data, download_era5,
                       download_goes, download_goes_aod, download_goes_cloud_mask, download_modis_aod,
                       download_tempo, goes_bucket)
from .footprints import (FootprintGrid, build_goes_footprint, build_tempo_footprint, load_footprint,
                         pixels_within_radius)
from .gap_filling import AOD_ERA5_CONFIG, goes_aod_gap_filling, tempo_no2_gap_filling
from .hrrr import (HRRR_CLOUD_COVER, HRRR_VARIABLES, load_hrrr, run_hrrr_supervised, sample_hrrr,
                   verify_hrrr_variable)
from .io import (combine_goes_hrrr, combine_tempo_hrrr, load_goes_regridded, load_many,
                 load_tempo_regridded, merge_with_met)
from .regrid import build_modis_footprint, regrid_era5, regrid_goes_aod, regrid_modis_aod, regrid_tempo
from .validation import (collocate, compute_stats, extract_filled_at_site, extract_raw_goes_at_site,
                         load_aeronet_lev20, threshold_sensitivity, validate_site)

__all__ = [n for n in dir() if not n.startswith("_")]
