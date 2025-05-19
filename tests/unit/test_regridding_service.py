"""Tests the regridding service module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import xarray as xr
from harmony_service_lib.message import Message as HarmonyMessage
from harmony_service_lib.message import Source as HarmonySource

from harmony_regridding_service.regridding_service import regrid


def test_regrid_projected_data_end_to_end(smap_projected_netcdf_file, tmp_path):
    """Test the full regrid process for projected input data."""
    input_filename = str(smap_projected_netcdf_file)
    output_filename = str(tmp_path / 'regridded_output.nc')
    logger_mock = MagicMock()

    # Define a target CRS and grid (example: Geographic WGS84)
    params = {
        'format': {
            'mime': 'application/x-netcdf',
            'crs': 'EPSG:4326',
            'width': 100,
            'height': 50,
            'scaleExtent': {
                'x': {'min': -180, 'max': 180},
                'y': {'min': -90, 'max': 90},
            },
        },
        'sources': [{'collection': 'C123-TEST', 'shortName': 'SPL4SMAU'}],
    }
    message = HarmonyMessage(params)
    source = HarmonySource({'collection': 'C123-TEST', 'shortName': 'SPL4SMAU'})

    # Mock generate_output_filename to control the output path
    with patch(
        'harmony_regridding_service.regridding_service.generate_output_filename',
        return_value=output_filename,
    ):
        result_filename = regrid(message, input_filename, source, logger_mock)

    assert result_filename == output_filename
    assert Path(output_filename).exists()

    with xr.open_datatree(output_filename) as ds_out:
        assert 'crs' in ds_out

        assert ds_out.dims['y'] == 50
        assert ds_out.dims['x'] == 100

        assert 'sm_profile_forecast' in ds_out['Forecast_Data']
        assert 'sm_profile_analysis' in ds_out['Analysis_Data']
        assert 'tb_v_obs' in ds_out['Observations_Data']

        assert (
            ds_out['/Metadata/DatasetIdentification'].attrs['shortName'] == 'SPL4SMAU'
        )

        assert (
            ds_out['Observations_Data/tb_v_obs'].attrs['long_name']
            == 'Composite resolution observed (L2_SM_AP or L1C_TB) V-pol ...'
        )
