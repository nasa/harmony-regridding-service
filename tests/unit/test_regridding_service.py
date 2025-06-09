"""Tests the regridding service module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import xarray as xr
from harmony_service_lib.message import Message as HarmonyMessage
from harmony_service_lib.message import Source as HarmonySource

from harmony_regridding_service.regridding_service import regrid

test_scale_extent = {
    'x': {'min': -180, 'max': 180},
    'y': {'min': -90, 'max': 90},
}


@pytest.mark.parametrize(
    'width, height, scale_extent, expected_width, expected_height, description',
    [
        (100, 50, test_scale_extent, 100, 50, 'Grid parameters are provided.'),
        (None, None, None, 5, 9, 'Grid parameters are excluded from message.'),
    ],
)
def test_regrid_projected_data_end_to_end(
    width,
    height,
    scale_extent,
    expected_width,
    expected_height,
    description,
    smap_projected_netcdf_file,
    tmp_path,
):
    """Test the full regrid process for projected input data."""
    input_filename = str(smap_projected_netcdf_file)
    output_filename = str(tmp_path / 'regridded_output.nc')
    logger_mock = MagicMock()

    # Define a target CRS [and optionally grid parameters]
    params = {
        'format': {
            'mime': 'application/x-netcdf',
            'crs': 'EPSG:4326',
            'width': width,
            'height': height,
            'scaleExtent': scale_extent,
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
        assert 'crs' in ds_out, description

        assert ds_out.dims['y'] == expected_height, description
        assert ds_out.dims['x'] == expected_width, description

        assert 'sm_profile_forecast' in ds_out['Forecast_Data'], description
        assert 'sm_profile_analysis' in ds_out['Analysis_Data'], description
        assert 'tb_v_obs' in ds_out['Observations_Data'], description

        assert (
            ds_out['/Metadata/DatasetIdentification'].attrs['shortName'] == 'SPL4SMAU'
        ), description

        assert (
            ds_out['Observations_Data/tb_v_obs'].attrs['long_name']
            == 'Composite resolution observed (L2_SM_AP or L1C_TB) V-pol ...'
        ), description
