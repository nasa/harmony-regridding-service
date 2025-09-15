"""Tests the regridding service module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import xarray as xr
from harmony_service_lib.message import Message as HarmonyMessage
from harmony_service_lib.message import Source as HarmonySource

from harmony_regridding_service.exceptions import InvalidVariableRequest
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


def test_regrid_smap_file(
    test_spl3ftp_ncfile,
    tmp_path,
):
    """Test the full regrid process for projected input data."""
    input_filename = str(test_spl3ftp_ncfile)
    output_filename = str(tmp_path / 'regridded_output.nc')
    logger_mock = MagicMock()

    # Define a target CRS [and optionally grid parameters]
    params = {
        'format': {
            'mime': 'application/x-netcdf',
            'crs': 'EPSG:4326',
        },
        'sources': [{'collection': 'C123-test', 'shortName': 'SPL3FTP'}],
    }
    message = HarmonyMessage(params)
    source = HarmonySource({'collection': 'C123-TEST', 'shortName': 'SPL3FTP'})

    # Mock generate_output_filename to control the output path
    with patch(
        'harmony_regridding_service.regridding_service.generate_output_filename',
        return_value=output_filename,
    ):
        result_filename = regrid(message, input_filename, source, logger_mock)

    assert result_filename == output_filename
    assert Path(output_filename).exists()

    expected_groups = [
        '/Freeze_Thaw_Retrieval_Data_Polar',
        '/Freeze_Thaw_Retrieval_Data_Global',
    ]
    expected = {
        '/Freeze_Thaw_Retrieval_Data_Polar': {
            'width': 263,
            'height': 122,
        },
        '/Freeze_Thaw_Retrieval_Data_Global': {
            'width': 186,
            'height': 73,
        },
    }

    with xr.open_datatree(output_filename) as dt:
        for group in expected_groups:
            expects = expected[group]

            assert 'crs' in dt[group], f'failed: {group}'

            assert dt[group].dims['y'] == expects['height'], f'failed: {group}'
            assert dt[group].dims['x'] == expects['width'], f'failed: {group}'

            assert 'longitude' in dt[group], f'failed: {group}'
            assert 'latitude' in dt[group], f'failed: {group}'
            assert 'altitude_dem' in dt[group], f'failed: {group}'


def test_regrid_smap_excluded_variable_file(
    test_spl3ftp_ncfile,
    tmp_path,
):
    """Test the full regrid process with excluded variables.

    This test adds a test varinfo config that excludes science variables:
    "/.*altitude_dem.*"

    Is it the same as the previous test, but the last assertion is that the
    variable is not in the output.

    """
    input_filename = str(test_spl3ftp_ncfile)
    output_filename = str(tmp_path / 'regridded_output.nc')
    logger_mock = MagicMock()

    # Define a target CRS [and optionally grid parameters]
    params = {
        'format': {
            'mime': 'application/x-netcdf',
            'crs': 'EPSG:4326',
        },
        'sources': [{'collection': 'C123-test', 'shortName': 'SPL3FTP'}],
    }
    message = HarmonyMessage(params)
    source = HarmonySource({'collection': 'C123-TEST', 'shortName': 'SPL3FTP'})

    # Mock generate_output_filename to control the output path
    with (
        patch(
            'harmony_regridding_service.regridding_service.generate_output_filename',
            return_value=output_filename,
        ),
        patch(
            'harmony_regridding_service.regridding_service.varinfo_config_filename',
            return_value=str(
                Path(Path(__file__).parent / 'fixtures/test_HRS_varinfo_config.json')
            ),
        ),
    ):
        result_filename = regrid(message, input_filename, source, logger_mock)

    assert result_filename == output_filename
    assert Path(output_filename).exists()

    expected_groups = [
        '/Freeze_Thaw_Retrieval_Data_Polar',
        '/Freeze_Thaw_Retrieval_Data_Global',
    ]
    expected = {
        '/Freeze_Thaw_Retrieval_Data_Polar': {
            'width': 263,
            'height': 122,
        },
        '/Freeze_Thaw_Retrieval_Data_Global': {
            'width': 186,
            'height': 73,
        },
    }

    with xr.open_datatree(output_filename) as dt:
        for group in expected_groups:
            expects = expected[group]

            assert 'crs' in dt[group], f'failed: {group}'

            assert dt[group].dims['y'] == expects['height'], f'failed: {group}'
            assert dt[group].dims['x'] == expects['width'], f'failed: {group}'

            assert 'longitude' in dt[group], f'failed: {group}'
            assert 'latitude' in dt[group], f'failed: {group}'
            ### This is the change from the previous test. altitude_dem is
            ### configured to be an excluded science variable

            assert 'altitude_dem' not in dt[group], f'failed: {group}'


def test_regrid_smap_bad_user_requested_variable_data_end_to_end(
    test_spl3ftp_ncfile,
    tmp_path,
):
    """Test a Request that specifies an explicitly excluded variable.

    This test repeats the previous test but a new test config is used that
    explicitly excludes that user's variable. we expect this request to fail.

    """
    input_filename = str(test_spl3ftp_ncfile)
    output_filename = str(tmp_path / 'regridded_output.nc')
    logger_mock = MagicMock()

    # Define a user selected variable
    user_var = {
        'id': 'V12345789-EEDTEST',
        'name': 'Freeze_Thaw_Retrieval_Data_Global/altitude_dem',
    }

    # Define a target CRS [and optionally grid parameters]
    params = {
        'format': {
            'mime': 'application/x-netcdf',
            'crs': 'EPSG:4326',
        },
        'sources': [
            {'collection': 'C123-test', 'shortName': 'SPL3FTP', 'variables': [user_var]}
        ],
    }
    message = HarmonyMessage(params)
    source = HarmonySource(message['sources'][0])

    # Mock generate_output_filename to control the output path
    with (
        patch(
            'harmony_regridding_service.regridding_service.generate_output_filename',
            return_value=output_filename,
        ),
        patch(
            'harmony_regridding_service.regridding_service.varinfo_config_filename',
            return_value=str(
                Path(Path(__file__).parent / 'fixtures/test_HRS_varinfo_config.json')
            ),
        ),
    ):
        # when harmony-service-lib is updated use this.
        # expected_message = re.escape(
        #     r"Request for unprocessable variable(s): "
        #     "{'/Freeze_Thaw_Retrieval_Data_Global/altitude_dem'}."
        # )
        # with pytest.raises(InvalidVariableRequest, match=expected_message):
        #     regrid(message, input_filename, source, logger_mock)

        expected_message = (
            'Request for unprocessable variable(s): '
            "{'/Freeze_Thaw_Retrieval_Data_Global/altitude_dem'}."
        )

        with pytest.raises(InvalidVariableRequest) as exception_info:
            regrid(message, input_filename, source, logger_mock)

        assert exception_info.value.message == expected_message
