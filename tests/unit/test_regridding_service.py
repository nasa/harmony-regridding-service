"""Tests the regridding service module."""

import json
import logging
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import xarray as xr
from harmony_service_lib.message import Message as HarmonyMessage
from harmony_service_lib.message import Source as HarmonySource

from harmony_regridding_service.exceptions import InvalidVariableRequest
from harmony_regridding_service.provenance import PROGRAM, get_semantic_version
from harmony_regridding_service.regridding_service import regrid

test_scale_extent = {
    'x': {'min': -180, 'max': 180},
    'y': {'min': -90, 'max': 90},
}


@pytest.mark.parametrize(
    'width, height, scale_extent, expected_width, expected_height, description',
    [
        (100, 50, test_scale_extent, 100, 50, 'Grid parameters are provided.'),
        (None, None, None, 6, 10, 'Grid parameters are excluded from message.'),
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
    logger = logging.getLogger()

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
        result_filename = regrid(message, input_filename, source, logger)

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
    logger = logging.getLogger()

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
        result_filename = regrid(message, input_filename, source, logger)

    assert result_filename == output_filename
    assert Path(output_filename).exists()

    expected_groups = [
        '/Freeze_Thaw_Retrieval_Data_Polar',
        '/Freeze_Thaw_Retrieval_Data_Global',
    ]
    expected = {
        '/Freeze_Thaw_Retrieval_Data_Polar': {
            'width': 265,
            'height': 124,
        },
        '/Freeze_Thaw_Retrieval_Data_Global': {
            'width': 187,
            'height': 78,
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


@pytest.mark.parametrize(
    'use_spl3ftp, expected_record_count, expected_derived_from',
    [
        (
            True,
            2,
            'https://opendap.uat.earthdata.nasa.gov/collections/C1268617120-EEDTEST/'
            'granules/SC:SPL3FTP.004:296000525.dap.nc4',
        ),
        (False, 1, 'https://archive.example.com/SMAP_L4_SM_aup.h5'),
    ],
    ids=['input_with_provenance', 'input_without_provenance'],
)
def test_regrid_writes_provenance(
    use_spl3ftp,
    expected_record_count,
    expected_derived_from,
    test_spl3ftp_ncfile,
    smap_projected_netcdf_file,
    tmp_path,
):
    """Regridded output carries history and history_json provenance.

    With SPL3FTP input, the upstream OPeNDAP and Metadata Annotator provenance
    is preserved and the OPeNDAP request URL is the `derived_from` value.
    With input that has no provenance, the supplied source URL is recorded without
    its query string.

    """
    input_filename = str(
        test_spl3ftp_ncfile if use_spl3ftp else smap_projected_netcdf_file
    )
    short_name = 'SPL3FTP' if use_spl3ftp else 'SPL4SMAU'
    output_filename = str(tmp_path / 'regridded_output.nc')

    message = HarmonyMessage(
        {
            'format': {'mime': 'application/x-netcdf', 'crs': 'EPSG:4326'},
            'sources': [{'collection': 'C123-TEST', 'shortName': short_name}],
        }
    )
    source = HarmonySource({'collection': 'C123-TEST', 'shortName': short_name})

    # Mock generate_output_filename to control the output path
    with patch(
        'harmony_regridding_service.regridding_service.generate_output_filename',
        return_value=output_filename,
    ):
        regrid(
            message,
            input_filename,
            source,
            logging.getLogger(),
            source_url='https://archive.example.com/SMAP_L4_SM_aup.h5?token=abc',
        )

    with xr.open_datatree(output_filename) as dt:
        history = dt.attrs['history']
        history_json = json.loads(dt.attrs['history_json'])

    assert len(history_json) == expected_record_count
    assert history_json[-1]['program'] == PROGRAM
    assert history_json[-1]['version'] == get_semantic_version()
    assert history_json[-1]['parameters'] == {'crs': 'EPSG:4326'}
    assert history_json[-1]['derived_from'] == expected_derived_from
    assert history.split('\n')[-1] == (
        f'{history_json[-1]["date_time"]} {PROGRAM} {get_semantic_version()} '
        '{"crs": "EPSG:4326"}'
    )

    if use_spl3ftp:
        assert history_json[0]['program'] == 'hyrax'
        assert 'Harmony Metadata Annotator 1.0.1' in history


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
    logger = logging.getLogger()

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
        result_filename = regrid(message, input_filename, source, logger)

    assert result_filename == output_filename
    assert Path(output_filename).exists()

    expected_groups = [
        '/Freeze_Thaw_Retrieval_Data_Polar',
        '/Freeze_Thaw_Retrieval_Data_Global',
    ]
    expected = {
        '/Freeze_Thaw_Retrieval_Data_Polar': {
            'width': 265,
            'height': 124,
        },
        '/Freeze_Thaw_Retrieval_Data_Global': {
            'width': 187,
            'height': 78,
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
    logger = MagicMock()

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
        expected_message = re.escape(
            r'Request for unprocessable variable(s): '
            "{'/Freeze_Thaw_Retrieval_Data_Global/altitude_dem'}."
        )
        with pytest.raises(InvalidVariableRequest, match=expected_message):
            regrid(message, input_filename, source, logger)
