"""Tests for the provenance module."""

import json
from datetime import datetime

import pytest
from harmony_service_lib.message import Message as HarmonyMessage
from netCDF4 import Dataset

from harmony_regridding_service.provenance import (
    HISTORY_JSON_SCHEMA,
    PROGRAM,
    PROGRAM_REF,
    get_regridding_parameters,
    get_request_url,
    get_semantic_version,
    read_history_attrs,
    read_history_json_attrs,
    update_history_metadata,
)

SPL3FTP_REQUEST_URL = (
    'https://opendap.uat.earthdata.nasa.gov/collections/C1268617120-EEDTEST/'
    'granules/SC:SPL3FTP.004:296000525.dap.nc4'
)
SOURCE_URL = 'https://archive.example.com/granule.nc4'
SCALE_EXTENT = {'x': {'min': -180, 'max': 180}, 'y': {'min': -90, 'max': 90}}


@pytest.fixture
def regrid_message():
    """Harmony message with grid parameters and properties that are ignored."""
    return HarmonyMessage(
        {
            'format': {
                'mime': 'application/x-netcdf4',
                'crs': 'EPSG:4326',
                'srs': {'epsg': 'EPSG:4326'},
                'scaleExtent': SCALE_EXTENT,
                'height': 180,
                'width': 360,
                'interpolation': None,
            },
        }
    )


def write_global_attributes(file_path, attributes: dict):
    """Write a NetCDF-4 file containing only the given global attributes."""
    with Dataset(file_path, mode='w') as dataset:
        dataset.setncatts(attributes)
    return file_path


@pytest.mark.parametrize(
    'input_fixture, expected_name, expected_value',
    [
        (
            'test_MERRA2_ncfile',
            'History',
            'Original file generated: Sun Jun  1 09:02:04 2014 GMT',
        ),
        ('test_ATL14_ncfile', 'history', '2022-08-15T17:54:15.100678Z'),
        ('test_IMERG_ncfile', 'history', None),
    ],
)
def test_read_history_attrs(request, input_fixture, expected_name, expected_value):
    """The attribute casing and value of the input are returned."""
    with Dataset(request.getfixturevalue(input_fixture), mode='r') as dataset:
        assert read_history_attrs(dataset) == (expected_name, expected_value)


def test_get_request_url_list_parameters(test_spl3ftp_ncfile):
    """The OPeNDAP request URL is read from a hyrax `parameters` list."""
    with Dataset(test_spl3ftp_ncfile, mode='r') as dataset:
        assert get_request_url(dataset) == SPL3FTP_REQUEST_URL


@pytest.mark.parametrize(
    'record, expected_url',
    [
        ({'parameters': {'request_url': f'{SOURCE_URL}?dap4.ce=/var'}}, SOURCE_URL),
        ({'parameters': [{'other': 'value'}]}, None),
    ],
    ids=['dict_parameters', 'no_request_url'],
)
def test_get_request_url_other_forms(tmp_path, record, expected_url):
    """A `parameters` object is also read; a missing URL returns None."""
    file_path = write_global_attributes(
        tmp_path / 'input.nc', {'history_json': json.dumps([record])}
    )

    with Dataset(file_path, mode='r') as dataset:
        assert get_request_url(dataset) == expected_url


@pytest.mark.parametrize(
    'message_content, expected_parameters',
    [
        (
            None,
            {
                'crs': 'EPSG:4326',
                'scaleExtent': SCALE_EXTENT,
                'height': 180,
                'width': 360,
            },
        ),
        ({}, {}),
        ({'format': {'mime': 'application/x-netcdf4'}}, {}),
    ],
    ids=['grid_parameters', 'no_format', 'no_grid_parameters'],
)
def test_get_regridding_parameters(
    regrid_message, message_content, expected_parameters
):
    """Only specified, grid-determining parameters are returned."""
    message = (
        regrid_message if message_content is None else HarmonyMessage(message_content)
    )
    assert get_regridding_parameters(message) == expected_parameters


@pytest.mark.parametrize(
    'input_fixture, expected_name, expected_record_count, expected_derived_from',
    [
        ('test_spl3ftp_ncfile', 'history', 2, SPL3FTP_REQUEST_URL),
        ('test_MERRA2_ncfile', 'History', 1, SOURCE_URL),
        ('test_ATL14_ncfile', 'history', 1, SOURCE_URL),
        ('test_IMERG_ncfile', 'history', 1, SOURCE_URL),
    ],
)
def test_update_history_metadata(
    request,
    tmp_path,
    regrid_message,
    input_fixture,
    expected_name,
    expected_record_count,
    expected_derived_from,
):
    """Existing provenance is preserved and a regridding record appended."""
    output_path = tmp_path / 'output.nc'

    with (
        Dataset(request.getfixturevalue(input_fixture), mode='r') as source_ds,
        Dataset(output_path, mode='w') as target_ds,
    ):
        _, input_history = read_history_attrs(source_ds)
        input_records = read_history_json_attrs(source_ds)
        update_history_metadata(
            source_ds, target_ds, regrid_message, f'{SOURCE_URL}?token=abc'
        )

    with Dataset(output_path, mode='r') as output_ds:
        output_attributes = output_ds.ncattrs()
        output_history = output_ds.getncattr(expected_name)
        output_records = json.loads(output_ds.getncattr('history_json'))

    # Only the history attribute casing used by the input is written.
    assert {'history', 'History'}.intersection(output_attributes) == {expected_name}

    # Existing records are preserved, in order, before the new record.
    assert len(output_records) == expected_record_count
    assert output_records[:-1] == input_records

    new_record = output_records[-1]
    expected_parameters = get_regridding_parameters(regrid_message)
    assert new_record['$schema'] == HISTORY_JSON_SCHEMA
    assert datetime.fromisoformat(new_record['date_time']).utcoffset().seconds == 0
    assert new_record['program'] == PROGRAM
    assert new_record['version'] == get_semantic_version()
    assert new_record['parameters'] == expected_parameters
    assert new_record['derived_from'] == expected_derived_from
    assert new_record['program_ref'] == PROGRAM_REF

    # Existing history text is preserved and the new line is appended.
    new_history_line = ' '.join(
        [
            new_record['date_time'],
            PROGRAM,
            get_semantic_version(),
            json.dumps(expected_parameters),
        ]
    )
    expected_history = '\n'.join(filter(None, [input_history, new_history_line]))
    assert output_history == expected_history
