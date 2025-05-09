"""Tests for regridding_utility.py module."""

from unittest.mock import MagicMock, patch

import pytest
from harmony_service_lib.message import Message as HarmonyMessage
from harmony_service_lib.message import Source as HarmonySource

from harmony_regridding_service.regridding_utility import (
    get_harmony_message_from_params,
    regrid_cli_entry,
)


@pytest.fixture
def message_params():
    """Fixture for creating Harmony Messages."""
    params = {
        'mime': 'application/x-netcdf',
        'crs': {'epsg': 'EPSG:4326'},
        'srs': {
            'epsg': 'EPSG:4326',
            'proj4': '+proj=longlat +datum=WGS84 +no_defs',
            'wkt': 'GEOGCS["WGS 84",DATUM...',
        },
        'scale_extent': {
            'x': {'min': -180, 'max': 180},
            'y': {'min': -90, 'max': 90},
        },
        'scale_size': {'x': 10, 'y': 9},
        'height': 100,
        'width': 99,
    }
    return params


@patch('harmony_regridding_service.regridding_utility.regrid')
def test_regrid_cli_entry(mock_regrid, message_params):
    """Test the regrid_cli_entry function."""
    source_filename = 'source_filename.nc'
    params = message_params
    source = {'collection': 'collection shortname'}
    call_logger = MagicMock()

    regrid_cli_entry(source_filename, params, source, call_logger)

    # Assert that regrid was called once
    mock_regrid.assert_called_once()

    # Get the arguments that regrid was called with
    args, kwargs = mock_regrid.call_args

    # Assert the arguments
    harmony_message_arg, input_filename_arg, source_arg, logger_arg = args

    assert isinstance(harmony_message_arg, HarmonyMessage)
    assert isinstance(source_arg, HarmonySource)
    assert input_filename_arg == source_filename
    assert logger_arg == call_logger


def test_get_harmony_message_all_params(message_params):
    """Test with all parameters supplied."""
    params = message_params

    message = get_harmony_message_from_params(params)
    assert isinstance(message, HarmonyMessage)
    assert message.format.mime == 'application/x-netcdf'
    assert message.format.crs == {'epsg': 'EPSG:4326'}
    assert message.format.srs.epsg == 'EPSG:4326'
    assert message.format.srs.wkt == 'GEOGCS["WGS 84",DATUM...'
    assert message.format.srs.proj4 == '+proj=longlat +datum=WGS84 +no_defs'
    assert message.format.height == 100
    assert message.format.width == 99
    assert message.format.scaleExtent.x.min == -180
    assert message.format.scaleExtent.x.max == 180
    assert message.format.scaleExtent.y.min == -90
    assert message.format.scaleExtent.y.max == 90
    assert message.format.scaleSize.x == 10
    assert message.format.scaleSize.y == 9


def test_get_harmony_message_empty_params():
    """Test with empty parameters dictionary."""
    params = {}

    message = get_harmony_message_from_params(params)

    assert isinstance(message, HarmonyMessage)
    assert message.format.mime == 'application/netcdf'
    assert message.format.crs is None
    assert message.format.srs is None
    assert message.format.scaleExtent is None
    assert message.format.scaleSize is None
    assert message.format.height is None
    assert message.format.width is None


def test_get_harmony_message_no_params():
    """Test with no parameters supplied."""
    message = get_harmony_message_from_params(None)

    assert isinstance(message, HarmonyMessage)
    assert message.format.mime == 'application/netcdf'
    assert message.format.crs is None
    assert message.format.srs is None
    assert message.format.scaleExtent is None
    assert message.format.scaleSize is None
    assert message.format.height is None
    assert message.format.width is None
