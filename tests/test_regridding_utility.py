"""Tests for regridding_utility.py module."""

from harmony_service_lib.message import Message as HarmonyMessage

from harmony_regridding_service.regridding_utility import (
    get_harmony_message_from_params,
)


def test_get_harmony_message_all_params():
    """Test with all parameters supplied."""
    params = {
        'mime': 'application/x-netcdf',
        'crs': {'epsg': 'EPSG:4326'},
        'scale_extent': {
            'x': {'min': -180, 'max': 180},
            'y': {'min': -90, 'max': 90},
        },
        'scale_size': {'x': 10, 'y': 9},
        'height': 100,
        'width': 99,
    }

    message = get_harmony_message_from_params(params)
    assert isinstance(message, HarmonyMessage)
    assert message.format.mime == 'application/x-netcdf'
    assert message.format.crs == {'epsg': 'EPSG:4326'}
    assert message.format.srs.epsg == 'EPSG:4326'
    assert message.format.srs.wkt is None
    assert message.format.srs.proj4 is None
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
