"""Tests the message_utilities module."""

import pytest
from harmony_service_lib.message import Message as HarmonyMessage

from harmony_regridding_service.exceptions import InvalidTargetCRS
from harmony_regridding_service.message_utilities import (
    has_valid_crs,
    is_geographic_crs,
)
from harmony_regridding_service.regridding_cli import (
    get_harmony_message_from_params,
)


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


@pytest.mark.parametrize(
    'message, expected, description',
    [
        (HarmonyMessage({}), True, 'CRS = None is valid'),
        (HarmonyMessage({'format': {}}), True, 'format.crs = None is valid'),
        (
            HarmonyMessage({'format': {'crs': 'EPSG:4326'}}),
            True,
            'format.crs = "EPSG:4326" is valid',
        ),
        (
            HarmonyMessage({'format': {'crs': '+proj=longlat'}}),
            True,
            'format.crs = "+proj=longlat" is valid',
        ),
        (
            HarmonyMessage({'format': {'crs': '4326'}}),
            True,
            'format.crs = "4326" is valid',
        ),
        (
            HarmonyMessage({'format': {'crs': 'EPSG:6933'}}),
            False,
            'Non-geographic EPSG code is invalid',
        ),
        (
            HarmonyMessage({'format': {'crs': '+proj=cea'}}),
            False,
            'Non-geographic proj4 string is invalid',
        ),
    ],
)
def test_has_valid_crs(message, expected, description):
    """Test has_valid_crs.

    Ensure the function correctly determines if the input Harmony
    message has a target Coordinate Reference System (CRS) that is
    compatible with the service. Currently this is either to not
    define the target CRS (assuming it to be geographic), or explicitly
    requesting geographic CRS via EPSG code or proj4 string.

    """
    assert has_valid_crs(message) == expected, f'Failed for {description}'


def test_has_valid_crs_raises_exception():
    """Test has_valid_crs when an exception is thrown."""
    crs_string = 'invalid CRS'
    message = HarmonyMessage({'format': {'crs': crs_string}})
    with pytest.raises(InvalidTargetCRS, match=crs_string):
        has_valid_crs(message)


@pytest.mark.parametrize(
    'message, expected, description',
    [
        ('EPSG:4326', True, 'EPSG:4326 is geographic'),
        ('+proj=longlat', True, '+proj=longlat is geographic'),
        ('4326', True, '4326 is geographic'),
        ('EPSG:6933', False, 'EPSG:6933 is non-geographic'),
        ('+proj=cea', False, '+proj=cea is non-geographic'),
    ],
)
def test_is_geographic_crs(message, expected, description):
    """Test is_geographic_crs.

    Ensure function correctly determines if a supplied string resolves
    to a `pyproj.CRS` object with a geographic Coordinate Reference
    System (CRS). Exceptions arising from invalid CRS strings should
    also be handled.

    """
    assert is_geographic_crs(message) == expected, f'Failed for {description}'


def test_is_geographic_raises_exception():
    """Test is_geographic_crs when it throws an exception."""
    crs_string = 'invalid CRS'
    with pytest.raises(InvalidTargetCRS, match=crs_string):
        is_geographic_crs(crs_string)
