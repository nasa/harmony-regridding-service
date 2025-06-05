"""Module containing Harmony Message utilities."""

from harmony_service_lib.message import Message
from harmony_service_lib.message_utility import rgetattr
from pyproj import CRS
from pyproj.exceptions import CRSError

from harmony_regridding_service.exceptions import (
    InvalidTargetCRS,
)

VALID_INTERPOLATION_METHODS = ('Elliptical Weighted Averaging',)


def get_harmony_message_from_params(params: dict | None) -> Message:
    """Constructs a harmony message from the input parms.

    We have to create a harmony message to pass to the regrid function so that
    both the CLI entry and service calls are identical.

    """
    if params is None:
        params = {}
    mime = params.get('mime', 'application/netcdf')
    crs = params.get('crs', None)
    srs = params.get('srs', None)
    scale_extent = params.get('scale_extent', None)
    scale_size = params.get('scale_size', None)
    height = params.get('height', None)
    width = params.get('width', None)

    return Message(
        {
            'format': {
                'mime': mime,
                'crs': crs,
                'srs': srs,
                'scaleExtent': scale_extent,
                'scaleSize': scale_size,
                'height': height,
                'width': width,
            },
        }
    )


def has_valid_interpolation(message: Message) -> bool:
    """Ensure valid interpolation.

    Check the interpolation method in the input Harmony message is
    compatible with the methods supported by the regridder. In the MVP,
    only the EWA algorithm is used, so either the interpolation should be
    unspecified in the message, or it should be the string
    "Elliptical Weighted Averaging".

    """
    interpolation_method = rgetattr(message, 'format.interpolation')

    return (
        interpolation_method is None
        or interpolation_method in VALID_INTERPOLATION_METHODS
    )


def has_valid_crs(message: Message) -> bool:
    """Validate CRS.

    Check the target Coordinate Reference System (CRS) in the Harmony
    message is compatible with the CRS types supported by the regridder. In
    the MVP, only a geographic CRS is supported, so `Message.format.crs`
    should either be undefined or specify a geographic CRS.

    """
    target_crs = get_message_crs(message)
    return target_crs is None or is_geographic_crs(target_crs)


def get_message_crs(message: Message) -> str | None:
    """Return the crs information contained in the harmony message."""
    return rgetattr(message, 'format.crs')


def is_geographic_crs(crs_string: str) -> bool:
    """Infer if CRS is geographic.

    Use pyproj to ascertain if the supplied Coordinate Reference System
    (CRS) is geographic.
    """
    try:
        crs = CRS(crs_string)
        is_geographic = crs.is_geographic
    except CRSError as exception:
        raise InvalidTargetCRS(crs_string) from exception

    return is_geographic


def target_crs_from_message(message: Message) -> CRS:
    """Return the message's CRS or default to one from EPSG::4326."""
    target_crs = get_message_crs(message)
    return CRS(target_crs or 'EPSG:4326')
