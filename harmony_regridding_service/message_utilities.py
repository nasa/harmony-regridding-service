"""Module containing Harmony Message utilities."""

from harmony_service_lib.message import Message
from harmony_service_lib.message_utility import rgetattr

from harmony_regridding_service.crs import _is_geographic_crs

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
    target_crs = rgetattr(message, 'format.crs')
    return target_crs is None or _is_geographic_crs(target_crs)
