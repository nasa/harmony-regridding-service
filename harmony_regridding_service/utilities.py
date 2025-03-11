"""Utility modules for use within the Harmony Regridding Service. These
include MIME type determination and basic components of message validation.

"""

from mimetypes import guess_type as guess_mime_type
from os.path import splitext
from typing import Optional

from harmony_service_lib.message import Message
from harmony_service_lib.message_utility import rgetattr
from pyproj import CRS
from pyproj.exceptions import CRSError

from harmony_regridding_service.exceptions import InvalidTargetCRS

KNOWN_MIME_TYPES = {
    '.nc4': 'application/x-netcdf4',
    '.h5': 'application/x-hdf5',
    '.hdf5': 'application/x-hdf5',
}
VALID_INTERPOLATION_METHODS = ('Elliptical Weighted Averaging',)


def get_file_mime_type(file_name: str) -> Optional[str]:
    """This function tries to infer the MIME type of a file string. If the
    `mimetypes.guess_type` function cannot guess the MIME type of the
    granule, a dictionary of known file types is checked using the file
    extension. That dictionary only contains keys for MIME types that
    `mimetypes.guess_type` cannot resolve.

    """
    mime_type = guess_mime_type(file_name, False)

    if not mime_type or mime_type[0] is None:
        mime_type = (KNOWN_MIME_TYPES.get(splitext(file_name)[1].lower()), None)

    return mime_type[0]


def has_valid_crs(message: Message) -> bool:
    """Check the target Coordinate Reference System (CRS) in the Harmony
    message is compatible with the CRS types supported by the regridder. In
    the MVP, only a geographic CRS is supported, so `Message.format.crs`
    should either be undefined or specify a geographic CRS.

    """
    target_crs = rgetattr(message, 'format.crs')
    return target_crs is None or _is_geographic_crs(target_crs)


def _is_geographic_crs(crs_string: str) -> bool:
    """Use pyproj to ascertain if the supplied Coordinate Reference System
    (CRS) is geographic.

    """
    try:
        crs = CRS(crs_string)
        is_geographic = crs.is_geographic
    except CRSError as exception:
        raise InvalidTargetCRS(crs_string) from exception

    return is_geographic


def has_valid_interpolation(message: Message) -> bool:
    """Check the interpolation method in the input Harmony message is
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
