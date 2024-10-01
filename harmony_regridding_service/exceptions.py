""" The exceptions in this module are designed to supply understandable
    messages, which will be propagated out to the end-user via the main Harmony
    application.

"""

from harmony_service_lib.util import HarmonyException


class RegridderException(HarmonyException):
    """Base service exception."""

    def __init__(self, message=None):
        super().__init__(message, 'harmony-regridding-service')


class InvalidTargetCRS(RegridderException):
    """Raised when a request specifies an unsupported target Coordinate
    Reference System.

    """

    def __init__(self, target_crs: str):
        super().__init__(f'Target CRS not supported: "{target_crs}"')


class InvalidInterpolationMethod(RegridderException):
    """Raised when a user specifies an unsupported interpolation method."""

    def __init__(self, interpolation_method: str):
        super().__init__(
            'Interpolation method not supported: ' f'"{interpolation_method}"'
        )


class InvalidTargetGrid(RegridderException):
    """Raised when a request specifies an incomplete or invalid grid."""

    def __init__(self):
        super().__init__('Insufficient or invalid target grid parameters.')


class InvalidSourceDimensions(RegridderException):
    """Raised when a source granule does not meet the expected dimension shapes."""

    def __init__(self, message: str):
        super().__init__(message)
