"""Custom exceptions module.

The exceptions in this module are designed to supply understandable
messages, which will be propagated out to the end-user via the main Harmony
application.
"""

from harmony_service_lib.util import HarmonyException


class RegridderException(HarmonyException):
    """Base service exception."""

    def __init__(self, message=None):
        super().__init__(message, 'harmony-regridding-service')


class SourceDataError(RegridderException):
    """Incorrect or missing information in the source data."""

    def __init__(self, message: str):
        super().__init__(f'Source Data Error: "{message}"')


class InvalidSourceCRS(RegridderException):
    """An Unsupported or incomplete Source Coordinate Reference System."""

    def __init__(self, message: str):
        super().__init__(f'Source CRS not supported: "{message}"')


class InvalidTargetCRS(RegridderException):
    """An unsupported target Coordinate Reference System."""

    def __init__(self, target_crs: str):
        super().__init__(f'Target CRS not supported: "{target_crs}"')


class InvalidInterpolationMethod(RegridderException):
    """Raised when a user specifies an unsupported interpolation method."""

    def __init__(self, interpolation_method: str):
        super().__init__(
            f'Interpolation method not supported: "{interpolation_method}"'
        )


class InvalidTargetGrid(RegridderException):
    """Raised when a request specifies an incomplete or invalid grid."""

    def __init__(self):
        super().__init__('Insufficient or invalid target grid parameters.')


class InvalidSourceDimensions(RegridderException):
    """Raised when a source granule does not meet the expected dimension shapes."""

    def __init__(self, message: str):
        super().__init__(message)


class InvalidCRSResampling(RegridderException):
    """Raised when target and source CRS match and message has no grid parameters."""

    def __init__(self, message: str):
        super().__init__(message)
