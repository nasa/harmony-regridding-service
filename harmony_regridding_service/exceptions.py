"""Custom exceptions module.

The exceptions in this module are designed to supply understandable
messages, which will be propagated out to the end-user via the main Harmony
application.
"""

from harmony_service_lib.exceptions import NoRetryException


class RegridderNoRetryException(NoRetryException):
    """Regridding service exception.

    This exception is inhertited by errors that should not be retried by
    Harmony Service.

    """

    def __init__(self, message=None):
        super().__init__(message)


class SourceDataError(RegridderNoRetryException):
    """Incorrect or missing information in the source data."""

    def __init__(self, message: str):
        super().__init__(f'Source Data Error: "{message}"')


class InvalidSourceCRS(RegridderNoRetryException):
    """An Unsupported or incomplete Source Coordinate Reference System."""

    def __init__(self, message: str):
        super().__init__(f'Source CRS not supported: "{message}"')


class InvalidTargetCRS(RegridderNoRetryException):
    """An unsupported target Coordinate Reference System."""

    def __init__(self, target_crs: str):
        super().__init__(f'Target CRS not supported: "{target_crs}"')


class InvalidInterpolationMethod(RegridderNoRetryException):
    """Raised when a user specifies an unsupported interpolation method."""

    def __init__(self, interpolation_method: str):
        super().__init__(
            f'Interpolation method not supported: "{interpolation_method}"'
        )


class InvalidTargetGrid(RegridderNoRetryException):
    """Raised when a request specifies an incomplete or invalid grid."""

    def __init__(self):
        super().__init__('Insufficient or invalid target grid parameters.')


class InvalidSourceDimensions(RegridderNoRetryException):
    """Raised when a source granule does not meet the expected dimension shapes."""

    def __init__(self, message: str):
        super().__init__(message)


class InvalidCRSResampling(RegridderNoRetryException):
    """Raised when target and source CRS match and message has no grid parameters."""

    def __init__(self, message: str):
        super().__init__(message)


class InvalidVariableRequest(RegridderNoRetryException):
    """Raised when a user requests an unprocessable variable."""

    def __init__(self, bad_vars: set[str]):
        message = f'Request for unprocessable variable(s): {bad_vars}.'
        super().__init__(message)
