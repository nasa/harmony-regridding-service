"""logging context.

Logging context from harmony is on a per request basis and adds a user and requestId.

"""

from logging import getLogger

_LOGGER = None


def set_logger(logger):
    """Set the logger for this request."""
    global _logger
    _LOGGER = logger


def get_logger(default_name='harmony-service.regridder'):
    """Get the context logger or fall back to module logger."""
    return _LOGGER if _LOGGER else getLogger(default_name)
