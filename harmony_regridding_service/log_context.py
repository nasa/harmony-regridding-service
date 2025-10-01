"""logging context.

Logging context from harmony is on a per request basis and adds a user and requestId.

"""

from logging import getLogger

_logger = None


def set_logger(logger):
    """Set the logger for this request."""
    global _logger
    _logger = logger


def get_logger(default_name='harmony-service.regridder'):
    """Get the context logger or fall back to module logger."""
    return _logger if _logger else getLogger(default_name)
