"""Logging Context.

This module is used to capture the logging context from harmony and allow easy
access to all of the modules in this service.

We are capturing the logger that harmony service lib has created and allowing
all of the modules in the service to access and use without having to pass a
logging object in each function signature.

"""

from logging import getLogger

_LOGGER = None


def set_logger(logger):
    """Set the logger context for this request's session."""
    global _LOGGER
    _LOGGER = logger


def get_logger(default_name='harmony-service.regridder'):
    """Get the context logger or fall back to module logger."""
    return _LOGGER if _LOGGER else getLogger(default_name)
