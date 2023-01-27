from harmony.util import HarmonyException


class RegridderException(HarmonyException):
    """ Base service exception. """
    def __init__(self, message=None):
        super().__init__(message, 'sds/harmony-regridder')
