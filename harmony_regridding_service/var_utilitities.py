"""Variable helper functions.

A collection of functions for getting information about variables, mostly with
varinfo.

"""

from varinfo import VarInfoFromNetCDF4


def get_unprocessable_variables(
    var_info: VarInfoFromNetCDF4, var_list: set[str]
) -> set[str]:
    """Variables that can't be processed by Harmony Regridding Service.

    Currently string variables are unprocessable because pyresample cannot
    handle them. All string variables are excluded.

    Science variables that are excluded explicitly by varInfo are also removed
    """
    string_vars = {var for var in var_list if is_string_variable(var_info, var)}
    excluded_vars = var_info.get_excluded_science_variables()

    return string_vars | excluded_vars


def is_string_variable(var_info: VarInfoFromNetCDF4, var_name: str) -> bool:
    """Returns True if variable is a string type."""
    return var_info.get_variable(var_name).data_type in ['str', 'bytes8']
