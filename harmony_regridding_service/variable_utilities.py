"""Module for handling NetCDF4 variables."""

from logging import getLogger
from pathlib import PurePath

from netCDF4 import Dataset, Variable
from varinfo import VarInfoFromNetCDF4

from harmony_regridding_service.exceptions import SourceDataError

logger = getLogger(__name__)


def _copy_var_with_attrs(
    source_ds: Dataset, target_ds: Dataset, variable_name: str
) -> (Variable, Variable):
    """Copy a source variable and metadata to target.

    Copy both the variable and metadata from a souce variable into a target,
    return both source and target variables.
    """
    s_var, t_var = _copy_var_without_metadata(source_ds, target_ds, variable_name)

    for att in s_var.ncattrs():
        if att != '_FillValue':
            t_var.setncattr(att, s_var.getncattr(att))

    return (s_var, t_var)


def _copy_var_without_metadata(
    source_ds: Dataset, target_ds: Dataset, variable_name: str
) -> (Variable, Variable):
    """Clones a single variable and returns both source and target variables.

    This function uses the netCDF4 createGroup('/[optionalgroup/andsubgroup]')
    call This will return an existing group, or create one that does not
    already exists. So this is not clobbering the source data.

    """
    var = PurePath(variable_name)
    s_var = _get_variable(source_ds, variable_name)

    # Create target variable
    t_group = target_ds.createGroup(var.parent)
    fill_value = getattr(s_var, '_FillValue', None)
    t_var = t_group.createVariable(
        var.name, s_var.dtype, s_var.dimensions, fill_value=fill_value
    )
    s_var.set_auto_maskandscale(False)
    t_var.set_auto_maskandscale(False)

    return (s_var, t_var)


def _clone_variables(
    source_ds: Dataset, target_ds: Dataset, variables: set[str]
) -> set[str]:
    """Clone variables from source to target.

    Copy variables and their attributes directly from the source Dataset to the
    target Dataset.
    """
    for variable_name in variables:
        (s_var, t_var) = _copy_var_with_attrs(source_ds, target_ds, variable_name)
        try:
            t_var[:] = s_var[:]
        except IndexError as vlen_error:
            # Handle snowflake metadata with vlen string.
            if s_var.dtype == str and s_var.shape == ():
                t_var[0] = s_var[0]
            else:
                logger.error('Unable to clone variable {s_var}')
                raise SourceDataError('Unhandled variable clone') from vlen_error

    return variables


def _get_variable(dataset: Dataset, variable_name: str) -> Variable:
    """Return a variable from a fully qualified variable name.

    This will return an existing or create a new variable.
    """
    var = PurePath(variable_name)
    group = dataset.createGroup(var.parent)
    return group[var.name]


def _get_bounds_var(var_info: VarInfoFromNetCDF4, dim_name: str) -> str:
    return next(
        (
            var_info.get_variable(f'{dim_name}_{ext}').name
            for ext in ['bnds', 'bounds']
            if var_info.get_variable(f'{dim_name}_{ext}') is not None
        ),
        None,
    )
