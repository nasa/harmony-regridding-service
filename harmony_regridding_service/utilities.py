"""Utility functions for use within the Harmony Regridding Service.

This module contains lower-level functionality that can be abstracted into
functions. Primarily this improves readability of the source code, and
allows finer-grained unit testing of each smaller part of functionality.

"""

from logging import getLogger
from mimetypes import guess_type as guess_mime_type
from os.path import splitext
from pathlib import PurePath

from netCDF4 import (
    Dataset,
    Group,
    Variable,
)
from varinfo import VarInfoFromNetCDF4

from harmony_regridding_service.exceptions import SourceDataError

logger = getLogger(__name__)

KNOWN_MIME_TYPES = {
    '.nc4': 'application/x-netcdf4',
    '.h5': 'application/x-hdf5',
    '.hdf5': 'application/x-hdf5',
}


def get_file_mime_type(file_name: str) -> str | None:
    """Infer file's MIME type.

    This function tries to infer the MIME type of a file string. If the
    `mimetypes.guess_type` function cannot guess the MIME type of the
    granule, a dictionary of known file types is checked using the file
    extension. That dictionary only contains keys for MIME types that
    `mimetypes.guess_type` cannot resolve.

    """
    mime_type = guess_mime_type(file_name, False)

    if not mime_type or mime_type[0] is None:
        mime_type = (KNOWN_MIME_TYPES.get(splitext(file_name)[1].lower()), None)

    return mime_type[0]


def walk_groups(node: Dataset | Group) -> Group:
    """Traverse a netcdf file yielding each group."""
    yield node.groups.values()
    for value in node.groups.values():
        yield from walk_groups(value)


def transfer_metadata(source_ds: Dataset, target_ds: Dataset) -> None:
    """Transfer over global and group metadata to target file."""
    global_metadata = {}
    for attr in source_ds.ncattrs():
        global_metadata[attr] = source_ds.getncattr(attr)

    target_ds.setncatts(global_metadata)

    for groups in walk_groups(source_ds):
        for group in groups:
            group_metadata = {}
            for attr in group.ncattrs():
                group_metadata[attr] = group.getncattr(attr)
            t_group = target_ds.createGroup(group.path)
            t_group.setncatts(group_metadata)


def copy_var_with_attrs(
    source_ds: Dataset, target_ds: Dataset, variable_name: str
) -> tuple[Variable, Variable]:
    """Copy a source variable and metadata to target.

    Copy both the variable and metadata from a souce variable into a target,
    return both source and target variables.
    """
    s_var, t_var = copy_var_without_metadata(source_ds, target_ds, variable_name)

    for att in s_var.ncattrs():
        if att != '_FillValue':
            t_var.setncattr(att, s_var.getncattr(att))

    return (s_var, t_var)


def copy_var_without_metadata(
    source_ds: Dataset, target_ds: Dataset, variable_name: str
) -> tuple[Variable, Variable]:
    """Clones a single variable and returns both source and target variables.

    This function uses the netCDF4 createGroup('/[optionalgroup/andsubgroup]')
    call This will return an existing group, or create one that does not
    already exists. So this is not clobbering the source data.

    """
    var = PurePath(variable_name)
    s_var = get_variable(source_ds, variable_name)

    # Create target variable
    t_group = target_ds.createGroup(var.parent)
    fill_value = getattr(s_var, '_FillValue', None)
    t_var = t_group.createVariable(
        var.name, s_var.dtype, s_var.dimensions, fill_value=fill_value
    )
    s_var.set_auto_maskandscale(False)
    t_var.set_auto_maskandscale(False)

    return (s_var, t_var)


def clone_variables(
    source_ds: Dataset, target_ds: Dataset, variables: set[str]
) -> set[str]:
    """Clone variables from source to target.

    Copy variables and their attributes directly from the source Dataset to the
    target Dataset.
    """
    for variable_name in variables:
        (s_var, t_var) = copy_var_with_attrs(source_ds, target_ds, variable_name)
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


def get_variable(dataset: Dataset, variable_name: str) -> Variable:
    """Return a variable from a fully qualified variable name.

    This will return an existing or create a new variable.
    """
    var = PurePath(variable_name)
    group = dataset.createGroup(var.parent)
    return group[var.name]


def get_bounds_var(var_info: VarInfoFromNetCDF4, dim_name: str) -> str:
    """Return the bounds variable associated with the given dimension."""
    return next(
        (
            var_info.get_variable(f'{dim_name}_{ext}').name
            for ext in ['bnds', 'bounds']
            if var_info.get_variable(f'{dim_name}_{ext}') is not None
        ),
        None,
    )
