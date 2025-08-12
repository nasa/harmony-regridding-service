"""Module that handles source file copy and writing to target output."""

from logging import getLogger
from mimetypes import guess_type as guess_mime_type
from os.path import splitext
from pathlib import PurePath

import numpy as np
from netCDF4 import (
    Dataset,
    Group,
    Variable,
)

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
    source_ds: Dataset,
    target_ds: Dataset,
    variable_name: str,
    override_dimensions: tuple | None = None,
) -> tuple[Variable, Variable]:
    """Copy a source variable and metadata to target.

    Copy both the variable and metadata from a souce variable into a target,
    return both source and target variables.
    """
    s_var, t_var = copy_var_without_metadata(
        source_ds, target_ds, variable_name, override_dimensions=override_dimensions
    )

    for att in s_var.ncattrs():
        if att != '_FillValue':
            t_var.setncattr(att, s_var.getncattr(att))

    return (s_var, t_var)


def copy_var_without_metadata(
    source_ds: Dataset,
    target_ds: Dataset,
    variable_name: str,
    override_dimensions: tuple | None = None,
) -> tuple[Variable, Variable]:
    """Clones a single variable and returns both source and target variables.

    This function uses the netCDF4 createGroup('/[optionalgroup/andsubgroup]')
    call This will return an existing group, or create one that does not
    already exists. So this is not clobbering the source data.

    override_dimensions is an optional input to allow you to reorder the
    target's dimensions. If provided it should be a tuple of the target's
    dimension names in preferred order (horizontal dims last).

    """
    var = PurePath(variable_name)
    s_var = get_or_create_variable_in_dataset(source_ds, variable_name)

    # Create target variable
    t_group = target_ds.createGroup(var.parent)
    fill_value = getattr(s_var, '_FillValue', None)

    compress_opts = {}
    if is_compressible(s_var.dtype):
        compress_opts = {'zlib': True, 'complevel': 6}

    t_var = t_group.createVariable(
        var.name,
        s_var.dtype,
        override_dimensions or s_var.dimensions,
        fill_value=fill_value,
        **compress_opts,
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


def get_or_create_variable_in_dataset(dataset: Dataset, variable_name: str) -> Variable:
    """Return a variable from a fully qualified variable name.

    This will return an existing or create a new variable.
    """
    var = PurePath(variable_name)
    group = dataset.createGroup(var.parent)
    return group[var.name]


def is_compressible(dtype: np.dtype) -> bool:
    """Returns false if the variable has a non-compressible type."""
    return not (np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.object_))


def input_grid_mappings(dataset: Dataset, variables: set[str]) -> set[str]:
    """Collect all grid_mapping attribute values from the given variables.

    Args:
        dataset: The NetCDF4 Dataset to search
        variables: Set of full variable paths to check (e.g., 'group/subgroup/variable')

    Returns:
        Set of the values of any grid_mapping attribute found
    """
    grid_mappings = set()

    for var_path in variables:
        try:
            var = dataset[var_path]
            if hasattr(var, 'grid_mapping'):
                grid_mappings.add(var.grid_mapping)
        except (KeyError, IndexError):
            # Variable, Group doesn't exist, skip it
            continue

    return grid_mappings


def filter_grid_mappings_to_variables(grid_mapping_values: set[str]) -> set[str]:
    """Return the grid mapping variable names from grid_mapping values.

    In CF the  grid_mapping attribute can take two formats:
    https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#grid-mappings-and-projections

    In the first format, it is a single word, which names a grid mapping variable.

    In the second format, it is a blank-separated list of words:


    <gridMappingVariable>: <coordinatesVariable> [<coordinatesVariable>...]
    [<gridMappingVariable>: <coordinatesVariable> [<coordinatesVariable>...]..]

    Which identifies one or more grid mapping variables, and with each grid
    mapping associates one or more coordinatesVariables, i.e. coordinate
    variables or auxiliary coordinate variables.

    This function will return a list of the grid mapping variable names,
    dropping any  coordinate variables.
    """
    grid_mapping_variables = set()

    for grid_mapping_value in grid_mapping_values:
        if ':' in grid_mapping_value:
            # find variable names in the second form
            # "var: coord1 coord2 var2: coord3 coord4"
            grid_mapping_variables.update(
                var_part[:-1]
                for var_part in grid_mapping_value.split()
                if var_part.endswith(':')
            )
        else:
            grid_mapping_variables.add(grid_mapping_value)

    return grid_mapping_variables
