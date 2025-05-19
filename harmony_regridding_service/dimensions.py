"""Module for handling dimension variables."""

from collections.abc import Iterable
from pathlib import PurePath

from netCDF4 import Dataset, Dimension
from pyresample.geometry import AreaDefinition
from varinfo import VarInfoFromNetCDF4

from harmony_regridding_service.exceptions import RegridderException
from harmony_regridding_service.utilities import (
    copy_var_without_metadata,
    get_bounds_var,
)


def horizontal_dims_for_variable(
    var_info: VarInfoFromNetCDF4, var_name: str
) -> tuple[str, str]:
    """Return the horizontal dimensions for desired variable."""
    group_vars = var_info.group_variables_by_horizontal_dimensions()
    return next(
        (dims for dims, var_names in group_vars.items() if var_name in var_names),
        None,
    )


def get_row_dims(dims: Iterable[str], var_info: VarInfoFromNetCDF4) -> str:
    """Return name for vertical grid dimension [row/latitude/y].

    This is the up/down dimension for a normal grid.
    """
    return [dim for dim in dims if is_row_dim(dim, var_info)]


def get_column_dims(dims: Iterable[str], var_info: VarInfoFromNetCDF4) -> list[str]:
    """Return name for grid dimension [column/longitude/x].

    This is the right/left dimension for a normal grid.

    """
    return [dim for dim in dims if is_column_dim(dim, var_info)]


def create_dimension(dataset: Dataset, dimension_name: str, size: int) -> Dimension:
    """Create a fully qualified dimension on the dataset."""
    dim = PurePath(dimension_name)
    group = dataset.createGroup(dim.parent)
    return group.createDimension(dim.name, size)


def is_column_dim(dim: str, var_info: VarInfoFromNetCDF4) -> str:
    """Test if dim is a horizontal dimension."""
    try:
        dim_var = var_info.get_variable(dim)
        is_x_dim = dim_var.is_longitude() or dim_var.is_projection_x()
    except AttributeError:
        is_x_dim = False
    return is_x_dim


def is_row_dim(dim: str, var_info: VarInfoFromNetCDF4) -> str:
    """Test if dim is a projection Y dimension."""
    is_y_dim = False
    try:
        dim_var = var_info.get_variable(dim)
        is_y_dim = dim_var.is_latitude() or dim_var.is_projection_y()
    except AttributeError:
        pass
    return is_y_dim


def get_all_dimensions(var_info: VarInfoFromNetCDF4) -> set[str]:
    """Return a list of all dimensions in the file."""
    dimensions = set()
    for variable_name in var_info.get_all_variables():
        variable = var_info.get_variable(variable_name)
        for dim in variable.dimensions:
            dimensions.add(dim)

    return dimensions


def copy_dimensions(
    dimensions: set[str], source_ds: Dataset, target_ds: Dataset
) -> set[str]:
    """Copy each dimension from source to target.

    ensure the first dimensions copied are the UNLIMITED dimensions.
    """

    def sort_unlimited_first(dimension_name):
        """Sort dimensions so that unlimited are first in list."""
        the_dim = get_dimension(source_ds, dimension_name)
        return not the_dim.isunlimited()

    sorted_dims = sorted(list(dimensions), key=sort_unlimited_first)

    for dim in sorted_dims:
        copy_dimension(dim, source_ds, target_ds)


def copy_dimension(dimension_name: str, source_ds: Dataset, target_ds: Dataset) -> str:
    """Copy dimension from source to target file."""
    source_dimension = get_dimension(source_ds, dimension_name)

    source_size = None
    if not source_dimension.isunlimited():
        source_size = source_dimension.size

    dim = PurePath(dimension_name)
    target_group = target_ds.createGroup(dim.parent)
    return target_group.createDimension(dim.name, source_size)


def get_dimension(dataset: Dataset, dimension_name: str) -> Dimension:
    """Return a dimension object for a dimension name.

    Return the Dimension for an arbitrarily nested dimension name.
    """
    dim = PurePath(dimension_name)
    return dataset.createGroup(dim.parent).dimensions[dim.name]


def copy_1d_dimension_variables(
    source_ds: Dataset,
    target_ds: Dataset,
    dim_var_names: set[str],
    target_area: AreaDefinition,
    var_info: VarInfoFromNetCDF4,
) -> set[str]:
    """Copy 1 dimensional dimension variables.

    These are the variables associated directly with the resampled
    longitudes, latitudes, Columns, rows, x-variables, and y-variables.
    """
    # pylint: disable-msg=too-many-locals
    one_d_vars = {
        dim_var_name
        for dim_var_name in dim_var_names
        if len(var_info.get_variable(dim_var_name).dimensions) == 1
    }

    xdims = get_column_dims(one_d_vars, var_info)
    ydims = get_row_dims(one_d_vars, var_info)

    for dim_name in one_d_vars:
        if dim_name in xdims:
            target_coords = target_area.projection_x_coords
            standard_metadata = {
                'long_name': 'longitude',
                'standard_name': 'longitude',
                'units': 'degrees_east',
            }
        elif dim_name in ydims:
            target_coords = target_area.projection_y_coords
            standard_metadata = {
                'long_name': 'latitude',
                'standard_name': 'latitude',
                'units': 'degrees_north',
            }
        else:
            raise RegridderException(
                f'dim_name: {dim_name} not found in projection dimensions'
            )

        (_, t_var) = copy_var_without_metadata(source_ds, target_ds, dim_name)

        bounds_var = get_bounds_var(var_info, dim_name)

        if bounds_var:
            standard_metadata['bounds'] = bounds_var
        t_var.setncatts(standard_metadata)

        t_var[:] = target_coords

    return one_d_vars
