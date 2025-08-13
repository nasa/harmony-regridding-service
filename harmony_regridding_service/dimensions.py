"""Module for handling dimension variables."""

from collections.abc import Iterable
from typing import NamedTuple

from varinfo import VarInfoFromNetCDF4


class GridDimensionPair(NamedTuple):
    """Horizontal grid dimensions."""

    dim1: str
    dim2: str


def horizontal_dims_for_variable(
    var_info: VarInfoFromNetCDF4, var_name: str
) -> GridDimensionPair | None:
    """Return the horizontal dimensions for desired variable."""
    group_vars = var_info.group_variables_by_horizontal_dimensions()
    dim_pair = next(
        (dims for dims, var_names in group_vars.items() if var_name in var_names),
        None,
    )
    return GridDimensionPair(*dim_pair) if dim_pair else None


def get_row_dims(dims: Iterable[str], var_info: VarInfoFromNetCDF4) -> list[str]:
    """Return name for vertical grid dimension [row/latitude/y].

    This is the up/down dimension for a normal grid.
    """
    return [dim for dim in dims if is_row_dim(dim, var_info)]


def get_column_dims(dims: Iterable[str], var_info: VarInfoFromNetCDF4) -> list[str]:
    """Return name for grid dimension [column/longitude/x].

    This is the right/left dimension for a normal grid.

    """
    return [dim for dim in dims if is_column_dim(dim, var_info)]


def is_column_dim(dim: str, var_info: VarInfoFromNetCDF4) -> str:
    """Test if dim is a column dimension."""
    try:
        dim_var = var_info.get_variable(dim)
        is_x_dim = dim_var.is_longitude() or dim_var.is_projection_x()
    except AttributeError:
        is_x_dim = False
    return is_x_dim


def is_row_dim(dim: str, var_info: VarInfoFromNetCDF4) -> str:
    """Test if dim is a row dimension."""
    is_y_dim = False
    try:
        dim_var = var_info.get_variable(dim)
        is_y_dim = dim_var.is_latitude() or dim_var.is_projection_y()
    except AttributeError:
        pass
    return is_y_dim


def get_resampled_dimension_pairs(
    var_info: VarInfoFromNetCDF4,
) -> list[GridDimensionPair]:
    """Return a list of the resampled horizontal spatial dimensions.

    Gives a list of the 2-element horizontal dimensions that are used in
    regridding this granule file.
    """
    return [
        GridDimensionPair(*dims)
        for dims in var_info.group_variables_by_horizontal_dimensions()
        if len(dims) == 2
    ]
