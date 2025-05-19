"""Module for handling dimension variables."""

from collections.abc import Iterable

from varinfo import VarInfoFromNetCDF4


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
