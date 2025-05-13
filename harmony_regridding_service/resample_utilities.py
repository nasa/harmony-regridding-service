"""Module for utility functions for resampling projection data."""

from logging import getLogger

import numpy as np
from varinfo import VarInfoFromNetCDF4

from harmony_regridding_service.dimensions import (
    _is_column_dim,
    _is_row_dim,
)

logger = getLogger(__name__)


def _get_rows_per_scan(total_rows: int) -> int:
    """Gets optimum value for rows per scan.

    Finds the smallest divisor of the total number of rows. If no divisor is
    found, return the total number of rows.

    """
    if total_rows < 2:
        return 1
    for row_number in range(2, int(total_rows**0.5) + 1):
        if total_rows % row_number == 0:
            logger.info(f'rows_per_scan = {row_number}')
            return row_number

    logger.info(f'returning all rows for rows_per_scan = {total_rows}')
    return total_rows


def _prepare_data_plane(
    data: np.array,
    var_info: VarInfoFromNetCDF4,
    var_name: str,
    cast_to: np.dtype | None,
) -> np.array:
    """Perform Type casting and transpose 2d data array when necessary.

    Also perform a transposition if the data dimension organization requires.
    """
    if cast_to is not None and data.dtype != cast_to:
        data = data.astype(cast_to)

    if _needs_rotation(var_info, var_name):
        data = np.ma.copy(data.T, order='C')

    return data


def _needs_rotation(var_info: VarInfoFromNetCDF4, variable: str) -> bool:
    """Check if variable must be rotated before resampling.

    pyresample's EWA assumes swath input which implies the x projection
    dimension must be the fastest varying dimension.

    So if the lon comes before lat in the variables dimensions you must rotate
    the grid before resampling.

    """
    needs_rotation = False
    var_dims = var_info.get_variable(variable).dimensions
    xloc = next(
        (
            index
            for index, dimension in enumerate(var_dims)
            if _is_column_dim(dimension, var_info)
        ),
        None,
    )
    yloc = next(
        (
            index
            for index, dimension in enumerate(var_dims)
            if _is_row_dim(dimension, var_info)
        ),
        None,
    )
    if yloc > xloc:
        needs_rotation = True

    return needs_rotation


def _integer_like(test_type: np.dtype) -> bool:
    """Return True if the datatype is integer like."""
    return np.issubdtype(np.dtype(test_type), np.integer)
