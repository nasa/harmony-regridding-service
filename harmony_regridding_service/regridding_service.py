"""Regridding service code."""
from __future__ import annotations

import numpy as np
from netCDF4 import Dataset
from pyresample.geometry import AreaDefinition, SwathDefinition
from pyresample.ewa import DaskEWAResampler

from harmony.message import Message
from varinfo import VarInfoFromNetCDF4

from typing import Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from harmony_regridding_service.adapter import RegriddingServiceAdapter

from harmony_regridding_service.exceptions import InvalidSourceDimensions
from harmony_regridding_service.utilities import has_dimensions


def regrid(adapter: RegriddingServiceAdapter, input_filepath: str) -> str:
    """Regrid the input data at input_filepath."""
    _cache_resamplers(adapter, input_filepath)
    adapter.logger.info('regrid has cached resamplers')
    return input_filepath


def _cache_resamplers(adapter: RegriddingServiceAdapter,
                      filepath: str) -> None:
    """Precompute the resampling weights.

    Determine the desired ouptut Target Area from the Harmony Message.  Use
    this target area in conjunction with each shared horizontal dimension in
    the inupt source file to create an EWA Resampler and precompute the weights
    to be used in a resample from the shared horizontal dimension to the output
    target area.

    """
    var_info = VarInfoFromNetCDF4(filepath, adapter.logger)
    dimension_variables_mapping = \
        var_info.group_variables_by_horizontal_dimensions()

    target_area = _compute_target_area(adapter.message)

    for dimensions, variable_list in dimension_variables_mapping.items():
        # create source swath definition from 2D grids
        if len(dimensions) != 2:
            continue

        source_swath = _compute_source_swath(dimensions, filepath, var_info)
        adapter.cache['grids'][dimensions] = DaskEWAResampler(
            source_swath, target_area).precompute(rows_per_scan=0)

    adapter.logger.info(
        f'cached resamplers for {adapter.cache["grids"].keys()}')


def _compute_target_area(message: Message) -> AreaDefinition:
    """Parse the harmony message and build a target AreaDefinition."""
    # ScaleExtent is required and validated.
    area_extent = (message.format.scaleExtent.x.min,
                   message.format.scaleExtent.y.min,
                   message.format.scaleExtent.x.max,
                   message.format.scaleExtent.x.min)

    height = _grid_height(message)
    width = _grid_width(message)
    projection = message.format.crs

    return AreaDefinition('target_area_id', 'target area definition', None,
                          projection, width, height, area_extent)


def _grid_height(message: Message) -> int:
    """Compute grid height from Message.

    Compute the height of grid from the scaleExtents and scale_sizes.
    """
    if has_dimensions(message):
        return message.format.height
    return _compute_num_elements(message, 'y')


def _grid_width(message: Message) -> int:
    """Compute grid height from Message.

    Compute the height of grid from the scaleExtents and scale_sizes.
    """
    if has_dimensions(message):
        return message.format.width
    return _compute_num_elements(message, 'x')


def _compute_num_elements(message: Message, dimension_name: str) -> int:
    """Compute the number of gridcells based on scaleExtents and scaleSize."""
    scale_extent = getattr(message.format.scaleExtent, dimension_name)
    scale_size = getattr(message.format.scaleSize, dimension_name)

    num_elements = int(
        np.round((scale_extent.max - scale_extent.min) / scale_size) + 1)
    return num_elements


def _get_column_dim(dims: Tuple[str, str],
                    var_info: VarInfoFromNetCDF4) -> str:
    """Return name for horizontal grid dimension [column/longitude/x]."""
    try:
        for dim in dims:
            if (var_info.get_variable(dim).is_longitude()):
                return dim
    except AttributeError:
        raise InvalidSourceDimensions(
            f'No longitude dimension found in {dims}')


def _get_row_dim(dims: Tuple[str, str], var_info: VarInfoFromNetCDF4) -> str:
    """Return name for vertical grid dimension [row/latitude/y]."""
    try:
        for dim in dims:
            if (var_info.get_variable(dim).is_latitude()):
                return dim
    except AttributeError:
        raise InvalidSourceDimensions(f'No latitude dimension found in {dims}')


def _compute_source_swath(grid_dimensions: Tuple[str, str], filepath: str,
                          var_info: VarInfoFromNetCDF4) -> SwathDefinition:
    """Return a SwathDefinition for the input gridDimensions."""
    longitudes, latitudes = _compute_horizontal_source_grids(
        grid_dimensions, filepath, var_info)

    return SwathDefinition(lons=longitudes, lats=latitudes)


def _compute_horizontal_source_grids(
        grid_dimensions: Tuple[str, str], filepath: str,
        var_info: VarInfoFromNetCDF4) -> Tuple[np.array, np.array]:
    """Return 2D np.arrays of longitude and latitude."""
    row_dim = _get_row_dim(grid_dimensions, var_info)
    column_dim = _get_column_dim(grid_dimensions, var_info)

    with Dataset(filepath, mode='r') as ds:
        row_shape = ds[row_dim].shape
        column_shape = ds[column_dim].shape

        if (len(row_shape) == 1 and len(column_shape) == 1):
            num_rows = row_shape[0]
            num_columns = column_shape[0]
            longitudes = np.broadcast_to(ds[column_dim],
                                         (num_rows, num_columns))
            latitudes = np.broadcast_to(
                np.broadcast_to(ds[row_dim], (1, num_rows)).T,
                (num_rows, num_columns))
            longitudes = np.ascontiguousarray(longitudes)
            latitudes = np.ascontiguousarray(latitudes)
        else:
            # Only handling the case of 1-Dimensional dimensions on MVP
            raise InvalidSourceDimensions(
                'Incorrect source data dimensions. '
                'rows:{row_shape}, columns:{columns_shape}')

    return (longitudes, latitudes)
