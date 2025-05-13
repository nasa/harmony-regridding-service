"""Module for accessing and creating grid paramerers."""

from logging import getLogger

import numpy as np
import xarray as xr
from harmony_service_lib.message import Message as HarmonyMessage
from harmony_service_lib.message_utility import has_dimensions
from netCDF4 import Dataset
from pyproj import CRS
from pyproj.exceptions import CRSError
from pyresample import create_area_def
from pyresample.geometry import AreaDefinition, SwathDefinition
from varinfo import VarInfoFromNetCDF4
from xarray import DataTree

from harmony_regridding_service.dimensions import (
    _dims_are_lon_lat,
    _dims_are_projected_x_y,
    _get_column_dims,
    _get_row_dims,
)
from harmony_regridding_service.exceptions import (
    InvalidSourceCRS,
    InvalidSourceDimensions,
    SourceDataError,
)

logger = getLogger(__name__)


def _compute_target_area(message: HarmonyMessage) -> AreaDefinition:
    """Define the output area for your regridding operation.

    Parse the harmony message and build a target AreaDefinition.  All
    multi-dimensional variables will be regridded to this target.

    """
    # ScaleExtent is required and validated.
    logger.info('compute target_area')
    area_extent = (
        message.format.scaleExtent.x.min,
        message.format.scaleExtent.y.min,
        message.format.scaleExtent.x.max,
        message.format.scaleExtent.y.max,
    )

    height = _grid_height(message)
    width = _grid_width(message)
    projection = message.format.crs or 'EPSG:4326'

    return AreaDefinition(
        'target_area_id',
        'target area definition',
        None,
        projection,
        width,
        height,
        area_extent,
    )


def _grid_height(message: HarmonyMessage) -> int:
    """Compute grid height from Message.

    Compute the height of grid from the scaleExtents and scale_sizes.
    """
    if has_dimensions(message):
        return message.format.height
    return _compute_num_elements(message, 'y')


def _grid_width(message: HarmonyMessage) -> int:
    """Compute grid height from Message.

    Compute the height of grid from the scaleExtents and scale_sizes.
    """
    if has_dimensions(message):
        return message.format.width
    return _compute_num_elements(message, 'x')


def _compute_num_elements(message: HarmonyMessage, dimension_name: str) -> int:
    """Compute the number of gridcells based on scaleExtents and scaleSize."""
    scale_extent = getattr(message.format.scaleExtent, dimension_name)
    scale_size = getattr(message.format.scaleSize, dimension_name)

    num_elements = int(np.round((scale_extent.max - scale_extent.min) / scale_size))
    return num_elements


def _compute_source_swath(
    grid_dimensions: tuple[str, str],
    filepath: str,
    var_info: VarInfoFromNetCDF4,
    variable_set: set,
) -> SwathDefinition:
    """Return a SwathDefinition for the input grid_dimensions."""
    if _dims_are_lon_lat(grid_dimensions, var_info):
        longitudes, latitudes = _compute_horizontal_source_grids(
            grid_dimensions, filepath, var_info
        )
    elif _dims_are_projected_x_y(grid_dimensions, var_info):
        longitudes, latitudes = _compute_projected_horizontal_source_grids(
            grid_dimensions, filepath, var_info, variable_set
        )
    else:
        raise SourceDataError(
            'Cannot determine correct dimension type from source {grid_dimensions}.'
        )

    return SwathDefinition(lons=longitudes, lats=latitudes)


def _compute_horizontal_source_grids(
    grid_dimensions: tuple[str, str], filepath: str, var_info: VarInfoFromNetCDF4
) -> tuple[np.ndarray, np.ndarray]:
    """Return grids for longitude and latitude for the grid_dimension pair.

    Given the input grid_dimension names, create longitude and latitude grids
    underlay the source data for dimensions.

    Each input 2D source data variable, described by the grid_dimensions tuple
    is used to find a 1D longitude array (columns) and a 1D latitude array
    (rows).  These 1D arrays are broadcast to 2 dimensions.

    We return the new longitude[column x row] and latitude[column x row] arrays.

    """
    row_dim = _get_row_dims(grid_dimensions, var_info)[0]
    column_dim = _get_column_dims(grid_dimensions, var_info)[0]
    logger.info(f'found row_dim: {row_dim}')
    logger.info(f'found column_dim: {column_dim}')

    with Dataset(filepath, mode='r') as data_set:
        row_shape = data_set[row_dim].shape
        column_shape = data_set[column_dim].shape
        if len(row_shape) == 1 and len(column_shape) == 1:
            num_rows = row_shape[0]
            num_columns = column_shape[0]
            longitudes = np.broadcast_to(data_set[column_dim], (num_rows, num_columns))
            latitudes = np.broadcast_to(
                np.broadcast_to(data_set[row_dim], (1, num_rows)).T,
                (num_rows, num_columns),
            )
            longitudes = np.ascontiguousarray(longitudes)
            latitudes = np.ascontiguousarray(latitudes)
        else:
            # Only handling the case of 1-Dimensional dimensions on MVP
            raise InvalidSourceDimensions(
                f'Incorrect source data dimensions. '
                f'rows:{row_shape}, columns:{column_shape}'
            )

    return (longitudes, latitudes)


def _compute_projected_horizontal_source_grids(
    grid_dimensions: tuple[str, str],
    filepath: str,
    var_info: VarInfoFromNetCDF4,
    variables: set,
) -> tuple[np.ndarray, np.ndarray]:
    """Return longitude and latitude grids for a projected grid_dimensions pair.

    Given the input grid_dimensions pair, find the projected coordinate dimensions
    in the source data and use those to generate 2D longitude and latitude arrays.

    """
    xdim_name = _get_column_dims(grid_dimensions, var_info)[0]
    ydim_name = _get_row_dims(grid_dimensions, var_info)[0]
    try:
        with xr.open_datatree(filepath) as dt:
            xvalues = dt[xdim_name].data
            yvalues = dt[ydim_name].data
            area_extent = _compute_area_extent_from_regular_x_y_coords(xvalues, yvalues)
            source_crs = _crs_from_source_data(dt, variables)
            cell_width = np.abs(xvalues[1] - xvalues[0])
            cell_height = np.abs(yvalues[1] - yvalues[0])
            source_area = create_area_def(
                'source grid area',
                source_crs,
                area_extent=area_extent,
                shape=(len(yvalues), len(xvalues)),
                resolution=(cell_width, cell_height),
            )
            return source_area.get_lonlats()
    except Exception as e:
        logger.error(e)
        raise SourceDataError('cannot compute projected source grids') from e


def _crs_from_source_data(dt: DataTree, variables: set) -> CRS:
    """Create a CRS describing the grid in the source file.

    Look through the variables for metadata that points to a grid_mapping
    and generate a CRS from that information.

    The metadata is not always clear or easy to parse into a CRS. Take a
    shortcut when possible.

    if the grid_mapping has a known EASE2 grid name, use the EPSG code known
    apriori.

    Args:
      dt: the source file as an opened DataTree

      variables: set of variables all sharing the same 2-dimensional grid is
                 traversed looking for a grid_mapping.

    Returns:
      CRS object

    """
    for varname in variables:
        var = dt[varname]
        if 'grid_mapping' in var.attrs:
            try:
                return CRS.from_cf(dt[var.attrs['grid_mapping']].attrs)
            except CRSError as e:
                raise InvalidSourceCRS(
                    'Could not create a CRS from grid_mapping metadata'
                ) from e

    raise InvalidSourceCRS('No grid_mapping metadata found.')


def _compute_area_extent_from_regular_x_y_coords(
    xvalues: np.ndarray, yvalues: np.ndarray
) -> tuple[np.float64, np.float64, np.float64, np.float64]:
    """Return outer extent of regularly defined grid.

    Given xvalues and yvalues represent the center values of a regularly spaced
    array, compute the cell height and width, return the outer bounds extent of
    the grid area.

    Returns:
      tuple: area_extent defintion
          (lower_left_x, lower_left_y, upper_right_x, upper_right_y)

    """
    min_x, max_x = _compute_array_bounds(xvalues)
    min_y, max_y = _compute_array_bounds(yvalues)
    return (
        np.min([min_x, max_x]),
        np.min([min_y, max_y]),
        np.max([min_x, max_x]),
        np.max([min_y, max_y]),
    )


def _compute_array_bounds(values: np.ndarray) -> tuple[np.float64, np.float64]:
    """Returns external edges of array bounds.

    If values holds an array of regulary spaced cell centers, return the outer
    edges of the array by computing the cell width and adding half of that to
    each end of the vector.

    Args:
      values: np.array of regularly spaced values with length > 1 representing
      cell centers of a gridcell.

    Returns:
      tuple: bounding extent of the input array

    """
    if len(values) < 2:
        raise SourceDataError('coordinates must have at least 2 values')

    diffs = np.diff(values)
    # SPL4CMDL (v7,v8) have +/- 1.5m variance in their cell centers. relax
    # spacing to allow for up to 3 meters difference in adjacent cells.
    if not np.allclose(diffs, diffs[0], atol=3):
        raise SourceDataError('coordinates are not regularly spaced')

    half_width = (values[1] - values[0]) / 2.0
    left = values[0] - half_width
    right = values[-1] + half_width
    return (left, right)
