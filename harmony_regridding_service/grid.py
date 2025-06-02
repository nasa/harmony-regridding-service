"""Module for accessing and creating grid parameters."""

from logging import getLogger

import numpy as np
import xarray as xr
from harmony_service_lib.message import Message as HarmonyMessage
from harmony_service_lib.message_utility import (
    has_dimensions,
    has_scale_extents,
    has_scale_sizes,
    has_self_consistent_grid,
)
from netCDF4 import Dataset
from pyproj import CRS
from pyproj.exceptions import CRSError
from pyresample import create_area_def
from pyresample.geometry import AreaDefinition
from varinfo import VarInfoFromNetCDF4

from harmony_regridding_service.dimensions import (
    get_column_dims,
    get_resampled_dimension_pairs,
    get_row_dims,
)
from harmony_regridding_service.exceptions import (
    InvalidCRSResampling,
    InvalidSourceCRS,
    InvalidSourceDimensions,
    InvalidTargetGrid,
    SourceDataError,
)

logger = getLogger(__name__)


def compute_target_area(
    message: HarmonyMessage,
    filepath: str,
    var_info: VarInfoFromNetCDF4,
) -> AreaDefinition:
    """Define the output area for your regridding operation.

    Parse the harmony message and build a target AreaDefinition.  All
    multi-dimensional variables will be regridded to this target.

    Computed parameters:
    ----------------------
    Area extent: [tuple]
        The two real-world projection coordinate pairs of the grid's upper
        right and lower left points (in order):
        - lower left x coordinate of the lower left pixel
        - lower left y coordinate of the lower left pixel
        - upper right x coordinate of the upper right pixel
        - upper right y coordinate of the upper right pixel
    Height: [int]
        The number of grid rows.
    Width: [int]
        The number of grid columns.
    Projection: [dict]
        The target Coordinate Reference System (CRS) represented by a dictionary with an
        EPSG code, proj4 string, or wkt key string.

    """
    logger.info('compute target_area')

    if has_scale_extents(message) and (
        has_scale_sizes(message) or has_dimensions(message)
    ):
        return get_area_definition_from_message(message)

    if same_source_and_target_crs(var_info):
        raise InvalidCRSResampling(
            'requested a resampling with no grid parameters to same CRS'
        )

    return create_target_area_from_source(filepath, var_info)


def same_source_and_target_crs(
    var_info: VarInfoFromNetCDF4,
) -> bool:
    """Check if the requested CRS is the same as the input CRS.

    For now, it is assumed that only one geographic CRS is available, but this
    may not be the case in the future.
    """
    grid_dimensions = get_resampled_dimension_pairs(var_info)[0]
    return dims_are_lon_lat(grid_dimensions, var_info)


def create_target_area_from_source(
    filepath: str,
    var_info: VarInfoFromNetCDF4,
) -> AreaDefinition:
    """Create the target area definition using the source grid information.

    TODO: Create area definition for every dimension pair, if the collection
    had more than one grid.
    """
    dimension_pairs = get_resampled_dimension_pairs(var_info)
    return create_area_definition_for_source_grid(
        filepath, dimension_pairs[0], var_info
    )


def get_variables_for_dimension_pair(dim_pair, var_info):
    """Return the variables associated with the input 2D dimension pair."""
    dim_mapping = var_info.group_variables_by_horizontal_dimensions()
    return dim_mapping[dim_pair]


def get_area_definition_from_message(
    message: HarmonyMessage,
) -> AreaDefinition:
    """Retrieve the target grid area definition from the Harmony message.

    Create the area definition using the target grid specified in the Harmony
    request.
    """
    if not has_self_consistent_grid(message):
        raise InvalidTargetGrid()

    area_extent = (
        message.format.scaleExtent.x.min,
        message.format.scaleExtent.y.min,
        message.format.scaleExtent.x.max,
        message.format.scaleExtent.y.max,
    )

    height = grid_height(message)
    width = grid_width(message)

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


def grid_height(message: HarmonyMessage) -> int:
    """Compute grid height from Message.

    Compute the height of grid from the scaleExtents and scale_sizes.
    """
    if has_dimensions(message):
        return message.format.height
    return compute_num_elements(message, 'y')


def grid_width(message: HarmonyMessage) -> int:
    """Compute grid height from Message.

    Compute the height of grid from the scaleExtents and scale_sizes.
    """
    if has_dimensions(message):
        return message.format.width
    return compute_num_elements(message, 'x')


def compute_num_elements(message: HarmonyMessage, dimension_name: str) -> int:
    """Compute the number of gridcells based on scaleExtents and scaleSize."""
    scale_extent = getattr(message.format.scaleExtent, dimension_name)
    scale_size = getattr(message.format.scaleSize, dimension_name)

    num_elements = int(np.round((scale_extent.max - scale_extent.min) / scale_size))
    return num_elements


def compute_horizontal_source_grids(
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
    row_dim = get_row_dims(grid_dimensions, var_info)[0]
    column_dim = get_column_dims(grid_dimensions, var_info)[0]
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


def compute_projected_horizontal_source_grids(
    grid_dimensions: tuple[str, str],
    filepath: str,
    var_info: VarInfoFromNetCDF4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return longitude and latitude grids for a projected grid_dimensions pair.

    Given the input grid_dimensions pair, find the projected coordinate dimensions
    in the source data and use those to generate 2D longitude and latitude arrays.

    """
    source_area = create_area_definition_for_source_grid(
        filepath, grid_dimensions, var_info
    )
    return source_area.get_lonlats()


def create_area_definition_for_source_grid(
    filepath: str,
    dimension_pair: tuple[str, str],
    var_info: VarInfoFromNetCDF4,
) -> AreaDefinition:
    """Return the area definition given a grid dimensions pair.

    Find the projected coordinate dimensions in the source data
    and use those to create the correlating area definition.

    """
    variables = get_variables_for_dimension_pair(dimension_pair, var_info)
    xdim_name = get_column_dims(dimension_pair, var_info)[0]
    ydim_name = get_row_dims(dimension_pair, var_info)[0]
    try:
        with xr.open_datatree(
            filepath,
            decode_cf=False,
            decode_coords=False,
            decode_timedelta=False,
            decode_times=False,
        ) as dt:
            xvalues = dt[xdim_name].data
            yvalues = dt[ydim_name].data
            area_extent = compute_area_extent_from_regular_x_y_coords(xvalues, yvalues)
            source_crs = crs_from_source_data(dt, variables)
            cell_width = np.abs(xvalues[1] - xvalues[0])
            cell_height = np.abs(yvalues[1] - yvalues[0])
            return create_area_def(
                'source grid area',
                source_crs,
                area_extent=area_extent,
                shape=(len(yvalues), len(xvalues)),
                resolution=(cell_width, cell_height),
            )
    except Exception as e:
        logger.error(e)
        raise SourceDataError('cannot compute projected source grids') from e


def compute_area_extent_from_regular_x_y_coords(
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
    min_x, max_x = compute_array_bounds(xvalues)
    min_y, max_y = compute_array_bounds(yvalues)
    return (
        np.min([min_x, max_x]),
        np.min([min_y, max_y]),
        np.max([min_x, max_x]),
        np.max([min_y, max_y]),
    )


def compute_array_bounds(values: np.ndarray) -> tuple[np.float64, np.float64]:
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


def crs_from_source_data(dt: xr.DataTree, variables: set) -> CRS:
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


def dims_are_lon_lat(dimensions: tuple[str, str], var_info: VarInfoFromNetCDF4) -> bool:
    """Does the dimension pair represent longitudes/latitudes."""
    return all(
        var_info.get_variable(dim_name).is_geographic() for dim_name in dimensions
    )


def dims_are_projected_x_y(
    dimensions: tuple[str, str], var_info: VarInfoFromNetCDF4
) -> bool:
    """Does the dimension pair represent projected x/y values."""
    return all(
        var_info.get_variable(dim_name).is_projection_x_or_y()
        for dim_name in dimensions
    )
