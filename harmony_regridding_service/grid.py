"""Module for accessing and creating grid parameters."""

from logging import getLogger

import numpy as np
import xarray as xr
from harmony_service_lib.message import Message as HarmonyMessage
from harmony_service_lib.message_utility import (
    has_dimensions,
    has_scale_extents,
    has_scale_sizes,
)
from netCDF4 import Dataset
from pyproj import CRS, transformer
from pyproj.exceptions import CRSError
from pyresample import create_area_def
from pyresample.geometry import AreaDefinition, SwathDefinition
from varinfo import VarInfoFromNetCDF4

from harmony_regridding_service.dimensions import (
    get_column_dims,
    get_row_dims,
)
from harmony_regridding_service.exceptions import (
    InvalidSourceCRS,
    InvalidSourceDimensions,
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
    multi-dimensional variable will be regridded to this target.

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
    # ScaleExtent is required and validated.
    logger.info('compute target_area')

    # Get the target grid parameters from either the Harmony Message or
    # from input file if the grid parameters are not specified.
    area_extent, height, width = get_target_grid_parameters(message, filepath, var_info)
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


def get_target_grid_parameters(
    message: HarmonyMessage,
    filepath: str,
    var_info: VarInfoFromNetCDF4,
) -> tuple[tuple, int, int, str]:
    """Retrieve the target grid parameters.

    If all the required parameters exist in the Harmony message,
    they are simply extracted from the message. If all the parameters do not
    exist, they are created.

    """
    if has_scale_extents(message) and (
        has_scale_sizes(message) or has_dimensions(message)
    ):
        return get_grid_parameters_from_message(message)
    return create_grid_parameters(filepath, var_info, message.format.crs or 'EPSG:4326')


def create_grid_parameters(
    filepath: str,
    var_info: VarInfoFromNetCDF4,
    crs: str,
) -> tuple[tuple, int, int]:
    """Create the target grid parameters using the source grid information."""
    area_extent = get_source_area_extent(filepath, var_info, crs)
    scale_extent = {
        'x': {'min': area_extent[0], 'max': area_extent[2]},
        'y': {'min': area_extent[1], 'max': area_extent[3]},
    }

    x_res, y_res = calculate_source_resolution(filepath, var_info)
    scale_size = {
        'x': x_res,
        'y': y_res,
    }

    width = int(
        round((scale_extent['x']['max'] - scale_extent['x']['min']) / scale_size['x'])
    )
    height = int(
        round((scale_extent['y']['max'] - scale_extent['y']['min']) / scale_size['y'])
    )

    return area_extent, height, width


def get_source_area_extent(
    filepath: str,
    var_info: VarInfoFromNetCDF4,
    crs: str,
) -> tuple[np.float64, np.float64, np.float64, np.float64]:
    """Get the area extent from the input granule.

    If the source data is projection-gridded, transform it to latitude
    and longitude.

    """
    xvalues, yvalues = get_x_y_grid_values(filepath, var_info)
    area_extent = compute_area_extent_from_regular_x_y_coords(xvalues, yvalues)
    dimensions = var_info.get_required_dimensions(var_info.get_all_variables())

    if dims_are_projected_x_y(dimensions, var_info):
        return transform_area_extent_to_geographic(filepath, var_info, area_extent, crs)

    return area_extent


def get_x_y_grid_values(
    filepath: str, var_info: VarInfoFromNetCDF4
) -> tuple[np.array, np.array]:
    """Retrieve the x and y grid values.

    Note/future work: This code currently assumes the input granule only
    contains one grid, and thus only one set of dimension variables.
    This is not always the case, for example ATL16 contains three grids:
    global, north polar, and south polar. This will not return the
    correct x and y values for each grid when a multi-grid collection
    is requested.

    """
    try:
        with xr.open_datatree(filepath) as dt:
            dimensions = var_info.get_required_dimensions(var_info.get_all_variables())
            xvalues = dt[get_column_dims(dimensions, var_info)[0]].data
            yvalues = dt[get_row_dims(dimensions, var_info)[0]].data
    except Exception as e:
        logger.error(e)
        raise SourceDataError('Cannot retrieve source grid.') from e

    return xvalues, yvalues


def transform_area_extent_to_geographic(
    filepath: str,
    var_info: VarInfoFromNetCDF4,
    area_extent: tuple[np.float64, np.float64, np.float64, np.float64],
    output_crs: str,
) -> tuple[np.float64, np.float64, np.float64, np.float64]:
    """Convert area extent to latitude and longitude.

    Given an area extent (lower_left_x, lower_left_y, upper_right_x,
    upper_right_y), transform the values to geographic.

    """
    try:
        with xr.open_datatree(filepath) as dt:
            input_crs = crs_from_source_data(dt, var_info.get_all_variables())
    except Exception as e:
        logger.error(e)
        raise InvalidSourceCRS('Invalid source CRS') from e

    transform_to_geo = transformer.Transformer.from_crs(
        input_crs, output_crs, always_xy=True
    )
    longitude_extent, latitude_extent = transform_to_geo.transform(
        area_extent[2], area_extent[3]
    )  # xmax, ymax

    longitude_max = abs(longitude_extent)
    longitude_min = -longitude_max
    latitude_max = abs(latitude_extent)
    latitude_min = -latitude_max

    return (longitude_min, latitude_min, longitude_max, latitude_max)


def calculate_source_resolution(
    filepath: str,
    var_info: VarInfoFromNetCDF4,
) -> tuple[np.float64, np.float64]:
    """Calculate the resolution found in the input granule.

    If the input file is projection-gridded (resolution is in meters),
    then the resolution (i.e., scale size/cell spacing) is converted to
    degrees latitude at true scale.

    Meters to degrees conversion factor:
    - Earth circumference at the equator: 40,075,000 meters
    - Earth circumference / 360deg = 111,319.444444 meters/deg

    """
    METERS_PER_DEGREE = 111319.444444

    dimension_vars_mapping = var_info.group_variables_by_horizontal_dimensions()

    xvalues, yvalues = get_x_y_grid_values(filepath, var_info)
    x_res = abs(xvalues[1] - xvalues[0])
    y_res = abs(yvalues[1] - yvalues[0])

    for dimensions, _ in dimension_vars_mapping.items():
        if len(dimensions) == 2:
            if dims_are_projected_x_y(dimensions, var_info):
                x_res = x_res / METERS_PER_DEGREE
                y_res = y_res / METERS_PER_DEGREE

    return x_res, y_res


def get_grid_parameters_from_message(
    message: HarmonyMessage,
) -> tuple[tuple, int, int]:
    """Retrieve the target grid parameters specified in the Harmony request."""
    area_extent = (
        message.format.scaleExtent.x.min,
        message.format.scaleExtent.y.min,
        message.format.scaleExtent.x.max,
        message.format.scaleExtent.y.max,
    )

    height = grid_height(message)
    width = grid_width(message)

    return area_extent, height, width


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


def compute_source_swath(
    grid_dimensions: tuple[str, str],
    filepath: str,
    var_info: VarInfoFromNetCDF4,
    variable_set: set,
) -> SwathDefinition:
    """Return a SwathDefinition for the input grid_dimensions."""
    if dims_are_lon_lat(grid_dimensions, var_info):
        longitudes, latitudes = compute_horizontal_source_grids(
            grid_dimensions, filepath, var_info
        )
    elif dims_are_projected_x_y(grid_dimensions, var_info):
        longitudes, latitudes = compute_projected_horizontal_source_grids(
            grid_dimensions, filepath, var_info, variable_set
        )
    else:
        raise SourceDataError(
            'Cannot determine correct dimension type from source {grid_dimensions}.'
        )

    return SwathDefinition(lons=longitudes, lats=latitudes)


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
    variables: set,
) -> tuple[np.ndarray, np.ndarray]:
    """Return longitude and latitude grids for a projected grid_dimensions pair.

    Given the input grid_dimensions pair, find the projected coordinate dimensions
    in the source data and use those to generate 2D longitude and latitude arrays.

    """
    xdim_name = get_column_dims(grid_dimensions, var_info)[0]
    ydim_name = get_row_dims(grid_dimensions, var_info)[0]
    try:
        with xr.open_datatree(filepath) as dt:
            xvalues = dt[xdim_name].data
            yvalues = dt[ydim_name].data
            area_extent = compute_area_extent_from_regular_x_y_coords(xvalues, yvalues)
            source_crs = crs_from_source_data(dt, variables)
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
