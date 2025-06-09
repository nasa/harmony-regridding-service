"""Module for accessing and creating grid parameters."""

from collections.abc import Iterable
from logging import getLogger
from typing import Any

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
    GridDimensionPair,
    get_column_dims,
    get_resampled_dimension_pairs,
    get_row_dims,
    horizontal_dims_for_variable,
)
from harmony_regridding_service.exceptions import (
    InvalidCRSResampling,
    InvalidSourceCRS,
    InvalidSourceDimensions,
    InvalidTargetGrid,
    SourceDataError,
)
from harmony_regridding_service.message_utilities import (
    get_message_crs,
    target_crs_from_message,
)

logger = getLogger(__name__)


def compute_target_areas(
    message: HarmonyMessage,
    filepath: str,
    var_info: VarInfoFromNetCDF4,
) -> dict[GridDimensionPair, AreaDefinition]:
    """Define the output areas for your regridding operations.

    This function computes the target areas for resampling.

    First The HarmonyMessage is searched for grid information, if this is
    found this grid is used for all resampling operations in the file.

    If there is no grid information in the HarmonyMessage, a target area is
    generated from each horizontal grid pair in the source granule. That is,
    we generate a target area for each unique grid in the input data.

    We return a dictionary of the grid dimension pair pointing to their
    AreaDefinitions

    """
    if any(
        (has_scale_extents(message), has_scale_sizes(message), has_dimensions(message))
    ):
        # If there's any grid parts, get the target area from the message and
        # store it on every grid in the source.
        area_definition = get_area_definition_from_message(message)
        area_definitions = {
            dim_pair: area_definition
            for dim_pair in get_resampled_dimension_pairs(var_info)
        }

    elif same_source_and_target_crs(message, var_info):
        # Don't resample if you the source and target have the same CRS
        raise InvalidCRSResampling(
            'Requested a resampling with matching '
            'source and target CRS and no grid parameters.'
        )
    else:
        # compute target grids from the source data
        target_crs = CRS(get_message_crs(message) or 'epsg:4326')
        area_definitions = create_target_areas_from_source(
            filepath, var_info, target_crs
        )

    logger.debug('Using TARGET Area Definitions:')
    for dim_pair, area in area_definitions.items():
        logger.debug(f'dim_pair: {dim_pair}')
        logger.debug(f'target Area: {area}')
        logger.debug(f'target Area_extent: {area.area_extent}')
        logger.debug(f'target crs: {area.crs.to_proj4()}')

    return area_definitions


def same_source_and_target_crs(
    message: HarmonyMessage,
    var_info: VarInfoFromNetCDF4,
) -> bool:
    """Check if the requested CRS is the same as the input CRS.

    For this version we are assuming that the only grids we need to be
    concerned with are the first grid returned from
    get_resampled_dimension_pairs.

    """
    grid_dimensions = get_resampled_dimension_pairs(var_info)[0]
    vars_on_this_grid = get_variables_on_grid(grid_dimensions, var_info)
    source_crs = crs_from_source_data(vars_on_this_grid, var_info)
    target_crs = target_crs_from_message(message)
    return target_crs.equals(source_crs, ignore_axis_order=True)


def create_target_areas_from_source(
    filepath: str, var_info: VarInfoFromNetCDF4, target_crs: CRS
) -> dict[GridDimensionPair, AreaDefinition]:
    """Create the target areas using the source grid information.

    Loop over the source grids generating the correct target areas for each.

    return a dictionary of the grid dimension pairs to their corresponding
    AreaDefinitions.

    """
    dimension_pairs = get_resampled_dimension_pairs(var_info)
    target_areas = {}
    for dim_pair in dimension_pairs:
        logger.info(f'Generating Target Areas from Source for: {dim_pair}')
        projected_area = create_area_definition_for_projected_source_grid(
            filepath, dim_pair, var_info
        )
        # I have the correct projected areaDefinition, but I want to convert it to
        # a geographic area
        target_areas[dim_pair] = convert_projected_area_to_geographic(
            projected_area, target_crs
        )

    return target_areas


def convert_projected_area_to_geographic(
    projected_area: AreaDefinition, target_crs: CRS
) -> AreaDefinition:
    """Converts a projected AreaDefinition into the similar area in the target_CRS.

    For now the target_crs is always going to be CRS('epsg:4326')

    The geographic extent is found by looking at the lat/lon values of every
    grid cell in the projected area and computing the min/max of those to get
    an extent.

    The geographic resolution is computed in degrees at using a conversion from
    meters at the equator.

    create_area_def will adjust the area by rounding to the nearest whole
    number columns and rows and adjusting the resolution accordingly.

    """
    geographic_extent = get_geographic_area_extent(projected_area)
    resolution = get_geographic_resolution(projected_area)

    geographic_area = create_area_def(
        'Geographic Area',
        target_crs,
        area_extent=geographic_extent,
        resolution=resolution,
    )
    logger.debug(f'Source projected Area: {projected_area}')
    logger.debug(f'Converted Geographic Area: {geographic_area}')

    return geographic_area


def get_geographic_area_extent(
    projected_area: AreaDefinition,
) -> tuple[float, float, float, float]:
    """Return the geographic area extent.

    Compute the latitude and longitude for every grid cell in the
    projected_area's grid and return the area extent in lon and lat.

    """
    lons, lats = projected_area.get_lonlats()
    return (np.min(lons), np.min(lats), np.max(lons), np.max(lats))


def get_geographic_resolution(projected_area: AreaDefinition) -> tuple[float, float]:
    """Given a projected area, compute the equivalent geographic resolution.

    The projected grid resolution in x/y meters is converted to an equivalent
    lon/lat in degrees using the conversion from meters to degrees at the
    equator using the WGS84 equatorial radius of 6,378,137 meters.

    """
    meters_per_degree = 111320.0
    return tuple(
        cell_dimension / meters_per_degree
        for cell_dimension in projected_area.resolution
    )


def reorder_extents(min_x, min_y, max_x, max_y):
    """This is a way to ensure the correct area extents.

    # The pyresample generated area_extent_ll is not always returning the
    # expected values for the area extent point documented as (lower_left_lon,
    # lower_left_lat, upper_right_lon, upper_right_lat) Resulting in some bad
    # resampling outputs. This may be a stop-gap/workaround?

    Returns:
      tuple: (lower_left_x, lower_left_y, upper_right_x, upper_right_y)

    """
    return (
        np.min([min_x, max_x]),
        np.min([min_y, max_y]),
        np.max([min_x, max_x]),
        np.max([min_y, max_y]),
    )


def get_variables_on_grid(dim_pair, var_info):
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

    area_extent = reorder_extents(
        message.format.scaleExtent.x.min,
        message.format.scaleExtent.y.min,
        message.format.scaleExtent.x.max,
        message.format.scaleExtent.y.max,
    )

    height = grid_height(message)
    width = grid_width(message)

    projection = message.format.crs or 'EPSG:4326'

    logger.info(
        f'Creating target area from message:\n'
        f'proj:{projection}\n'
        f'area_extent:{area_extent}\n'
        f'height:{height}\nwidth:{width}'
    )

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
    source_area = create_area_definition_for_projected_source_grid(
        filepath, grid_dimensions, var_info
    )
    return source_area.get_lonlats()


def create_area_definition_for_projected_source_grid(
    filepath: str,
    dimension_pair: tuple[str, str],
    var_info: VarInfoFromNetCDF4,
) -> AreaDefinition:
    """Return the area definition for a source grid.

    Use the projected coordinate dimensions in the source data to compute the
    area definition.
    """
    variables = get_variables_on_grid(dimension_pair, var_info)
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
            area_crs = crs_from_source_data(variables, var_info)
            cell_width = np.abs(xvalues[1] - xvalues[0])
            cell_height = np.abs(yvalues[1] - yvalues[0])
            return create_area_def(
                'grid area',
                area_crs,
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

    return reorder_extents(min_x, min_y, max_x, max_y)


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


def crs_from_source_data(variables: Iterable, var_info: VarInfoFromNetCDF4) -> CRS:
    """Create a CRS from grid variables in the source data.

    Given a list of one or more variables, all on the same horizontal
    grid:

    Look through the variables' metadata for a grid_mapping that can be used to
    create a CRS and return it.

    If no grid_mapping information can be made from any of the variables'
    metadata, check the horizontal dimensions see if they are geographic, if so,
    assume a CRS of EPSG:4326 and return that.

    If you can't determine a CRS after that, raise an InvalidSourceCRS
    exception.

    Args:
      variables:  variables all sharing the same 2-dimensional grid.

      var_info: used to retrive grid_mapping metadata.

    Returns:
      CRS object

    """
    for var_name in variables:
        cf_attributes = get_grid_mapping_attributes(var_name, var_info)
        if cf_attributes:
            try:
                return CRS.from_cf(cf_attributes)
            except CRSError as e:
                raise InvalidSourceCRS(
                    'Could not create a CRS from grid_mapping metadata'
                ) from e

    # No grid_mapping metadata was found so check the dimensions for geographic
    # information and assume EPSG:4326 if so.
    for var_name in variables:
        if has_geographic_grid_dimensions(var_name, var_info):
            return CRS.from_epsg(4326)

    raise InvalidSourceCRS('No grid_mapping metadata found.')


def get_grid_mapping_attributes(
    var_name: str, var_info: VarInfoFromNetCDF4
) -> dict[str, Any]:
    """Return the grid mapping attributes for a variable.

    Use varinfo to get the metadata associated with the grid mapping variable.
    """
    grid_mapping = var_info.get_variable(var_name).references.get('grid_mapping', set())
    coordinates = var_info.get_variable(var_name).references.get('coordinates', set())
    grid_mapping_var_name = list(grid_mapping - coordinates)
    if grid_mapping_var_name and len(grid_mapping_var_name) == 1:
        return var_info.get_variable(grid_mapping_var_name[0]).attributes
    return {}


def dims_are_lon_lat(dimensions: tuple[str, str], var_info: VarInfoFromNetCDF4) -> bool:
    """Does the dimension pair represent longitudes/latitudes."""
    return all(
        var_info.get_variable(dim_name).is_geographic() for dim_name in dimensions
    )


def has_geographic_grid_dimensions(var_name: str, var_info: VarInfoFromNetCDF4) -> bool:
    """Returns true if the horizontal dimensions for the variable are geographic."""
    return dims_are_lon_lat(horizontal_dims_for_variable(var_info, var_name), var_info)


def dims_are_projected_x_y(
    dimensions: tuple[str, str], var_info: VarInfoFromNetCDF4
) -> bool:
    """Does the dimension pair represent projected x/y values."""
    return all(
        var_info.get_variable(dim_name).is_projection_x_or_y()
        for dim_name in dimensions
    )
