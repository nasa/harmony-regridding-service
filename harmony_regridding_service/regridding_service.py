"""Regridding service code."""
from __future__ import annotations

from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Set, Tuple

import numpy as np
from harmony.message import Message, Source
from harmony.util import generate_output_filename
from netCDF4 import Dataset, Dimension, Variable
from pyresample.ewa import DaskEWAResampler
from pyresample.geometry import AreaDefinition, SwathDefinition
from varinfo import VarInfoFromNetCDF4

from harmony_regridding_service.exceptions import (InvalidSourceDimensions,
                                                   RegridderException)
from harmony_regridding_service.utilities import has_dimensions

if TYPE_CHECKING:
    from harmony_regridding_service.adapter import RegriddingServiceAdapter

HRS_VARINFO_CONFIG_FILENAME = str(
    Path(Path(__file__).parent, 'config', 'HRS_varinfo_config.json'))


def regrid(adapter: RegriddingServiceAdapter, input_filepath: str,
           source: Source) -> str:
    """Regrid the input data at input_filepath."""
    var_info = VarInfoFromNetCDF4(input_filepath,
                                  adapter.logger,
                                  short_name=source.shortName,
                                  config_file=HRS_VARINFO_CONFIG_FILENAME)
    target_area = _compute_target_area(adapter.message)

    resampler_cache = _cache_resamplers(input_filepath, var_info, target_area)
    adapter.logger.info(f'cached resamplers for {resampler_cache.keys()}')

    target_filepath = generate_output_filename(input_filepath,
                                               is_regridded=True)

    with Dataset(input_filepath, mode='r') as source_ds, \
         Dataset(target_filepath, mode='w', format='NETCDF4') as target_ds:

        _transfer_dimensions(source_ds, target_ds, target_area, var_info)

        vars_to_process = var_info.get_all_variables()
        vars_to_process -= _clone_variables(source_ds, target_ds,
                                            _unresampled_variables(var_info))
        vars_to_process -= _copy_dimension_variables(source_ds, target_ds,
                                                     target_area, var_info)

        adapter.logger.info(
            f'variables transferred: {var_info.get_all_variables() - vars_to_process}'
        )
        adapter.logger.info(f'variables to transfer: {vars_to_process}')

    return target_filepath


def _copy_resampled_bounds_variable(source_ds: Dataset, target_ds: Dataset,
                                    bounds_var: str,
                                    target_area: AreaDefinition,
                                    var_info: VarInfoFromNetCDF4):
    """Copy computed values for dimension variable bounds variables."""
    var_dims = var_info.get_variable(bounds_var).dimensions

    xdims = _get_projection_x_dim(var_dims, var_info)
    ydims = _get_projection_y_dim(var_dims, var_info)
    if xdims:
        target_coords = target_area.projection_x_coords
        dim_name = xdims[0]
    else:
        target_coords = target_area.projection_y_coords
        dim_name = ydims[0]

    if not var_dims[0] == dim_name:
        raise RegridderException(f'_bnds var {var_dims} with unexpected shape')

    # create the bounds variable and fill it with the correct values.
    (_, t_var) = _copy_var_without_metadata(target_ds, source_ds, bounds_var)
    bounds_width = (target_coords[1] - target_coords[0]) / 2
    lower_bounds = target_coords - bounds_width
    upper_bounds = target_coords + bounds_width
    t_var[:, 0] = lower_bounds
    t_var[:, 1] = upper_bounds

    return {bounds_var}


def _copy_dimension_variables(source_ds: Dataset, target_ds: Dataset,
                              target_area: AreaDefinition,
                              var_info: VarInfoFromNetCDF4) -> Set[str]:
    """Copy over dimension variables that are changed  in the target file."""
    dim_var_names = _resampled_dimension_variable_names(var_info)
    processed_vars = _copy_1d_dimension_variables(source_ds, target_ds,
                                                  dim_var_names, target_area,
                                                  var_info)

    bounds_vars = dim_var_names - processed_vars
    for bounds_var in bounds_vars:
        processed_vars -= _copy_resampled_bounds_variable(
            source_ds, target_ds, bounds_var, target_area, var_info)

    return processed_vars


def _copy_1d_dimension_variables(source_ds: Dataset, target_ds: Dataset,
                                 dim_var_names: Set[str],
                                 target_area: AreaDefinition,
                                 var_info: VarInfoFromNetCDF4) -> Set[str]:
    """Copy 1 dimensional dimension variables.

    These are the variables associated directly with the resampled
    longitudes, latitudes, Columns, rows, x-variables, and y-variables.
    """
    # pylint: disable-msg=too-many-locals
    one_d_vars = {
        dim_var_name for dim_var_name in dim_var_names
        if len(var_info.get_variable(dim_var_name).dimensions) == 1
    }

    xdims = _get_projection_x_dim(one_d_vars, var_info)
    ydims = _get_projection_y_dim(one_d_vars, var_info)

    for dim_name in one_d_vars:
        if dim_name in xdims:
            target_coords = target_area.projection_x_coords
            standard_metadata = {
                'long_name': 'longitude',
                'standard_name': 'longitude',
                'units': 'degrees_east'
            }
        elif dim_name in ydims:
            target_coords = target_area.projection_y_coords
            standard_metadata = {
                'long_name': 'latitude',
                'standard_name': 'latitude',
                'units': 'degrees_north'
            }
        else:
            raise RegridderException(
                f'dim_name: {dim_name} not found in projection dimensions')

        (_, t_var) = _copy_var_without_metadata(target_ds, source_ds, dim_name)

        bounds_var = _get_bounds_var(var_info, dim_name)

        if bounds_var:
            standard_metadata['bounds'] = bounds_var
        t_var.setncatts(standard_metadata)

        t_var[:] = target_coords

    return one_d_vars


def _get_bounds_var(var_info: VarInfoFromNetCDF4, dim_name: str):
    possible_extensions = ['bnds', 'bounds']
    for ext in possible_extensions:
        bounds_var = var_info.get_variable(f'{dim_name}_{ext}')
        if bounds_var:
            return bounds_var.name
    return None


def _resampled_dimension_variable_names(
        var_info: VarInfoFromNetCDF4) -> Set[str]:
    """Return the list of dimension variables to resample to target grid.

    This returns a list of the fully qualified variables that need to use the
    information from the target_area in order to compute the correct output
    values.

    """
    dims_to_transfer = set()
    resampled_dims = _resampled_dimensions(var_info)
    grouped_vars = var_info.group_variables_by_horizontal_dimensions()
    for dim in resampled_dims:
        dims_to_transfer.update(grouped_vars[(dim,)])

    return dims_to_transfer


def _transfer_dimensions(source_ds: Dataset, target_ds: Dataset,
                         target_area: AreaDefinition,
                         var_info: VarInfoFromNetCDF4) -> Tuple[str, ...]:
    """Transfer all dimensions from source to target.

    Horizontal source dimensions that are changed due to resampling, are
    add onto the target using the information from the target_area.
    """
    all_dimensions = _all_dimensions(var_info)
    resampled_dimensions = _resampled_dimensions(var_info)
    unchanged_dimensions = all_dimensions - resampled_dimensions
    _copy_dimensions(unchanged_dimensions, source_ds, target_ds)
    _create_resampled_dimensions(resampled_dimensions, target_ds, target_area,
                                 var_info)


def _clone_variables(source_ds: Dataset, target_ds: Dataset,
                     dimensions: Set[str]) -> Set[str]:
    """Clone variables from source to target.

    Copy variables and their attributes directly from the source Dataset to the
    target Dataset.
    """
    for dimension_name in dimensions:
        (s_var, t_var) = _copy_var_with_attrs(target_ds, source_ds,
                                              dimension_name)
        t_var[:] = s_var[:]

    return dimensions


def _copy_var_with_attrs(target_ds: Dataset, source_ds: Dataset,
                         variable_name: str) -> (Variable, Variable):
    """Copy a source variable and metadata to target.

    Copy both the variable and metadata from a souce variable into a target,
    return both source and target variables.
    """
    s_var, t_var = _copy_var_without_metadata(target_ds, source_ds,
                                              variable_name)

    for att in s_var.ncattrs():
        if att != '_FillValue':
            t_var.setncattr(att, s_var.getncattr(att))

    return (s_var, t_var)


def _copy_var_without_metadata(target_ds: Dataset, source_ds: Dataset,
                               variable_name: str) -> (Variable, Variable):
    """Clones a single variable and returns both source and target variables.

    This function uses the netCDF4 createGroup('/[optionalgroup/andsubgroup]')
    call This will return an existing group, or create one that does not
    already exists. So this is not clobbering the source data.

    """
    var = PurePath(variable_name)
    s_var = _get_variable(source_ds, variable_name)
    t_group = target_ds.createGroup(var.parent)
    fill_value = getattr(s_var, '_FillValue', None)
    t_var = t_group.createVariable(var.name,
                                   s_var.dtype,
                                   s_var.dimensions,
                                   fill_value=fill_value)

    return (s_var, t_var)


def _get_variable(dataset: Dataset, variable_name: str) -> Variable:
    """Return a variable from a fully qualified variable name.

    This will return an existing or create a new variable.
    """
    var = PurePath(variable_name)
    group = dataset.createGroup(var.parent)
    return group[var.name]


def _create_dimension(dataset: Dataset, dimension_name: str,
                      size: int) -> Dimension:
    """Create a fully qualified dimension on the dataset."""
    dim = PurePath(dimension_name)
    group = dataset.createGroup(dim.parent)
    return group.createDimension(dim.name, size)


def _create_resampled_dimensions(resampled_dims: Set[str], dataset: Dataset,
                                 area: AreaDefinition,
                                 var_info: VarInfoFromNetCDF4):
    """Create dimensions for the target resampled grids.

    TODO [MHS, 03/17/2023] I think this is broken for multiple x/y grids the
    way that it's called in _transfer_dimensions

    """
    xdim = _get_projection_x_dim(resampled_dims, var_info)[0]
    ydim = _get_projection_y_dim(resampled_dims, var_info)[0]

    _create_dimension(dataset, xdim, area.projection_x_coords.shape[0])
    _create_dimension(dataset, ydim, area.projection_y_coords.shape[0])


def _copy_dimension(dimension_name: str, source_ds: Dataset,
                    target_ds: Dataset) -> str:
    """Copy dimension from source to target file."""
    source_dimension = _get_dimension(source_ds, dimension_name)

    source_size = None
    if not source_dimension.isunlimited():
        source_size = source_dimension.size

    dim = PurePath(dimension_name)
    target_group = target_ds.createGroup(dim.parent)
    return target_group.createDimension(dim.name, source_size)


def _get_dimension(dataset: Dataset, dimension_name: str) -> Dimension:
    """Return a dimension object for a dimension name.

    Return the Dimension for an arbitrarily nested dimension name.
    """
    dim = PurePath(dimension_name)
    return dataset.createGroup(dim.parent).dimensions[dim.name]


def _copy_dimensions(dimensions: Set[str], source_ds: Dataset,
                     target_ds: Dataset) -> Set[str]:
    """Copy each dimension from source to target.

    ensure the first dimensions copied are the UNLIMITED dimensions.
    """

    def sort_unlimited_first(dimension_name):
        """Sort dimensions so that unlimited are first in list."""
        the_dim = _get_dimension(source_ds, dimension_name)
        return not the_dim.isunlimited()

    sorted_dims = sorted(list(dimensions), key=sort_unlimited_first)

    for dim in sorted_dims:
        _copy_dimension(dim, source_ds, target_ds)


def _all_dimensions(var_info: VarInfoFromNetCDF4) -> Set[str]:
    """Return a list of all dimensions in the file."""
    dimensions = set()
    for variable_name in var_info.get_all_variables():
        variable = var_info.get_variable(variable_name)
        for dim in variable.dimensions:
            dimensions.add(dim)

    return dimensions


def _unresampled_variables(var_info: VarInfoFromNetCDF4) -> Set[str]:
    """Variable names to be transfered from source to target without change.

    returns a set of variables that do not have any dimension that is also in
    the set of resampled_dimensions.

    """
    vars_by_dim = var_info.group_variables_by_dimensions()
    resampled_dims = _resampled_dimensions(var_info)

    return set.union(*[
        variable_set for dimension_name, variable_set in vars_by_dim.items()
        if not resampled_dims.intersection(set(dimension_name))
    ])


def _all_dimension_variables(var_info: VarInfoFromNetCDF4) -> Set[str]:
    """Return a set of every dimension variable name in the file."""
    return var_info.get_required_dimensions(var_info.get_all_variables())


def _resampled_dimensions(var_info: VarInfoFromNetCDF4) -> Set[str]:
    """Return a set of resampled dimension names.

    Using varinfo's group_by_horizontal_dimensions, return a list of the
    dimension variables that appear in the group with length 2 as these are the
    dimensions that we will be resampling against.

    """
    dimensions = set()
    for dims in var_info.group_variables_by_horizontal_dimensions():
        if len(dims) == 2:
            dimensions.update(list(dims))

    return dimensions


def _open_target_file(input_filepath: str) -> Dataset:
    """Open a working netcdf Dataset file."""
    target_filepath = generate_output_filename(input_filepath,
                                               is_regridded=True)
    target_ds = Dataset(target_filepath, mode='w', format='NETCDF4')
    return target_ds


def _cache_resamplers(filepath: str, var_info: VarInfoFromNetCDF4,
                      target_area: AreaDefinition) -> None:
    """Precompute the resampling weights.

    Determine the desired output Target Area from the Harmony Message.  Use
    this target area in conjunction with each shared horizontal dimension in
    the input source file to create an EWA Resampler and precompute the weights
    to be used in a resample from the shared horizontal dimension to the output
    target area.

    """
    grid_cache = {}
    dimension_variables_mapping = var_info.group_variables_by_horizontal_dimensions(
    )

    for dimensions in dimension_variables_mapping:
        # create source swath definition from 2D grids
        if len(dimensions) == 2:
            source_swath = _compute_source_swath(dimensions, filepath, var_info)
            grid_cache[dimensions] = DaskEWAResampler(source_swath, target_area)

    for resampler in grid_cache.values():
        resampler.precompute(rows_per_scan=0)

    return grid_cache


def _compute_target_area(message: Message) -> AreaDefinition:
    """Parse the harmony message and build a target AreaDefinition."""
    # ScaleExtent is required and validated.
    area_extent = (message.format.scaleExtent.x.min,
                   message.format.scaleExtent.y.min,
                   message.format.scaleExtent.x.max,
                   message.format.scaleExtent.y.max)

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
        np.round((scale_extent.max - scale_extent.min) / scale_size))
    return num_elements


def _get_projection_x_dim(dims: Tuple[str, str],
                          var_info: VarInfoFromNetCDF4) -> str:
    """Return name for horizontal grid dimension [column/longitude/x].

    TODO [MHS, 03/17/2023] This could/should just return empty set.
    """
    column_dims = []
    for dim in dims:
        try:
            if var_info.get_variable(dim).is_longitude():
                column_dims.append(dim)
        except AttributeError:
            pass

    return column_dims


def _get_projection_y_dim(dims: Tuple[str, str],
                          var_info: VarInfoFromNetCDF4) -> str:
    """Return name for vertical grid dimension [row/latitude/y]."""
    row_dims = []
    for dim in dims:
        try:
            if var_info.get_variable(dim).is_latitude():
                row_dims.append(dim)
        except AttributeError:
            pass

    return row_dims


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
    row_dim = _get_projection_y_dim(grid_dimensions, var_info)[0]
    column_dim = _get_projection_x_dim(grid_dimensions, var_info)[0]

    with Dataset(filepath, mode='r') as data_set:
        row_shape = data_set[row_dim].shape
        column_shape = data_set[column_dim].shape

        if (len(row_shape) == 1 and len(column_shape) == 1):
            num_rows = row_shape[0]
            num_columns = column_shape[0]
            longitudes = np.broadcast_to(data_set[column_dim],
                                         (num_rows, num_columns))
            latitudes = np.broadcast_to(
                np.broadcast_to(data_set[row_dim], (1, num_rows)).T,
                (num_rows, num_columns))
            longitudes = np.ascontiguousarray(longitudes)
            latitudes = np.ascontiguousarray(latitudes)
        else:
            # Only handling the case of 1-Dimensional dimensions on MVP
            raise InvalidSourceDimensions(
                f'Incorrect source data dimensions. '
                f'rows:{row_shape}, columns:{column_shape}')

    return (longitudes, latitudes)
