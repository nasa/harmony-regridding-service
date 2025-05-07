"""Regridding service code."""

from collections.abc import Iterable
from logging import Logger, getLogger
from pathlib import Path, PurePath

import numpy as np
import xarray as xr
from harmony_service_lib.message import Message as HarmonyMessage
from harmony_service_lib.message import Source as HarmonySource
from harmony_service_lib.message_utility import has_dimensions
from harmony_service_lib.util import generate_output_filename
from netCDF4 import (
    Dataset,
    Dimension,
    Group,
    Variable,
)
from pyproj import CRS
from pyproj.exceptions import CRSError
from pyresample import create_area_def
from pyresample.ewa import DaskEWAResampler
from pyresample.geometry import AreaDefinition, SwathDefinition
from varinfo import VarInfoFromNetCDF4
from xarray import DataTree

from harmony_regridding_service.exceptions import (
    InvalidSourceCRS,
    InvalidSourceDimensions,
    RegridderException,
    SourceDataError,
)

logger = getLogger(__name__)

HRS_VARINFO_CONFIG_FILENAME = str(
    Path(Path(__file__).parent, 'config', 'HRS_varinfo_config.json')
)


def regrid(
    message: HarmonyMessage,
    input_filepath: str,
    source: HarmonySource,
    call_logger: Logger,
) -> str:
    """Regrid the input data at input_filepath."""
    global logger
    logger = call_logger or logger
    logger.info(f'Format:\n {message.format}')
    logger.info(f'Source:\n {source}')

    var_info = VarInfoFromNetCDF4(
        input_filepath,
        short_name=source.shortName,
        config_file=HRS_VARINFO_CONFIG_FILENAME,
    )

    target_area = _compute_target_area(message)

    resampler_cache = _cache_resamplers(input_filepath, var_info, target_area)

    target_filepath = generate_output_filename(input_filepath, is_regridded=True)

    with (
        Dataset(input_filepath, mode='r') as source_ds,
        Dataset(target_filepath, mode='w', format='NETCDF4') as target_ds,
    ):
        _transfer_metadata(source_ds, target_ds)
        _transfer_dimensions(source_ds, target_ds, target_area, var_info)
        crs_map = _write_grid_mappings(
            target_ds, _resampled_dimension_pairs(var_info), target_area
        )

        vars_to_process = var_info.get_all_variables()

        cloned_vars = _clone_variables(
            source_ds, target_ds, _unresampled_variables(var_info)
        )
        logger.info(f'cloned variables: {cloned_vars}')
        vars_to_process -= cloned_vars

        dimension_vars = _copy_dimension_variables(
            source_ds, target_ds, target_area, var_info
        )
        logger.info(f'processed dimension variables: {dimension_vars}')
        vars_to_process -= dimension_vars

        resampled_vars = _resample_n_dimensional_variables(
            source_ds, target_ds, var_info, resampler_cache, set(vars_to_process)
        )
        vars_to_process -= resampled_vars
        logger.info(f'resampled variables: {resampled_vars}')

        _add_grid_mapping_metadata(target_ds, resampled_vars, var_info, crs_map)

        if vars_to_process:
            logger.warning(f'Unprocessed Variables: {vars_to_process}')
        else:
            logger.info('Processed all variables.')

    return target_filepath


def _walk_groups(node: Dataset | Group) -> Group:
    """Traverse a netcdf file yielding each group."""
    yield node.groups.values()
    for value in node.groups.values():
        yield from _walk_groups(value)


def _transfer_metadata(source_ds: Dataset, target_ds: Dataset) -> None:
    """Transfer over global and group metadata to target file."""
    global_metadata = {}
    for attr in source_ds.ncattrs():
        global_metadata[attr] = source_ds.getncattr(attr)

    target_ds.setncatts(global_metadata)

    for groups in _walk_groups(source_ds):
        for group in groups:
            group_metadata = {}
            for attr in group.ncattrs():
                group_metadata[attr] = group.getncattr(attr)
            t_group = target_ds.createGroup(group.path)
            t_group.setncatts(group_metadata)


def _add_grid_mapping_metadata(
    target_ds: Dataset, variables: set[str], var_info: VarInfoFromNetCDF4, crs_map: dict
) -> None:
    """Link regridded variables to the correct crs variable."""
    for var_name in variables:
        crs_variable_name = crs_map[_horizontal_dims_for_variable(var_info, var_name)]
        var = _get_variable(target_ds, var_name)
        var.setncattr('grid_mapping', crs_variable_name)


def _resample_variable_data(
    s_var: np.ndarray,
    t_var: np.ndarray,
    resampler: DaskEWAResampler,
    var_info: VarInfoFromNetCDF4,
    var_name: str,
    eventual_fill_value: np.number | None,
) -> None:
    """Recursively resample variable data in N-dimensions.

    A recursive function that reduces an N-dimensional variable to the base
    case of a 2-D layer representing a horizontal spatial slice. This slice
    is resampled with the supplied DaskEWAResampler

    """
    if len(s_var.shape) > 2:
        for layer_index in range(s_var.shape[0]):
            t_var[layer_index, ...] = _resample_variable_data(
                s_var[layer_index, ...],
                t_var[layer_index, ...],
                resampler,
                var_info,
                var_name,
                eventual_fill_value,
            )
        return t_var

    return _resample_layer(s_var[:], resampler, var_info, var_name, eventual_fill_value)


def _resample_n_dimensional_variables(
    source_ds: Dataset,
    target_ds: Dataset,
    var_info: VarInfoFromNetCDF4,
    resampler_cache: dict,
    variables: set[str],
) -> set[str]:
    """Function to resample any projected variable."""
    processed = set()

    for var_name in variables:
        processed.add(var_name)
        logger.debug(f'resampling {var_name}')
        resampler = resampler_cache[_horizontal_dims_for_variable(var_info, var_name)]

        (s_var, t_var) = _copy_var_with_attrs(source_ds, target_ds, var_name)

        eventual_fill_value = getattr(t_var, '_FillValue', None)
        logger.debug(f'eventual_fill {eventual_fill_value}')

        t_var[:] = _resample_variable_data(
            s_var[:], t_var[:], resampler, var_info, var_name, eventual_fill_value
        )
        logger.debug(f'Processed: {var_name}')

    return processed


def _resample_layer(
    source_plane: np.array,
    resampler: DaskEWAResampler,
    var_info: VarInfoFromNetCDF4,
    var_name: str,
    eventual_fill_value: np.number | None,
) -> np.array:
    """Prepare the input layer, resample and return the results."""
    # pyresample only uses float64 so cast all data to it before resampling and
    # then back to your original data size.
    cast_type = np.float64

    if eventual_fill_value is not None:
        resample_fill = eventual_fill_value.astype(cast_type)
    else:
        ## Use pyresample's default fill value, but still convert to float64.
        resample_fill = resampler._get_default_fill(source_plane)  # pylint: disable=W0212
        resample_fill = np.array([resample_fill]).astype(cast_type)[0]

    # Cast input to float64 and transpose if necessary.
    prepped_source = _prepare_data_plane(
        source_plane, var_info, var_name, cast_to=cast_type
    )

    target_data = resampler.compute(
        prepped_source,
        fill_value=resample_fill,
        **_resampler_kwargs(prepped_source, source_plane.dtype),
    )

    ## Convert the data back into the original datatype and transpose if necessary.
    return _prepare_data_plane(
        target_data, var_info, var_name, cast_to=source_plane.dtype
    )


def _integer_like(test_type: np.dtype) -> bool:
    """Return True if the datatype is integer like."""
    return np.issubdtype(np.dtype(test_type), np.integer)


def _prepare_data_plane(
    data: np.Array,
    var_info: VarInfoFromNetCDF4,
    var_name: str,
    cast_to: np.dtype | None,
) -> np.Array:
    """Perform Type casting and transpose 2d data array when necessary.

    Also perform a transposition if the data dimension organization requires.
    """
    if cast_to is not None and data.dtype != cast_to:
        data = data.astype(cast_to)

    if _needs_rotation(var_info, var_name):
        data = np.ma.copy(data.T, order='C')

    return data


def _resampler_kwargs(data: np.nd.array, original_dtype: np.dtype) -> dict:
    """Return kwargs to be used in resampling compute call.

    If an input data plane is like int, set maximum_weight_mode to true.
    """
    kwargs = {}

    kwargs['rows_per_scan'] = _get_rows_per_scan(data.shape[0])

    if _integer_like(original_dtype):
        kwargs['maximum_weight_mode'] = True

    return kwargs


def _needs_rotation(var_info: VarInfoFromNetCDF4, variable: str) -> bool:
    """Check if variable must be rotated before resamling.

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
            if _is_horizontal_dim(dimension, var_info)
        ),
        None,
    )
    yloc = next(
        (
            index
            for index, dimension in enumerate(var_dims)
            if _is_vertical_dim(dimension, var_info)
        ),
        None,
    )
    if yloc > xloc:
        needs_rotation = True

    return needs_rotation


def _validate_remaining_variables(resampled_variables: dict) -> None:
    """Ensure every remaining variable can be processed.

    We should not have any 0D or 1D variables left and we do not handle greater
    than 4D variables in the service.
    """
    valid_dimensions = {2, 3, 4}
    variable_dimensions = set(resampled_variables.keys())
    extra_dimensions = variable_dimensions.difference(valid_dimensions)
    if len(extra_dimensions) != 0:
        raise RegridderException(
            f'Variables with dimensions {extra_dimensions} cannot be handled.'
        )


def _group_by_ndim(var_info: VarInfoFromNetCDF4, variables: set) -> dict:
    """Sort a list of variables by their number of dimensions.

    Return a dictionary of {num_dimensions : set(variable names)}
    """
    grouped_vars = {}
    for v in variables:
        var = var_info.get_variable(v)
        n_dim = len(var.dimensions)
        try:
            grouped_vars[n_dim].update({v})
        except KeyError:
            grouped_vars[n_dim] = {v}

    return grouped_vars


def _copy_resampled_bounds_variable(
    source_ds: Dataset,
    target_ds: Dataset,
    bounds_var: str,
    target_area: AreaDefinition,
    var_info: VarInfoFromNetCDF4,
):
    """Copy computed values for dimension variable bounds variables."""
    var_dims = var_info.get_variable(bounds_var).dimensions

    xdims = _get_horizontal_dims(var_dims, var_info)
    ydims = _get_vertical_dims(var_dims, var_info)
    if xdims:
        target_coords = target_area.projection_x_coords
        dim_name = xdims[0]
    else:
        target_coords = target_area.projection_y_coords
        dim_name = ydims[0]

    if not var_dims[0] == dim_name:
        raise RegridderException(f'_bnds var {var_dims} with unexpected shape')

    # create the bounds variable and fill it with the correct values.
    (_, t_var) = _copy_var_without_metadata(source_ds, target_ds, bounds_var)
    bounds_width = (target_coords[1] - target_coords[0]) / 2
    lower_bounds = target_coords - bounds_width
    upper_bounds = target_coords + bounds_width
    t_var[:, 0] = lower_bounds
    t_var[:, 1] = upper_bounds

    return {bounds_var}


def _copy_dimension_variables(
    source_ds: Dataset,
    target_ds: Dataset,
    target_area: AreaDefinition,
    var_info: VarInfoFromNetCDF4,
) -> set[str]:
    """Copy over dimension variables that are changed  in the target file."""
    dim_var_names = _resampled_dimension_variable_names(var_info)
    processed_vars = _copy_1d_dimension_variables(
        source_ds, target_ds, dim_var_names, target_area, var_info
    )

    bounds_vars = dim_var_names - processed_vars
    for bounds_var in bounds_vars:
        processed_vars |= _copy_resampled_bounds_variable(
            source_ds, target_ds, bounds_var, target_area, var_info
        )

    return processed_vars


def _copy_1d_dimension_variables(
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

    xdims = _get_horizontal_dims(one_d_vars, var_info)
    ydims = _get_vertical_dims(one_d_vars, var_info)

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

        (_, t_var) = _copy_var_without_metadata(source_ds, target_ds, dim_name)

        bounds_var = _get_bounds_var(var_info, dim_name)

        if bounds_var:
            standard_metadata['bounds'] = bounds_var
        t_var.setncatts(standard_metadata)

        t_var[:] = target_coords

    return one_d_vars


def _get_bounds_var(var_info: VarInfoFromNetCDF4, dim_name: str) -> str:
    return next(
        (
            var_info.get_variable(f'{dim_name}_{ext}').name
            for ext in ['bnds', 'bounds']
            if var_info.get_variable(f'{dim_name}_{ext}') is not None
        ),
        None,
    )


def _resampled_dimension_variable_names(var_info: VarInfoFromNetCDF4) -> set[str]:
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


def _transfer_dimensions(
    source_ds: Dataset,
    target_ds: Dataset,
    target_area: AreaDefinition,
    var_info: VarInfoFromNetCDF4,
) -> None:
    """Transfer all dimensions from source to target.

    Horizontal source dimensions that are changed due to resampling, are
    add onto the target using the information from the target_area.
    """
    all_dimensions = _all_dimensions(var_info)
    resampled_dimensions = _resampled_dimensions(var_info)
    resampled_dimension_pairs = _resampled_dimension_pairs(var_info)

    unchanged_dimensions = all_dimensions - resampled_dimensions
    _copy_dimensions(unchanged_dimensions, source_ds, target_ds)

    _create_resampled_dimensions(
        resampled_dimension_pairs, target_ds, target_area, var_info
    )


def _write_grid_mappings(
    target_ds: Dataset,
    resampled_dim_pairs: list[tuple[str, str]],
    target_area: AreaDefinition,
) -> dict:
    """Add coordinate reference system metadata variables.

    Add placeholder variables that contain the metadata related the coordinate
    reference system for the target grid.

    Returns a dictionary of horizonal tuple[dim pair] to full crs name for
    pointing back to the correct crs variable in the regridded variables.

    """
    crs_metadata = target_area.crs.to_cf()
    crs_map = {}

    for dim_pair in resampled_dim_pairs:
        crs_variable_name = _crs_variable_name(dim_pair, resampled_dim_pairs)
        var = PurePath(crs_variable_name)
        t_group = target_ds.createGroup(var.parent)
        t_var = t_group.createVariable(var.name, 'S1')
        t_var.setncatts(crs_metadata)
        crs_map[dim_pair] = crs_variable_name

    return crs_map


def _crs_variable_name(
    dim_pair: tuple[str, str], resampled_dim_pairs: list[tuple[str, str]]
) -> str:
    """Return a crs variable name for this dimension pair.

    This will be "/<netcdf group>/crs" unless there are multiple grids in the
    same group. if there are multiple grids will require additional information
    on the variable name.
    """
    dim = PurePath(dim_pair[0])
    dim_group = dim.parent
    crs_var_name = str(PurePath(dim_group, 'crs'))

    all_groups = set()
    all_groups.update([PurePath(d0).parent for (d0, d1) in resampled_dim_pairs])

    if len(all_groups) != len(resampled_dim_pairs):
        crs_var_name += f'_{PurePath(dim_pair[0]).name}_{PurePath(dim_pair[1]).name}'

    return crs_var_name


def _clone_variables(
    source_ds: Dataset, target_ds: Dataset, variables: set[str]
) -> set[str]:
    """Clone variables from source to target.

    Copy variables and their attributes directly from the source Dataset to the
    target Dataset.
    """
    for variable_name in variables:
        (s_var, t_var) = _copy_var_with_attrs(source_ds, target_ds, variable_name)
        try:
            t_var[:] = s_var[:]
        except IndexError as vlen_error:
            # Handle snowflake metadata with vlen string.
            if s_var.dtype == str and s_var.shape == ():
                t_var[0] = s_var[0]
            else:
                logger.error('Unable to clone variable {s_var}')
                raise SourceDataError('Unhandled variable clone') from vlen_error

    return variables


def _copy_var_with_attrs(
    source_ds: Dataset, target_ds: Dataset, variable_name: str
) -> (Variable, Variable):
    """Copy a source variable and metadata to target.

    Copy both the variable and metadata from a souce variable into a target,
    return both source and target variables.
    """
    s_var, t_var = _copy_var_without_metadata(source_ds, target_ds, variable_name)

    for att in s_var.ncattrs():
        if att != '_FillValue':
            t_var.setncattr(att, s_var.getncattr(att))

    return (s_var, t_var)


def _copy_var_without_metadata(
    source_ds: Dataset, target_ds: Dataset, variable_name: str
) -> (Variable, Variable):
    """Clones a single variable and returns both source and target variables.

    This function uses the netCDF4 createGroup('/[optionalgroup/andsubgroup]')
    call This will return an existing group, or create one that does not
    already exists. So this is not clobbering the source data.

    """
    var = PurePath(variable_name)
    s_var = _get_variable(source_ds, variable_name)

    # Create target variable
    t_group = target_ds.createGroup(var.parent)
    fill_value = getattr(s_var, '_FillValue', None)
    t_var = t_group.createVariable(
        var.name, s_var.dtype, s_var.dimensions, fill_value=fill_value
    )
    s_var.set_auto_maskandscale(False)
    t_var.set_auto_maskandscale(False)

    return (s_var, t_var)


def _get_variable(dataset: Dataset, variable_name: str) -> Variable:
    """Return a variable from a fully qualified variable name.

    This will return an existing or create a new variable.
    """
    var = PurePath(variable_name)
    group = dataset.createGroup(var.parent)
    return group[var.name]


def _create_dimension(dataset: Dataset, dimension_name: str, size: int) -> Dimension:
    """Create a fully qualified dimension on the dataset."""
    dim = PurePath(dimension_name)
    group = dataset.createGroup(dim.parent)
    return group.createDimension(dim.name, size)


def _create_resampled_dimensions(
    resampled_dim_pairs: list[tuple[str, str]],
    dataset: Dataset,
    target_area: AreaDefinition,
    var_info: VarInfoFromNetCDF4,
):
    """Create dimensions for the target resampled grids."""
    for dim_pair in resampled_dim_pairs:
        xdim = _get_horizontal_dims(set(dim_pair), var_info)[0]
        ydim = _get_vertical_dims(set(dim_pair), var_info)[0]

        _create_dimension(dataset, xdim, target_area.projection_x_coords.shape[0])
        _create_dimension(dataset, ydim, target_area.projection_y_coords.shape[0])


def _copy_dimension(dimension_name: str, source_ds: Dataset, target_ds: Dataset) -> str:
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


def _copy_dimensions(
    dimensions: set[str], source_ds: Dataset, target_ds: Dataset
) -> set[str]:
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


def _horizontal_dims_for_variable(
    var_info: VarInfoFromNetCDF4, var_name: str
) -> tuple[str, str]:
    """Return the horizontal dimensions for desired variable."""
    group_vars = var_info.group_variables_by_horizontal_dimensions()
    return next(
        (dims for dims, var_names in group_vars.items() if var_name in var_names),
        None,
    )


def _all_dimensions(var_info: VarInfoFromNetCDF4) -> set[str]:
    """Return a list of all dimensions in the file."""
    dimensions = set()
    for variable_name in var_info.get_all_variables():
        variable = var_info.get_variable(variable_name)
        for dim in variable.dimensions:
            dimensions.add(dim)

    return dimensions


def _unresampled_variables(var_info: VarInfoFromNetCDF4) -> set[str]:
    """Variable names to be transfered from source to target without change.

    returns a set of variables that do not have any dimension that is also in
    the set of resampled_dimensions.

    """
    vars_by_dim = var_info.group_variables_by_dimensions()
    resampled_dims = _resampled_dimensions(var_info)

    return set.union(
        *[
            variable_set
            for dimension_name, variable_set in vars_by_dim.items()
            if not resampled_dims.intersection(set(dimension_name))
        ]
    )


def _all_dimension_variables(var_info: VarInfoFromNetCDF4) -> set[str]:
    """Return a set of every dimension variable name in the file."""
    return var_info.get_required_dimensions(var_info.get_all_variables())


def _resampled_dimension_pairs(var_info: VarInfoFromNetCDF4) -> list[tuple[str, str]]:
    """Return a list of the resampled horizontal spatial dimensions.

    Gives a list of the 2-element horizontal dimensions that are used in
    regridding this granule file.

    """
    dimension_pairs = []
    for dims in var_info.group_variables_by_horizontal_dimensions():
        if len(dims) == 2:
            dimension_pairs.append(dims)
    return dimension_pairs


def _resampled_dimensions(var_info: VarInfoFromNetCDF4) -> set[str]:
    """Return a set of all resampled dimension names."""
    dimensions = set()
    for dim_pair in _resampled_dimension_pairs(var_info):
        dimensions.update(list(dim_pair))
    return dimensions


def _cache_resamplers(
    filepath: str, var_info: VarInfoFromNetCDF4, target_area: AreaDefinition
) -> None:
    """Precompute the resampling weights.

    Use the regridding target area in conjunction with each 2D horizontal
    dimension pair in the input source file to create an resampler and precompute
    the weights to be used when resampling.

    """
    grid_cache = {}

    dimension_vars_mapping = var_info.group_variables_by_horizontal_dimensions()

    for dimensions, variable_set in dimension_vars_mapping.items():
        # create swath definitions from each unique 2D grid dimensions found in
        # the input file.
        if len(dimensions) == 2:
            logger.debug(f'computing weights for dimensions {dimensions}')
            source_swath = _compute_source_swath(
                dimensions, filepath, var_info, variable_set
            )
            grid_cache[dimensions] = DaskEWAResampler(source_swath, target_area)
            grid_cache[dimensions].precompute(
                rows_per_scan=_get_rows_per_scan(source_swath.shape[0]),
            )

    return grid_cache


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


def _is_horizontal_dim(dim: str, var_info: VarInfoFromNetCDF4) -> str:
    """Test if dim is a horizontal dimension."""
    try:
        dim_var = var_info.get_variable(dim)
        is_x_dim = dim_var.is_longitude() or dim_var.is_projection_x()
    except AttributeError:
        is_x_dim = False
    return is_x_dim


def _is_vertical_dim(dim: str, var_info: VarInfoFromNetCDF4) -> str:
    """Test if dim is a projection Y dimension."""
    is_y_dim = False
    try:
        dim_var = var_info.get_variable(dim)
        is_y_dim = dim_var.is_latitude() or dim_var.is_projection_y()
    except AttributeError:
        pass
    return is_y_dim


def _get_horizontal_dims(
    dims: Iterable[str], var_info: VarInfoFromNetCDF4
) -> list[str]:
    """Return name for horizontal grid dimension [column/longitude/x]."""
    return [dim for dim in dims if _is_horizontal_dim(dim, var_info)]


def _get_vertical_dims(dims: Iterable[str], var_info: VarInfoFromNetCDF4) -> str:
    """Return name for vertical grid dimension [row/latitude/y]."""
    return [dim for dim in dims if _is_vertical_dim(dim, var_info)]


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


def _dims_are_projected_x_y(
    dimensions: tuple[str, str], var_info: VarInfoFromNetCDF4
) -> bool:
    """Does the dimension pair represent projected x/y values."""
    for dim_name in dimensions:
        dim_var = var_info.get_variable(dim_name)
        if not dim_var.is_projection_x_or_y():
            return False
    return True


def _dims_are_lon_lat(
    dimensions: tuple[str, str], var_info: VarInfoFromNetCDF4
) -> bool:
    """Does the dimension pair represent longitudes/latitudes."""
    for dim_name in dimensions:
        dim_var = var_info.get_variable(dim_name)
        if not dim_var.is_geographic():
            return False
    return True


def _compute_area_extent_from_regular_x_y_coords(
    xvalues: np.ndarray, yvalues: np.ndarray
) -> tuple(np.float64, np.float64, np.float64, np.float64):
    """Return outer extent of regularly defined grid.

    given xvalues and yvalues represent the center values of a regularly spaced array,
    compute the cell height and width, return the total extent of the grid area.

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


def _compute_array_bounds(values: np.ndarray) -> tuple(np.float64, np.float64):
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
    # SPL4CMDL (v7,v8) have +/1 1.5m variance in their cell
    # centers. relax spacing to allow for up to 3 meters difference in adjacent cells.
    if not np.allclose(diffs, diffs[0], atol=3):
        raise SourceDataError('coordinates are not regularly spaced')

    half_width = (values[1] - values[0]) / 2.0
    left = values[0] - half_width
    right = values[-1] + half_width
    return (left, right)


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
    xdim_name = _get_horizontal_dims(grid_dimensions, var_info)[0]
    ydim_name = _get_vertical_dims(grid_dimensions, var_info)[0]
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
    row_dim = _get_vertical_dims(grid_dimensions, var_info)[0]
    column_dim = _get_horizontal_dims(grid_dimensions, var_info)[0]
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
