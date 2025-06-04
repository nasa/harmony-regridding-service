"""Module for resampling functions."""

from logging import getLogger
from pathlib import PurePath

import numpy as np
from netCDF4 import (
    Dataset,
    Dimension,
)
from pyresample.ewa import DaskEWAResampler
from pyresample.geometry import AreaDefinition, SwathDefinition
from varinfo import VarInfoFromNetCDF4

from harmony_regridding_service.dimensions import (
    get_column_dims,
    get_resampled_dimension_pairs,
    get_row_dims,
    horizontal_dims_for_variable,
    is_column_dim,
    is_row_dim,
)
from harmony_regridding_service.exceptions import (
    RegridderException,
    SourceDataError,
)
from harmony_regridding_service.file_io import (
    copy_var_with_attrs,
    copy_var_without_metadata,
)
from harmony_regridding_service.grid import (
    compute_horizontal_source_grids,
    compute_projected_horizontal_source_grids,
    dims_are_lon_lat,
    dims_are_projected_x_y,
)

logger = getLogger(__name__)


def resample_variable_data(
    s_var: np.ndarray,
    t_var: np.ndarray,
    resampler: DaskEWAResampler,
    var_info: VarInfoFromNetCDF4,
    var_name: str,
    fill_value: np.number | None,
) -> None:
    """Recursively resample variable data in N-dimensions.

    A recursive function that reduces an N-dimensional variable to the base
    case of a 2-D layer representing a horizontal spatial slice. This slice
    is resampled with the supplied DaskEWAResampler

    """
    if len(s_var.shape) > 2:
        for layer_index in range(s_var.shape[0]):
            t_var[layer_index, ...] = resample_variable_data(
                s_var[layer_index, ...],
                t_var[layer_index, ...],
                resampler,
                var_info,
                var_name,
                fill_value,
            )
        return t_var

    return resample_layer(s_var[:], resampler, var_info, var_name, fill_value)


def resample_n_dimensional_variables(
    source_ds: Dataset,
    target_ds: Dataset,
    var_info: VarInfoFromNetCDF4,
    resampler_cache: dict,
    variables: set[str],
) -> set[str]:
    """Function to resample any projected variable."""
    processed = set()

    for var_name in variables:
        logger.debug(f'resampling {var_name}')
        try:
            resampler = resampler_cache[
                horizontal_dims_for_variable(var_info, var_name)
            ]

            target_dimensions = get_preferred_ordered_dimension_names(
                var_info, var_name
            )

            (s_var, t_var) = copy_var_with_attrs(
                source_ds, target_ds, var_name, override_dimensions=target_dimensions
            )

            # We have to get the fill value off of the variable here because we
            # only pass the numpy.ndarray into _resample_variable_data and we need
            # to know the fill value for the actual resampler.compute call.
            fill_value = getattr(t_var, '_FillValue', None)

            source_variable = order_source_variable(s_var[:], var_info, var_name)

            t_var[:] = resample_variable_data(
                source_variable, t_var[:], resampler, var_info, var_name, fill_value
            )
            processed.add(var_name)
            logger.debug(f'Processed: {var_name}')
        except ValueError as ve:
            logger.info(f'Failed to process: {var_name}')
            logger.info(ve)
        except Exception as e:
            logger.error(f'Failed to process: {var_name}')
            logger.error(e)
            raise RegridderException(f'Failed to process: {var_name}') from e

    return processed


def resample_layer(
    source_plane: np.array,
    resampler: DaskEWAResampler,
    var_info: VarInfoFromNetCDF4,
    var_name: str,
    fill_value: np.number | None,
) -> np.array:
    """Prepare the input layer, resample and return the results."""
    # pyresample only uses float64 so cast all data before resampling and
    # then back to your original data size.
    cast_type = np.float64

    if fill_value is not None:
        resample_fill = fill_value.astype(cast_type)
    else:
        ## Use pyresample's default fill value, but still convert to float64.
        resample_fill = resampler._get_default_fill(source_plane)  # pylint: disable=W0212
        resample_fill = np.array([resample_fill]).astype(cast_type)[0]

    # Cast input to float64 and transpose if necessary.
    prepped_source = prepare_data_plane(
        source_plane, var_info, var_name, cast_to=cast_type
    )

    target_data = resampler.compute(
        prepped_source,
        fill_value=resample_fill,
        **resampler_kwargs(prepped_source, source_plane.dtype),
    )

    # Convert the data back into the original datatype and transpose if necessary.
    return prepare_data_plane(
        target_data, var_info, var_name, cast_to=source_plane.dtype
    )


def resampler_kwargs(data: np.ndarray, original_dtype: np.dtype) -> dict:
    """Return kwargs to be used in resampling compute call.

    If an input data plane is like int, set maximum_weight_mode to true.
    """
    kwargs = {}

    kwargs['rows_per_scan'] = get_rows_per_scan(data.shape[0])

    if integer_like(original_dtype):
        kwargs['maximum_weight_mode'] = True

    return kwargs


def integer_like(test_type: np.dtype) -> bool:
    """Return True if the datatype is integer like."""
    return np.issubdtype(np.dtype(test_type), np.integer)


def copy_resampled_bounds_variable(
    source_ds: Dataset,
    target_ds: Dataset,
    bounds_var: str,
    target_area: AreaDefinition,
    var_info: VarInfoFromNetCDF4,
):
    """Copy computed values for dimension variable bounds variables."""
    var_dims = var_info.get_variable(bounds_var).dimensions

    xdims = get_column_dims(var_dims, var_info)
    ydims = get_row_dims(var_dims, var_info)
    if xdims:
        target_coords = target_area.projection_x_coords
        dim_name = xdims[0]
    else:
        target_coords = target_area.projection_y_coords
        dim_name = ydims[0]

    if not var_dims[0] == dim_name:
        raise RegridderException(f'_bnds var {var_dims} with unexpected shape')

    # create the bounds variable and fill it with the correct values.
    (_, t_var) = copy_var_without_metadata(source_ds, target_ds, bounds_var)
    bounds_width = (target_coords[1] - target_coords[0]) / 2
    lower_bounds = target_coords - bounds_width
    upper_bounds = target_coords + bounds_width
    t_var[:, 0] = lower_bounds
    t_var[:, 1] = upper_bounds

    return {bounds_var}


def order_source_variable(
    source: np.ndarray, var_info: VarInfoFromNetCDF4, var_name: str
) -> np.ndarray:
    """Return the input source array with preferred ordered dimensions.

    For spatial regridding, we need to ensure that horizontal dimensions
    (latitude and longitude or x and y) are the last two dimensions in the
    array. This allows the regridding algorithm to properly operate on the
    spatial coordinates for n-dimensional variables.

    Parameters:
        source: 2D array to be regridded
        var_info: VarInfoFromNetCDF4 containing metadata about the file.
        var_name: name of the variable being processed

    Returns:
        Array with dimensions reordered if needed (horizontal dims last)

    Raises:
        RegridderException: If the variable has only one dimension

    """
    if len(source.shape) == 2:
        return source

    if len(source.shape) == 1:
        raise RegridderException('Attempted to resample a 1-D Variable.')

    correct_dims = get_fully_qualified_preferred_ordered_dimensions(var_info, var_name)

    if correct_dims == var_info.get_variable(var_name).dimensions:
        return source

    all_dims = var_info.get_variable(var_name).dimensions
    positions = [all_dims.index(dim) for dim in correct_dims]

    ordered_source = np.transpose(source, axes=positions)
    return ordered_source


def resampled_dimension_variable_names(var_info: VarInfoFromNetCDF4) -> set[str]:
    """Return the list of dimension variables to resample to target grid.

    This returns a list of the fully qualified variables that need to use the
    information from the target_area in order to compute the correct output
    values. However, this list will also include dimensions bounds variables
    if the collection contains them, which we want to return as well.

    """
    dims_to_transfer = set()
    resampled_dims = get_resampled_dimensions(var_info)
    grouped_vars = var_info.group_variables_by_horizontal_dimensions()
    for dim in resampled_dims:
        dims_to_transfer.update(grouped_vars[(dim,)])

    return dims_to_transfer


def create_resampled_dimensions(
    resampled_dim_pairs: list[tuple[str, str]],
    dataset: Dataset,
    target_area: AreaDefinition,
    var_info: VarInfoFromNetCDF4,
):
    """Create dimensions for the target resampled grids."""
    for dim_pair in resampled_dim_pairs:
        xdim = get_column_dims(set(dim_pair), var_info)[0]
        ydim = get_row_dims(set(dim_pair), var_info)[0]

        create_dimension(dataset, xdim, target_area.projection_x_coords.shape[0])
        create_dimension(dataset, ydim, target_area.projection_y_coords.shape[0])


def unresampled_variables(var_info: VarInfoFromNetCDF4) -> set[str]:
    """Variable names to be transfered from source to target without change.

    Returns a set of the variables that do not have any resampled dimension and
    are safe to copy directly over to the target file.

    """
    vars_by_dim = var_info.group_variables_by_dimensions()
    resampled_dims = get_resampled_dimensions(var_info)

    return set.union(
        *[
            variable_set
            for dimension_name_tuple, variable_set in vars_by_dim.items()
            if not resampled_dims.intersection(set(dimension_name_tuple))
        ]
    )


def get_resampled_dimensions(var_info: VarInfoFromNetCDF4) -> set[str]:
    """Return a set of all resampled dimension names."""
    return {
        dim for dim_pair in get_resampled_dimension_pairs(var_info) for dim in dim_pair
    }


def cache_resamplers(
    filepath: str, var_info: VarInfoFromNetCDF4, target_area: AreaDefinition
) -> None:
    """Precompute the resampling weights.

    Use the regridding target area in conjunction with each 2D horizontal
    dimension pair in the input source file to create an resampler and precompute
    the weights to be used when resampling.

    """
    grid_cache = {}

    dimension_vars_mapping = var_info.group_variables_by_horizontal_dimensions()

    for dimensions in dimension_vars_mapping:
        # create swath definitions from each unique 2D grid dimensions found in
        # the input file.
        if len(dimensions) == 2:
            logger.debug(f'computing weights for dimensions {dimensions}')
            source_swath = compute_source_swath(dimensions, filepath, var_info)
            grid_cache[dimensions] = DaskEWAResampler(source_swath, target_area)
            grid_cache[dimensions].precompute(
                rows_per_scan=get_rows_per_scan(source_swath.shape[0]),
            )

    return grid_cache


def compute_source_swath(
    grid_dimensions: tuple[str, str],
    filepath: str,
    var_info: VarInfoFromNetCDF4,
) -> SwathDefinition:
    """Return a SwathDefinition for the input grid_dimensions."""
    if dims_are_lon_lat(grid_dimensions, var_info):
        longitudes, latitudes = compute_horizontal_source_grids(
            grid_dimensions, filepath, var_info
        )
    elif dims_are_projected_x_y(grid_dimensions, var_info):
        longitudes, latitudes = compute_projected_horizontal_source_grids(
            grid_dimensions, filepath, var_info
        )
    else:
        raise SourceDataError(
            f'Cannot determine correct dimension type from source {grid_dimensions}.'
        )

    return SwathDefinition(lons=longitudes, lats=latitudes)


def transfer_resampled_dimensions(
    source_ds: Dataset,
    target_ds: Dataset,
    target_area: AreaDefinition,
    var_info: VarInfoFromNetCDF4,
) -> None:
    """Transfer all dimensions from source to target.

    Horizontal source dimensions that are changed due to resampling, are
    add onto the target using the information from the target_area.
    """
    all_dimensions = get_all_dimensions(var_info)
    resampled_dimensions = get_resampled_dimensions(var_info)
    resampled_dimension_pairs = get_resampled_dimension_pairs(var_info)

    unchanged_dimensions = all_dimensions - resampled_dimensions
    copy_dimensions(unchanged_dimensions, source_ds, target_ds)

    create_resampled_dimensions(
        resampled_dimension_pairs, target_ds, target_area, var_info
    )


def copy_resampled_dimension_variables(
    source_ds: Dataset,
    target_ds: Dataset,
    target_area: AreaDefinition,
    var_info: VarInfoFromNetCDF4,
) -> set[str]:
    """Copy over dimension variables that are changed in the target file."""
    dim_var_names = resampled_dimension_variable_names(var_info)
    processed_vars = copy_1d_dimension_variables(
        source_ds, target_ds, dim_var_names, target_area, var_info
    )

    bounds_vars = dim_var_names - processed_vars
    for bounds_var in bounds_vars:
        processed_vars |= copy_resampled_bounds_variable(
            source_ds, target_ds, bounds_var, target_area, var_info
        )

    return processed_vars


def get_rows_per_scan(total_rows: int) -> int:
    """Gets optimum value for rows per scan.

    Finds the smallest divisor of the total number of rows. If no divisor is
    found, return the total number of rows.

    """
    if total_rows < 2:
        return 1
    for row_number in range(2, int(total_rows**0.5) + 1):
        if total_rows % row_number == 0:
            return row_number

    logger.info(f'returning all rows for rows_per_scan = {total_rows}')
    return total_rows


def prepare_data_plane(
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

    if needs_rotation(var_info, var_name):
        data = np.ma.copy(data.T, order='C')

    return data


def needs_rotation(var_info: VarInfoFromNetCDF4, variable: str) -> bool:
    """Check if variable must be rotated before resampling.

    pyresample's EWA assumes swath input which implies the x projection
    dimension must be the fastest varying dimension.

    So if the lon comes before lat in the variables dimensions you must rotate
    the grid before resampling.

    """
    variable_needs_rotation = False
    var_dims = var_info.get_variable(variable).dimensions
    column_loc = next(
        (
            index
            for index, dimension in enumerate(var_dims)
            if is_column_dim(dimension, var_info)
        ),
        None,
    )
    row_loc = next(
        (
            index
            for index, dimension in enumerate(var_dims)
            if is_row_dim(dimension, var_info)
        ),
        None,
    )
    if row_loc > column_loc:
        logger.info(f'Incorrect dimension order on {variable}, needs rotation.')
        variable_needs_rotation = True

    return variable_needs_rotation


def copy_dimensions(
    dimensions: set[str], source_ds: Dataset, target_ds: Dataset
) -> set[str]:
    """Copy each dimension from source to target.

    ensure the first dimensions copied are the UNLIMITED dimensions.
    """

    def sort_unlimited_first(dimension_name):
        """Sort dimensions so that unlimited are first in list."""
        the_dim = get_dimension(source_ds, dimension_name)
        return not the_dim.isunlimited()

    sorted_dims = sorted(list(dimensions), key=sort_unlimited_first)

    for dim in sorted_dims:
        copy_dimension(dim, source_ds, target_ds)


def copy_dimension(dimension_name: str, source_ds: Dataset, target_ds: Dataset) -> str:
    """Copy dimension from source to target file."""
    source_dimension = get_dimension(source_ds, dimension_name)

    source_size = None
    if not source_dimension.isunlimited():
        source_size = source_dimension.size

    dim = PurePath(dimension_name)
    target_group = target_ds.createGroup(dim.parent)
    return target_group.createDimension(dim.name, source_size)


def get_dimension(dataset: Dataset, dimension_name: str) -> Dimension:
    """Return a dimension object for a dimension name.

    Return the Dimension for an arbitrarily nested dimension name.
    """
    dim = PurePath(dimension_name)
    return dataset.createGroup(dim.parent).dimensions[dim.name]


def create_dimension(dataset: Dataset, dimension_name: str, size: int) -> Dimension:
    """Create a fully qualified dimension on the dataset."""
    dim = PurePath(dimension_name)
    group = dataset.createGroup(dim.parent)
    return group.createDimension(dim.name, size)


def get_all_dimensions(var_info: VarInfoFromNetCDF4) -> set[str]:
    """Return a list of all dimensions in the file."""
    dimensions = set()
    for variable_name in var_info.get_all_variables():
        variable = var_info.get_variable(variable_name)
        for dim in variable.dimensions:
            dimensions.add(dim)

    return dimensions


def copy_1d_dimension_variables(
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

    xdims = get_column_dims(one_d_vars, var_info)
    ydims = get_row_dims(one_d_vars, var_info)

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

        (_, t_var) = copy_var_without_metadata(source_ds, target_ds, dim_name)

        bounds_var = get_bounds_var(var_info, dim_name)

        if bounds_var:
            standard_metadata['bounds'] = bounds_var
        t_var.setncatts(standard_metadata)

        t_var[:] = target_coords

    return one_d_vars


def get_bounds_var(var_info: VarInfoFromNetCDF4, dim_name: str) -> str:
    """Return the bounds variable associated with the given dimension."""
    return next(
        (
            var_info.get_variable(f'{dim_name}_{ext}').name
            for ext in ['bnds', 'bounds']
            if var_info.get_variable(f'{dim_name}_{ext}') is not None
        ),
        None,
    )


def get_preferred_ordered_dimension_names(
    var_info: VarInfoFromNetCDF4, var_name: str
) -> tuple[str]:
    """Return the base names of the full dimensions.

    Used to create a target variable.
    """
    existing_dims = var_info.get_variable(var_name).dimensions

    full_path_dims = get_fully_qualified_preferred_ordered_dimensions(
        var_info, var_name
    )

    if full_path_dims != existing_dims:
        # return the newly ordered dims only if they changed.
        return tuple(PurePath(dim).name for dim in full_path_dims)

    return None


def get_fully_qualified_preferred_ordered_dimensions(
    var_info: VarInfoFromNetCDF4, var_name: str
) -> list | None:
    """Return the preferred order of the dimensions for the variable.

    This will take a variable's list of dimensions and shuffle them so that the
    horizontal dimensions shift to the end of the dimension list.

    ['/y', '/height', '/x', '/season'] -> ['/height', '/season','/y', '/x']

    This function is used when creating a variable, and returns None when the
    order does not change.

    This is a step towards CF Conventions but does *not* re-order the
    horizontal dimenions. If the input dims are ['/x', '/'y] they remain that
    in the output this may be changed in the future with DAS-2374.

    This also makes no attempt order time or vertical dimensions per CF
    Conventions.

    """
    all_dims = var_info.get_variable(var_name).dimensions

    if len(all_dims) <= 2:
        return all_dims

    horizontal_dims = horizontal_dims_for_variable(var_info, var_name)

    if all_dims[-1] in horizontal_dims and all_dims[-2] in horizontal_dims:
        # Already in correct order
        return all_dims

    non_horizontal_dims = [dim for dim in all_dims if dim not in horizontal_dims]

    return [*non_horizontal_dims, *horizontal_dims]
