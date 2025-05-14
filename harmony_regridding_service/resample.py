"""Module for resampling functions."""

from logging import getLogger

import numpy as np
from netCDF4 import (
    Dataset,
)
from pyresample.ewa import DaskEWAResampler
from pyresample.geometry import AreaDefinition
from varinfo import VarInfoFromNetCDF4

from harmony_regridding_service.dimensions import (
    _all_dimensions,
    _copy_1d_dimension_variables,
    _copy_dimensions,
    _create_dimension,
    _get_column_dims,
    _get_row_dims,
    _horizontal_dims_for_variable,
    _is_column_dim,
    _is_row_dim,
)
from harmony_regridding_service.exceptions import (
    RegridderException,
)
from harmony_regridding_service.grid import _compute_source_swath
from harmony_regridding_service.utilities import (
    _copy_var_with_attrs,
    _copy_var_without_metadata,
    _integer_like,
)

logger = getLogger(__name__)


def _resample_variable_data(
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
            t_var[layer_index, ...] = _resample_variable_data(
                s_var[layer_index, ...],
                t_var[layer_index, ...],
                resampler,
                var_info,
                var_name,
                fill_value,
            )
        return t_var

    return _resample_layer(s_var[:], resampler, var_info, var_name, fill_value)


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

        # We have to get the fill value off of the variable here because we
        # only pass the numpy.ndarray into _resample_variable_data and we need
        # to know the fill value for the actual resampler.compute call.
        fill_value = getattr(t_var, '_FillValue', None)

        t_var[:] = _resample_variable_data(
            s_var[:], t_var[:], resampler, var_info, var_name, fill_value
        )
        logger.debug(f'Processed: {var_name}')

    return processed


def _resample_layer(
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
    prepped_source = _prepare_data_plane(
        source_plane, var_info, var_name, cast_to=cast_type
    )

    target_data = resampler.compute(
        prepped_source,
        fill_value=resample_fill,
        **_resampler_kwargs(prepped_source, source_plane.dtype),
    )

    # Convert the data back into the original datatype and transpose if necessary.
    return _prepare_data_plane(
        target_data, var_info, var_name, cast_to=source_plane.dtype
    )


def _resampler_kwargs(data: np.ndarray, original_dtype: np.dtype) -> dict:
    """Return kwargs to be used in resampling compute call.

    If an input data plane is like int, set maximum_weight_mode to true.
    """
    kwargs = {}

    kwargs['rows_per_scan'] = _get_rows_per_scan(data.shape[0])

    if _integer_like(original_dtype):
        kwargs['maximum_weight_mode'] = True

    return kwargs


def _copy_resampled_bounds_variable(
    source_ds: Dataset,
    target_ds: Dataset,
    bounds_var: str,
    target_area: AreaDefinition,
    var_info: VarInfoFromNetCDF4,
):
    """Copy computed values for dimension variable bounds variables."""
    var_dims = var_info.get_variable(bounds_var).dimensions

    xdims = _get_column_dims(var_dims, var_info)
    ydims = _get_row_dims(var_dims, var_info)
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


def _resampled_dimension_variable_names(var_info: VarInfoFromNetCDF4) -> set[str]:
    """Return the list of dimension variables to resample to target grid.

    This returns a list of the fully qualified variables that need to use the
    information from the target_area in order to compute the correct output
    values. However, this list will also include dimensions bounds variables
    if the collection contains them, which we want to return as well.

    """
    dims_to_transfer = set()
    resampled_dims = _resampled_dimensions(var_info)
    grouped_vars = var_info.group_variables_by_horizontal_dimensions()
    for dim in resampled_dims:
        dims_to_transfer.update(grouped_vars[(dim,)])

    return dims_to_transfer


def _create_resampled_dimensions(
    resampled_dim_pairs: list[tuple[str, str]],
    dataset: Dataset,
    target_area: AreaDefinition,
    var_info: VarInfoFromNetCDF4,
):
    """Create dimensions for the target resampled grids."""
    for dim_pair in resampled_dim_pairs:
        xdim = _get_column_dims(set(dim_pair), var_info)[0]
        ydim = _get_row_dims(set(dim_pair), var_info)[0]

        _create_dimension(dataset, xdim, target_area.projection_x_coords.shape[0])
        _create_dimension(dataset, ydim, target_area.projection_y_coords.shape[0])


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


def _resampled_dimension_pairs(var_info: VarInfoFromNetCDF4) -> list[tuple[str, str]]:
    """Return a list of the resampled horizontal spatial dimensions.

    Gives a list of the 2-element horizontal dimensions that are used in
    regridding this granule file.
    """
    return [
        dims
        for dims in var_info.group_variables_by_horizontal_dimensions()
        if len(dims) == 2
    ]


def _resampled_dimensions(var_info: VarInfoFromNetCDF4) -> set[str]:
    """Return a set of all resampled dimension names."""
    return {
        dim for dim_pair in _resampled_dimension_pairs(var_info) for dim in dim_pair
    }


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


def _transfer_resampled_dimensions(
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


def _copy_resampled_dimension_variables(
    source_ds: Dataset,
    target_ds: Dataset,
    target_area: AreaDefinition,
    var_info: VarInfoFromNetCDF4,
) -> set[str]:
    """Copy over dimension variables that are changed in the target file."""
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
