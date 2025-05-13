"""Regridding service code."""

from logging import Logger, getLogger
from pathlib import Path, PurePath

from harmony_service_lib.message import Message as HarmonyMessage
from harmony_service_lib.message import Source as HarmonySource
from harmony_service_lib.util import generate_output_filename
from netCDF4 import Dataset, Group
from pyresample.geometry import AreaDefinition
from varinfo import VarInfoFromNetCDF4

from harmony_regridding_service.dimensions import (
    _copy_1d_dimension_variables,
    _horizontal_dims_for_variable,
)
from harmony_regridding_service.grid import _compute_target_area
from harmony_regridding_service.resample import (
    _cache_resamplers,
    _copy_resampled_bounds_variable,
    _resample_n_dimensional_variables,
    _resampled_dimension_pairs,
    _resampled_dimension_variable_names,
    _transfer_dimensions,
    _unresampled_variables,
)
from harmony_regridding_service.variable_utilities import (
    _clone_variables,
    _get_variable,
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
