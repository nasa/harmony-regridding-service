"""Regridding service code."""

from logging import Logger, getLogger
from pathlib import Path

from harmony_service_lib.message import Message as HarmonyMessage
from harmony_service_lib.message import Source as HarmonySource
from harmony_service_lib.util import generate_output_filename
from netCDF4 import Dataset
from varinfo import VarInfoFromNetCDF4

from harmony_regridding_service.crs import (
    add_grid_mapping_metadata,
    write_grid_mappings,
)
from harmony_regridding_service.dimensions import (
    get_resampled_dimension_pairs,
)
from harmony_regridding_service.exceptions import (
    InvalidCRSResampling,
    InvalidVariableRequest,
)
from harmony_regridding_service.file_io import (
    clone_variables,
    filter_grid_mappings_to_variables,
    input_grid_mappings,
    transfer_metadata,
)
from harmony_regridding_service.grid import compute_target_areas
from harmony_regridding_service.resample import (
    cache_resamplers,
    copy_resampled_dimension_variables,
    resample_n_dimensional_variables,
    transfer_resampled_dimensions,
    unresampled_variables,
)
from harmony_regridding_service.var_utilitities import get_unprocessable_variables

logger = getLogger(__name__)


def varinfo_config_filename() -> str:
    """Return a path to the varinfo config."""
    return str(Path(Path(__file__).parent, 'config', 'HRS_varinfo_config.json'))


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

    user_requested_variables = {f'/{v.name.lstrip("/")}' for v in source.variables}

    var_info = VarInfoFromNetCDF4(
        input_filepath,
        short_name=source.shortName,  # pyright: ignore[reportAttributeAccessIssue]
        config_file=varinfo_config_filename(),
    )

    try:
        target_areas = compute_target_areas(message, input_filepath, var_info)
    except InvalidCRSResampling as e:
        logger.warning(f'{e}: Returning your input file unchanged.')
        return input_filepath

    resampler_cache = cache_resamplers(input_filepath, var_info, target_areas)

    target_filepath = generate_output_filename(input_filepath, is_regridded=True)

    with (
        Dataset(input_filepath, mode='r') as source_ds,
        Dataset(target_filepath, mode='w', format='NETCDF4') as target_ds,
    ):
        transfer_metadata(source_ds, target_ds)
        transfer_resampled_dimensions(source_ds, target_ds, target_areas, var_info)
        crs_map = write_grid_mappings(
            target_ds, get_resampled_dimension_pairs(var_info), target_areas
        )

        vars_to_process = var_info.get_all_variables()
        unresampled_vars = unresampled_variables(var_info)

        grid_mapping_variable_names = filter_grid_mappings_to_variables(
            input_grid_mappings(source_ds, vars_to_process)
        )

        if grid_mapping_variable_names:
            logger.info(f'dropping grid_mappings: {grid_mapping_variable_names}')
            vars_to_process -= grid_mapping_variable_names

        cloned_vars = clone_variables(
            source_ds, target_ds, unresampled_vars - grid_mapping_variable_names
        )
        logger.info(f'cloned variables: {cloned_vars}')
        vars_to_process -= cloned_vars

        dimension_vars = copy_resampled_dimension_variables(
            source_ds, target_ds, target_areas, var_info
        )
        logger.info(f'processed dimension variables: {dimension_vars}')
        vars_to_process -= dimension_vars

        unprocessable_variables = get_unprocessable_variables(var_info, vars_to_process)
        if unprocessable_variables:
            if unprocessable_variables & user_requested_variables:
                raise InvalidVariableRequest(
                    unprocessable_variables & user_requested_variables
                )
            logger.info(f'Dropping unprocessable variables: {unprocessable_variables}')
            vars_to_process -= unprocessable_variables

        resampled_vars = resample_n_dimensional_variables(
            source_ds, target_ds, var_info, resampler_cache, vars_to_process
        )
        vars_to_process -= resampled_vars
        logger.info(f'resampled variables: {resampled_vars}')

        add_grid_mapping_metadata(target_ds, resampled_vars, var_info, crs_map)

        if vars_to_process:
            logger.warning(f'Unprocessed Variables: {vars_to_process}')
        else:
            logger.info('Processed all variables.')

    return target_filepath
