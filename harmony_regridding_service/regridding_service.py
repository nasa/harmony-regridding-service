"""Regridding service code."""

from logging import Logger, getLogger
from pathlib import Path

from harmony_service_lib.message import Message as HarmonyMessage
from harmony_service_lib.message import Source as HarmonySource
from harmony_service_lib.util import generate_output_filename
from netCDF4 import Dataset
from varinfo import VarInfoFromNetCDF4

from harmony_regridding_service.crs import (
    _add_grid_mapping_metadata,
    _write_grid_mappings,
)
from harmony_regridding_service.grid import _compute_target_area
from harmony_regridding_service.resample import (
    _cache_resamplers,
    _copy_resampled_dimension_variables,
    _resample_n_dimensional_variables,
    _resampled_dimension_pairs,
    _transfer_resampled_dimensions,
    _unresampled_variables,
)
from harmony_regridding_service.utilities import (
    _clone_variables,
    _transfer_metadata,
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
        _transfer_resampled_dimensions(source_ds, target_ds, target_area, var_info)
        crs_map = _write_grid_mappings(
            target_ds, _resampled_dimension_pairs(var_info), target_area
        )

        vars_to_process = var_info.get_all_variables()

        cloned_vars = _clone_variables(
            source_ds, target_ds, _unresampled_variables(var_info)
        )
        logger.info(f'cloned variables: {cloned_vars}')
        vars_to_process -= cloned_vars

        dimension_vars = _copy_resampled_dimension_variables(
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
