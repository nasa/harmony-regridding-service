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
)
from harmony_regridding_service.file_io import (
    clone_variables,
    transfer_metadata,
)
from harmony_regridding_service.grid import compute_target_area
from harmony_regridding_service.resample import (
    cache_resamplers,
    copy_resampled_dimension_variables,
    resample_n_dimensional_variables,
    transfer_resampled_dimensions,
    unresampled_variables,
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

    try:
        target_area = compute_target_area(message, input_filepath, var_info)
    except InvalidCRSResampling as e:
        logger.error(e)
        return input_filepath

    resampler_cache = cache_resamplers(input_filepath, var_info, target_area)

    target_filepath = generate_output_filename(input_filepath, is_regridded=True)

    with (
        Dataset(input_filepath, mode='r') as source_ds,
        Dataset(target_filepath, mode='w', format='NETCDF4') as target_ds,
    ):
        transfer_metadata(source_ds, target_ds)
        transfer_resampled_dimensions(source_ds, target_ds, target_area, var_info)
        crs_map = write_grid_mappings(
            target_ds, get_resampled_dimension_pairs(var_info), target_area
        )

        vars_to_process = var_info.get_all_variables()

        cloned_vars = clone_variables(
            source_ds, target_ds, unresampled_variables(var_info)
        )
        logger.info(f'cloned variables: {cloned_vars}')
        vars_to_process -= cloned_vars

        dimension_vars = copy_resampled_dimension_variables(
            source_ds, target_ds, target_area, var_info
        )
        logger.info(f'processed dimension variables: {dimension_vars}')
        vars_to_process -= dimension_vars

        resampled_vars = resample_n_dimensional_variables(
            source_ds, target_ds, var_info, resampler_cache, set(vars_to_process)
        )
        vars_to_process -= resampled_vars
        logger.info(f'resampled variables: {resampled_vars}')

        add_grid_mapping_metadata(target_ds, resampled_vars, var_info, crs_map)

        if vars_to_process:
            logger.warning(f'Unprocessed Variables: {vars_to_process}')
        else:
            logger.info('Processed all variables.')

    return target_filepath
