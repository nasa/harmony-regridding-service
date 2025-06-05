"""Module for handling Coordinate Reference Systems."""

from pathlib import PurePath

from netCDF4 import Dataset
from pyresample.geometry import AreaDefinition
from varinfo import VarInfoFromNetCDF4

from harmony_regridding_service.dimensions import (
    horizontal_dims_for_variable,
)
from harmony_regridding_service.file_io import (
    get_variable_from_dataset,
)


def get_crs_variable_name(
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


def write_grid_mappings(
    target_ds: Dataset,
    resampled_dim_pairs: list[tuple[str, str]],
    target_area: AreaDefinition,
) -> dict:
    """Add coordinate reference system metadata variables.

    Add placeholder variables that contain the metadata related the coordinate
    reference system for the target grid.

    Returns a dictionary of horizontal tuple[dim pair] to full crs name for
    pointing back to the correct crs variable in the regridded variables.

    """
    crs_metadata = target_area.crs.to_cf()
    crs_map = {}

    for dim_pair in resampled_dim_pairs:
        crs_variable_name = get_crs_variable_name(dim_pair, resampled_dim_pairs)
        var = PurePath(crs_variable_name)
        t_group = target_ds.createGroup(var.parent)
        t_var = t_group.createVariable(var.name, 'S1')
        t_var.setncatts(crs_metadata)
        crs_map[dim_pair] = crs_variable_name

    return crs_map


def add_grid_mapping_metadata(
    target_ds: Dataset, variables: set[str], var_info: VarInfoFromNetCDF4, crs_map: dict
) -> None:
    """Link regridded variables to the correct crs variable."""
    for var_name in variables:
        crs_variable_name = crs_map[horizontal_dims_for_variable(var_info, var_name)]
        var = get_variable_from_dataset(target_ds, var_name)
        var.setncattr('grid_mapping', crs_variable_name)
