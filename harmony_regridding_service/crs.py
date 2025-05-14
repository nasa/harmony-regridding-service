"""Module for handling Coordinate Reference Systems."""

from pathlib import PurePath

from netCDF4 import Dataset
from pyproj import CRS
from pyproj.exceptions import CRSError
from pyresample.geometry import AreaDefinition
from varinfo import VarInfoFromNetCDF4
from xarray import DataTree

from harmony_regridding_service.dimensions import (
    _horizontal_dims_for_variable,
)
from harmony_regridding_service.exceptions import (
    InvalidSourceCRS,
    InvalidTargetCRS,
)
from harmony_regridding_service.utilities import (
    _get_variable,
)


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


def _is_geographic_crs(crs_string: str) -> bool:
    """Infer if CRS is geographic.

    Use pyproj to ascertain if the supplied Coordinate Reference System
    (CRS) is geographic.
    """
    try:
        crs = CRS(crs_string)
        is_geographic = crs.is_geographic
    except CRSError as exception:
        raise InvalidTargetCRS(crs_string) from exception

    return is_geographic


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


def _add_grid_mapping_metadata(
    target_ds: Dataset, variables: set[str], var_info: VarInfoFromNetCDF4, crs_map: dict
) -> None:
    """Link regridded variables to the correct crs variable."""
    for var_name in variables:
        crs_variable_name = crs_map[_horizontal_dims_for_variable(var_info, var_name)]
        var = _get_variable(target_ds, var_name)
        var.setncattr('grid_mapping', crs_variable_name)
