"""Tests the crs module."""

from unittest.mock import MagicMock, call, patch

import pytest
import xarray as xr
from netCDF4 import Dataset
from pyproj import CRS

from harmony_regridding_service.crs import (
    _add_grid_mapping_metadata,
    _crs_from_source_data,
    _crs_variable_name,
    _is_geographic_crs,
    _write_grid_mappings,
)
from harmony_regridding_service.exceptions import (
    InvalidSourceCRS,
    InvalidTargetCRS,
)
from harmony_regridding_service.regridding_service import (
    _resampled_dimension_pairs,
    _transfer_metadata,
)
from harmony_regridding_service.resample import (
    _transfer_resampled_dimensions,
)
from harmony_regridding_service.utilities import _get_variable


@patch('harmony_regridding_service.crs._get_variable')
@patch('harmony_regridding_service.crs._horizontal_dims_for_variable')
def test_add_grid_mapping_metadata_sets_attributes(
    mock_horizontal_dims_for_variable,
    mock_get_variable,
):
    # Setup
    variables = {'var1', 'var2'}

    mock_varinfo = MagicMock()

    mock_dataset = MagicMock()
    mock_var1 = MagicMock()
    mock_var2 = MagicMock()
    mock_dataset['var1'] = mock_var1
    mock_dataset['var2'] = mock_var2

    def dims_side_effect(var_info, var_name):
        if var_name == 'var1':
            return ('/y', '/x')
        return ('/y2', '/x2')

    mock_horizontal_dims_for_variable.side_effect = dims_side_effect

    crs_map = {('/y', '/x'): 'crs_var1', ('/y2', '/x2'): 'crs_var2'}

    def _get_variable_side_effect(datset, var_name):
        return mock_var1 if var_name == 'var1' else mock_var2

    mock_get_variable.side_effect = _get_variable_side_effect

    _add_grid_mapping_metadata(mock_dataset, variables, mock_varinfo, crs_map)

    mock_horizontal_dims_for_variable.assert_has_calls(
        [call(mock_varinfo, 'var1'), call(mock_varinfo, 'var2')], any_order=True
    )

    mock_get_variable.assert_has_calls(
        [call(mock_dataset, 'var1'), call(mock_dataset, 'var2')], any_order=True
    )
    mock_var1.setncattr.assert_called_once_with('grid_mapping', 'crs_var1')
    mock_var2.setncattr.assert_called_once_with('grid_mapping', 'crs_var2')


def test__crs_from_source_data_expected_case(smap_projected_netcdf_file):
    dt = xr.open_datatree(smap_projected_netcdf_file)
    expected_crs = CRS('epsg:6933')
    crs = _crs_from_source_data(dt, set({'/Forecast_Data/sm_profile_forecast'}))
    assert crs.to_epsg() == expected_crs


def test__crs_from_source_data_missing(smap_projected_netcdf_file):
    dt = xr.open_datatree(smap_projected_netcdf_file)
    dt['/Forecast_Data/sm_profile_forecast'].attrs.pop('grid_mapping')
    with pytest.raises(InvalidSourceCRS, match='No grid_mapping metadata found'):
        _crs_from_source_data(dt, set({'/Forecast_Data/sm_profile_forecast'}))


def test__crs_from_source_data_bad(smap_projected_netcdf_file):
    dt = xr.open_datatree(smap_projected_netcdf_file)
    dt['EASE2_global_projection'].attrs['grid_mapping_name'] = 'nonsense projection'
    dt['/Forecast_Data/sm_profile_forecast'].attrs['grid_mapping']
    with pytest.raises(
        InvalidSourceCRS, match='Could not create a CRS from grid_mapping metadata'
    ):
        _crs_from_source_data(dt, set({'/Forecast_Data/sm_profile_forecast'}))


def test__crs_variable_name_multiple_grids_separate_groups():
    dim_pair = ('/Grid/lat', '/Grid/lon')
    dim_pairs = [
        ('/Grid/lat', '/Grid/lon'),
        ('/Grid2/lat', '/Grid2/lon'),
        ('/Grid3/lat', '/Grid3/lon'),
    ]

    expected_crs_name = '/Grid/crs'
    actual_crs_name = _crs_variable_name(dim_pair, dim_pairs)
    assert expected_crs_name == actual_crs_name


def test__crs_variable_name_single_grid():
    dim_pair = ('/lat', '/lon')
    dim_pairs = [('/lat', '/lon')]
    expected_crs_name = '/crs'

    actual_crs_name = _crs_variable_name(dim_pair, dim_pairs)
    assert expected_crs_name == actual_crs_name


def test__crs_variable_name_multiple_grids_share_group():
    dim_pair = ('/global_grid_lat', '/global_grid_lon')
    dim_pairs = [
        ('/npolar_grid_lat', '/npolar_grid_lon'),
        ('/global_grid_lat', '/global_grid_lon'),
        ('/spolar_grid_lat', '/spolar_grid_lon'),
    ]

    expected_crs_name = '/crs_global_grid_lat_global_grid_lon'
    actual_crs_name = _crs_variable_name(dim_pair, dim_pairs)
    assert expected_crs_name == actual_crs_name


@pytest.mark.parametrize(
    'message, expected, description',
    [
        ('EPSG:4326', True, 'EPSG:4326 is geographic'),
        ('+proj=longlat', True, '+proj=longlat is geographic'),
        ('4326', True, '4326 is geographic'),
        ('EPSG:6933', False, 'EPSG:6933 is non-geographic'),
        ('+proj=cea', False, '+proj=cea is non-geographic'),
    ],
)
def test_is_geographic_crs(message, expected, description):
    """Test _is_geographic_crs.

    Ensure function correctly determines if a supplied string resolves
    to a `pyproj.CRS` object with a geographic Coordinate Reference
    System (CRS). Exceptions arising from invalid CRS strings should
    also be handled.

    """
    assert _is_geographic_crs(message) == expected, f'Failed for {description}'


def test_is_geographic_raises_exception():
    """Test _is_geographic_crs when it throws an exception."""
    crs_string = 'invalid CRS'
    with pytest.raises(InvalidTargetCRS, match=crs_string):
        _is_geographic_crs(crs_string)


def test__write_grid_mappings(
    test_file,
    var_info_fxn,
    test_1D_dimensions_ncfile,
    test_area_fxn,
):
    target_file = test_file
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    _generate_test_area = test_area_fxn()
    expected_crs_map = {('/lon', '/lat'): '/crs'}

    with (
        Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds,
        Dataset(target_file, mode='w') as target_ds,
    ):
        _transfer_metadata(source_ds, target_ds)
        _transfer_resampled_dimensions(
            source_ds, target_ds, _generate_test_area, var_info
        )

        actual_crs_map = _write_grid_mappings(
            target_ds, _resampled_dimension_pairs(var_info), _generate_test_area
        )
        assert expected_crs_map == actual_crs_map

    with Dataset(target_file, mode='r') as validate:
        crs = _get_variable(validate, '/crs')
        expected_crs_metadata = _generate_test_area.crs.to_cf()

        actual_crs_metadata = {attr: crs.getncattr(attr) for attr in crs.ncattrs()}

        assert expected_crs_metadata == actual_crs_metadata
