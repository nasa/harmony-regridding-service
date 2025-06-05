"""Tests the crs module."""

from unittest.mock import MagicMock, call, patch

from netCDF4 import Dataset

from harmony_regridding_service.crs import (
    add_grid_mapping_metadata,
    get_crs_variable_name,
    write_grid_mappings,
)
from harmony_regridding_service.file_io import get_variable_from_dataset
from harmony_regridding_service.regridding_service import (
    get_resampled_dimension_pairs,
    transfer_metadata,
)
from harmony_regridding_service.resample import (
    transfer_resampled_dimensions,
)


@patch('harmony_regridding_service.crs.get_variable_from_dataset')
@patch('harmony_regridding_service.crs.horizontal_dims_for_variable')
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

    def get_variable_side_effect(datset, var_name):
        return mock_var1 if var_name == 'var1' else mock_var2

    mock_get_variable.side_effect = get_variable_side_effect

    add_grid_mapping_metadata(mock_dataset, variables, mock_varinfo, crs_map)

    mock_horizontal_dims_for_variable.assert_has_calls(
        [call(mock_varinfo, 'var1'), call(mock_varinfo, 'var2')], any_order=True
    )

    mock_get_variable.assert_has_calls(
        [call(mock_dataset, 'var1'), call(mock_dataset, 'var2')], any_order=True
    )
    mock_var1.setncattr.assert_called_once_with('grid_mapping', 'crs_var1')
    mock_var2.setncattr.assert_called_once_with('grid_mapping', 'crs_var2')


def test_crs_variable_name_multiple_grids_separate_groups():
    dim_pair = ('/Grid/lat', '/Grid/lon')
    dim_pairs = [
        ('/Grid/lat', '/Grid/lon'),
        ('/Grid2/lat', '/Grid2/lon'),
        ('/Grid3/lat', '/Grid3/lon'),
    ]

    expected_crs_name = '/Grid/crs'
    actual_crs_name = get_crs_variable_name(dim_pair, dim_pairs)
    assert expected_crs_name == actual_crs_name


def test_crs_variable_name_single_grid():
    dim_pair = ('/lat', '/lon')
    dim_pairs = [('/lat', '/lon')]
    expected_crs_name = '/crs'

    actual_crs_name = get_crs_variable_name(dim_pair, dim_pairs)
    assert expected_crs_name == actual_crs_name


def test_crs_variable_name_multiple_grids_share_group():
    dim_pair = ('/global_grid_lat', '/global_grid_lon')
    dim_pairs = [
        ('/npolar_grid_lat', '/npolar_grid_lon'),
        ('/global_grid_lat', '/global_grid_lon'),
        ('/spolar_grid_lat', '/spolar_grid_lon'),
    ]

    expected_crs_name = '/crs_global_grid_lat_global_grid_lon'
    actual_crs_name = get_crs_variable_name(dim_pair, dim_pairs)
    assert expected_crs_name == actual_crs_name


def test_write_grid_mappings(
    test_file,
    var_info_fxn,
    test_1D_dimensions_ncfile,
    test_area_fxn,
):
    target_file = test_file
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    _generate_test_area = test_area_fxn()
    test_areas = {('/lon', '/lat'): _generate_test_area}
    expected_crs_map = {('/lon', '/lat'): '/crs'}

    with (
        Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds,
        Dataset(target_file, mode='w') as target_ds,
    ):
        transfer_metadata(source_ds, target_ds)
        transfer_resampled_dimensions(source_ds, target_ds, test_areas, var_info)

        actual_crs_map = write_grid_mappings(
            target_ds, get_resampled_dimension_pairs(var_info), test_areas
        )
        assert expected_crs_map == actual_crs_map

    with Dataset(target_file, mode='r') as validate:
        crs = get_variable_from_dataset(validate, '/crs')
        expected_crs_metadata = _generate_test_area.crs.to_cf()

        actual_crs_metadata = {attr: crs.getncattr(attr) for attr in crs.ncattrs()}

        assert expected_crs_metadata == actual_crs_metadata
