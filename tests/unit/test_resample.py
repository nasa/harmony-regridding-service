"""Tests the resample module."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from netCDF4 import Dataset
from numpy.testing import assert_array_equal

from harmony_regridding_service.resample import (
    copy_resampled_bounds_variable,
    copy_resampled_dimension_variables,
    create_resampled_dimensions,
    get_resampled_dimension_pairs,
    get_resampled_dimensions,
    get_rows_per_scan,
    needs_rotation,
    prepare_data_plane,
    resample_layer,
    resampled_dimension_variable_names,
    resampler_kwargs,
    transfer_resampled_dimensions,
    unresampled_variables,
)


def test_resample_layer_compute_float_explicit_fill(var_info_fxn, test_MERRA2_ncfile):
    """Test resampler.compute with float input and explicit fill value."""
    var_info = var_info_fxn(test_MERRA2_ncfile)
    source_plane = np.array(np.arange(12).reshape(4, 3), dtype=np.float32)
    resampler_mock = MagicMock()
    var_name = '/SLP'
    fill_value = np.float64(-9999.0)

    expected_source = source_plane.astype(np.float64)
    expected_rps = get_rows_per_scan(source_plane.shape[0])

    resample_layer(source_plane, resampler_mock, var_info, var_name, fill_value)

    call_args, call_kwargs = resampler_mock.compute.call_args
    actual_source = call_args[0]
    actual_fill_value = call_kwargs['fill_value']
    actual_rps = call_kwargs['rows_per_scan']

    np.testing.assert_array_equal(expected_source, actual_source)
    assert actual_fill_value == fill_value
    assert actual_rps == expected_rps
    assert 'maximum_weight_mode' not in call_kwargs  # Default for float


def test_resample_layer_compute_int_explicit_fill(var_info_fxn, test_MERRA2_ncfile):
    """Test resampler.compute with int input and explicit fill value."""
    var_info = var_info_fxn(test_MERRA2_ncfile)
    source_plane = np.array(np.arange(12).reshape(4, 3), dtype=np.int32)
    resampler_mock = MagicMock()
    var_name = '/PS'
    fill_value = np.int32(9999.0)

    expected_source = source_plane.astype(np.float64)
    expected_rps = get_rows_per_scan(source_plane.shape[0])

    resample_layer(source_plane, resampler_mock, var_info, var_name, fill_value)

    call_args, call_kwargs = resampler_mock.compute.call_args
    actual_source = call_args[0]
    actual_fill_value = call_kwargs['fill_value']
    actual_rps = call_kwargs['rows_per_scan']

    np.testing.assert_array_equal(expected_source, actual_source)
    assert actual_fill_value == fill_value
    assert actual_rps == expected_rps
    assert call_kwargs['maximum_weight_mode'] is True


def test_resample_layer_compute_float_no_fill(var_info_fxn, test_MERRA2_ncfile):
    """Test resampler.compute with float input and no explicit fill value."""
    var_info = var_info_fxn(test_MERRA2_ncfile)
    source_plane = np.array(np.arange(12).reshape(4, 3), dtype=np.float32)
    resampler_mock = MagicMock()
    resampler_mock._get_default_fill.return_value = -999.0
    var_name = '/QI'

    expected_source = source_plane.astype(np.float64)
    expected_rps = get_rows_per_scan(source_plane.shape[0])

    resample_layer(source_plane, resampler_mock, var_info, var_name, None)

    call_args, call_kwargs = resampler_mock.compute.call_args
    actual_source = call_args[0]
    actual_fill_value = call_kwargs['fill_value']
    actual_rps = call_kwargs['rows_per_scan']

    np.testing.assert_array_equal(expected_source, actual_source)
    assert actual_fill_value == -999.0
    assert actual_rps == expected_rps
    assert 'maximum_weight_mode' not in call_kwargs

    assert resampler_mock._get_default_fill.call_count == 1


def test_resampler_kwargs_floating_data():
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype='float')
    expected_args = {'rows_per_scan': 2}
    actual_args = resampler_kwargs(data, 'float')
    assert expected_args == actual_args


def test_resampler_kwargs_all_rows_needed():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype='float')
    expected_args = {'rows_per_scan': 7}
    actual_args = resampler_kwargs(data, 'float')
    assert expected_args == actual_args


def test_resampler_kwargs_integer_data():
    data = np.array([1, 2, 3], dtype='int16')
    expected_args = {
        'rows_per_scan': 3,
        'maximum_weight_mode': True,
    }
    actual_args = resampler_kwargs(data, 'int16')
    assert expected_args == actual_args


def test_copy_resampled_bounds_variable(
    test_file, test_area_fxn, test_IMERG_ncfile, var_info_fxn
):
    target_file = test_file
    target_area = test_area_fxn()
    var_info = var_info_fxn(test_IMERG_ncfile)
    bnds_var = '/Grid/lat_bnds'
    var_copied = None

    expected_lat_bnds = np.array(
        [
            target_area.projection_y_coords + 0.5,
            target_area.projection_y_coords - 0.5,
        ]
    ).T

    with (
        Dataset(test_IMERG_ncfile, mode='r') as source_ds,
        Dataset(target_file, mode='w') as target_ds,
    ):
        transfer_resampled_dimensions(source_ds, target_ds, target_area, var_info)

        var_copied = copy_resampled_bounds_variable(
            source_ds, target_ds, bnds_var, target_area, var_info
        )

    assert {bnds_var} == var_copied
    with Dataset(target_file, mode='r') as validate:
        assert_array_equal(expected_lat_bnds, validate['Grid']['lat_bnds'][:])


def test_resampled_dimension_variable_names_root_level_dimensions(
    test_1D_dimensions_ncfile, var_info_fxn
):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    expected_resampled = {'/lon', '/lat'}

    actual_resampled = resampled_dimension_variable_names(var_info)
    assert expected_resampled == actual_resampled


def test_resampled_dimension_variable_names_grouped_dimensions(
    test_IMERG_ncfile, var_info_fxn
):
    var_info = var_info_fxn(test_IMERG_ncfile)
    expected_resampled = {
        '/Grid/lon',
        '/Grid/lat',
        '/Grid/lon_bnds',
        '/Grid/lat_bnds',
    }

    actual_resampled = resampled_dimension_variable_names(var_info)
    assert expected_resampled == actual_resampled


def test_multiple_resampled_dimension_variable_names(test_ATL14_ncfile, var_info_fxn):
    var_info = var_info_fxn(test_ATL14_ncfile)
    expected_resampled = {'/x', '/y', '/tile_stats/x', '/tile_stats/y'}

    actual_resampled = resampled_dimension_variable_names(var_info)
    assert expected_resampled == actual_resampled


def test_create_resampled_dimensions_root_dimensions(
    var_info_fxn,
    test_1D_dimensions_ncfile,
    test_area_fxn,
    test_file,
):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    width = 36
    height = 18
    _generate_test_area = test_area_fxn(width=width, height=height)
    target_file = test_file

    with Dataset(target_file, mode='w') as target_ds:
        create_resampled_dimensions(
            [('/lat', '/lon')], target_ds, _generate_test_area, var_info
        )

    with Dataset(target_file, mode='r') as validate:
        assert validate.dimensions['lat'].size == 18
        assert validate.dimensions['lon'].size == 36


def test_create_resampled_dimensions_group_level_dimensions(
    var_info_fxn,
    test_IMERG_ncfile,
    test_area_fxn,
    test_file,
):
    var_info = var_info_fxn(test_IMERG_ncfile)
    _generate_test_area = test_area_fxn()
    target_file = test_file
    with Dataset(target_file, mode='w') as target_ds:
        create_resampled_dimensions(
            [('/Grid/lon', '/Grid/lat')],
            target_ds,
            _generate_test_area,
            var_info,
        )

    with Dataset(target_file, mode='r') as validate:
        assert validate['Grid'].dimensions['lat'].size == 180
        assert validate['Grid'].dimensions['lon'].size == 360


def test_unresampled_variables_flat_ungrouped(var_info_fxn, test_1D_dimensions_ncfile):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    expected_vars = {'/time', '/time_bnds'}
    actual_vars = unresampled_variables(var_info)
    assert expected_vars == actual_vars


def test_unresampled_variables_IMERG_grouped(var_info_fxn, test_IMERG_ncfile):
    var_info = var_info_fxn(test_IMERG_ncfile)

    expected_vars = {'/Grid/time', '/Grid/time_bnds'}
    actual_vars = unresampled_variables(var_info)
    assert expected_vars == actual_vars


def test_unresampled_variables_MERRA2_includes_levels(var_info_fxn, test_MERRA2_ncfile):
    var_info = var_info_fxn(test_MERRA2_ncfile)

    expected_vars = {'/lev', '/time'}
    actual_vars = unresampled_variables(var_info)
    assert expected_vars == actual_vars


def test_unresampled_variables_ATL14_lots_of_deep_group_vars(
    var_info_fxn, test_ATL14_ncfile
):
    var_info = var_info_fxn(test_ATL14_ncfile)

    expected_vars = {
        '/Polar_Stereographic',
        '/orbit_info/bounding_polygon_dim1',
        '/orbit_info/bounding_polygon_lat1',
        '/orbit_info/bounding_polygon_lon1',
        '/quality_assessment/qa_granule_fail_reason',
        '/quality_assessment/qa_granule_pass_fail',
    }
    actual_vars = unresampled_variables(var_info)
    assert expected_vars == actual_vars


def test_resampled_dimenension_pairs_1d_file(var_info_fxn, test_1D_dimensions_ncfile):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    expected_pairs = [('/lon', '/lat')]
    actual_pairs = get_resampled_dimension_pairs(var_info)
    assert expected_pairs == actual_pairs


def test_resampled_dimenension_pairs_multiple_horizontal_pairs(
    var_info_fxn, test_ATL14_ncfile
):
    var_info = var_info_fxn(test_ATL14_ncfile)
    expected_pairs = [('/y', '/x'), ('/tile_stats/y', '/tile_stats/x')]
    actual_pairs = get_resampled_dimension_pairs(var_info)
    assert set(expected_pairs) == set(actual_pairs)


def test_transfer_resampled_dimensions(
    test_file, test_area_fxn, var_info_fxn, test_1D_dimensions_ncfile
):
    """Tests transfer of all dimensions.

    test transfer of dimensions from source to target including resizing
    for the target's area definition.  The internal functions of
    transfer_resampled_dimensions are tested further down in this file.

    """
    width = 36
    height = 18
    _generate_test_area = test_area_fxn(width=width, height=height)
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    target_file = test_file
    with (
        Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds,
        Dataset(target_file, mode='w') as target_ds,
    ):
        transfer_resampled_dimensions(
            source_ds, target_ds, _generate_test_area, var_info
        )

    with Dataset(target_file, mode='r') as validate:
        assert validate.dimensions['bnds'].size == 2
        assert validate.dimensions['time'].size == 0
        assert validate.dimensions['time'].isunlimited() is True
        assert validate.dimensions['lon'].size == width
        assert validate.dimensions['lat'].size == height


def test_copy_resampled_dimension_variables(
    test_file,
    test_area_fxn,
    test_MERRA2_ncfile,
    var_info_fxn,
):
    target_file = test_file
    width = 300
    height = 150
    target_area = test_area_fxn(width=width, height=height)
    var_info = var_info_fxn(test_MERRA2_ncfile)
    expected_vars_copied = {'/lon', '/lat'}

    with (
        Dataset(test_MERRA2_ncfile, mode='r') as source_ds,
        Dataset(target_file, mode='w') as target_ds,
    ):
        transfer_resampled_dimensions(source_ds, target_ds, target_area, var_info)

        vars_copied = copy_resampled_dimension_variables(
            source_ds, target_ds, target_area, var_info
        )

        assert expected_vars_copied == vars_copied

    with Dataset(target_file, mode='r') as validate:
        assert validate.dimensions['lon'].size == width
        assert validate.dimensions['lat'].size == height
        assert validate.dimensions['lev'].size == 42


@pytest.mark.parametrize(
    'input_value, expected, description',
    [
        (1, 1, 'number less than 2'),
        (4, 2, 'even composite number'),
        (9, 3, 'odd composite number'),
        (7, 7, 'prime number'),
    ],
)
def test_get_rows_per_scan(input_value, expected, description):
    """Test get_rows_per_scan with various input types."""
    assert get_rows_per_scan(input_value) == expected, f'Failed for {description}'


def test_prepare_data_plane_floating_without_rotation(var_info_fxn, test_MERRA2_ncfile):
    var_info = var_info_fxn(test_MERRA2_ncfile)
    test_data = np.array(np.arange(12).reshape(4, 3), dtype=np.float32)
    var_name = '/T'
    expected_data = np.copy(test_data)
    actual_data = prepare_data_plane(test_data, var_info, var_name, cast_to=np.float64)

    assert np.float64 == actual_data.dtype
    np.testing.assert_equal(expected_data, actual_data)


def test_prepare_data_plane_floating_with_rotation(var_info_fxn, test_IMERG_ncfile):
    var_info = var_info_fxn(test_IMERG_ncfile)
    test_data = np.array(np.arange(12).reshape(4, 3), dtype=np.float16)
    var_name = '/Grid/HQprecipitation'
    expected_data = np.copy(test_data.T)
    actual_data = prepare_data_plane(test_data, var_info, var_name, cast_to=np.float64)

    assert np.float64 == actual_data.dtype
    np.testing.assert_equal(expected_data, actual_data)


def test_prepare_data_plane_int_without_rotation(var_info_fxn, test_MERRA2_ncfile):
    var_info = var_info_fxn(test_MERRA2_ncfile)
    test_data = np.array(np.arange(12).reshape(4, 3), dtype=np.int8)
    var_name = '/T'
    expected_data = np.copy(test_data)
    actual_data = prepare_data_plane(test_data, var_info, var_name, cast_to=np.float64)

    assert np.float64 == actual_data.dtype
    np.testing.assert_equal(expected_data, actual_data)


def test_prepare_data_plane_int_with_rotation(var_info_fxn, test_IMERG_ncfile):
    var_info = var_info_fxn(test_IMERG_ncfile)
    test_data = np.array(np.arange(12).reshape(4, 3), dtype=np.int64)
    test_data[0, 0] = -99999999
    var_name = '/Grid/HQprecipitation'
    expected_data = np.copy(test_data.T).astype(np.float64)

    actual_data = prepare_data_plane(test_data, var_info, var_name, cast_to=np.float64)

    assert np.float64 == actual_data.dtype
    np.testing.assert_equal(expected_data, actual_data)


def test_resampled_dimensions_1D_file(var_info_fxn, test_1D_dimensions_ncfile):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    expected_dimensions = {'/lat', '/lon'}
    actual_dimensions = get_resampled_dimensions(var_info)
    assert expected_dimensions == actual_dimensions


def test_resampled_dimensions_ATL14_multiple_grids(var_info_fxn, test_ATL14_ncfile):
    var_info = var_info_fxn(test_ATL14_ncfile)

    expected_dimensions = {'/x', '/y', '/tile_stats/x', '/tile_stats/y'}
    actual_dimensions = get_resampled_dimensions(var_info)
    assert expected_dimensions == actual_dimensions


def test_needs_rotation_needs_rotation(var_info_fxn, test_1D_dimensions_ncfile):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    assert needs_rotation(var_info, '/data') is True


def test_needs_rotation_no_rotation(var_info_fxn, test_MERRA2_ncfile):
    var_info = var_info_fxn(test_MERRA2_ncfile)
    assert needs_rotation(var_info, '/PHIS') is False
    assert needs_rotation(var_info, '/OMEGA') is False
