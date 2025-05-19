"""Tests the dimensions module."""

import pytest
from netCDF4 import Dataset, Dimension
from numpy.testing import assert_array_equal

from harmony_regridding_service.dimensions import (
    copy_1d_dimension_variables,
    copy_dimension,
    copy_dimensions,
    create_dimension,
    get_all_dimensions,
    get_column_dims,
    get_dimension,
    get_row_dims,
    horizontal_dims_for_variable,
    is_column_dim,
    is_row_dim,
)
from harmony_regridding_service.resample import (
    transfer_resampled_dimensions,
)


def test_horizontal_dims_for_variable_grouped(test_IMERG_ncfile, var_info_fxn):
    var_info = var_info_fxn(test_IMERG_ncfile)
    expected_dims = ('/Grid/lon', '/Grid/lat')
    actual_dims = horizontal_dims_for_variable(var_info, '/Grid/IRkalmanFilterWeight')
    assert expected_dims == actual_dims


def test_horizontal_dims_for_variable(var_info_fxn, test_1D_dimensions_ncfile):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    expected_dims = ('/lon', '/lat')
    actual_dims = horizontal_dims_for_variable(var_info, '/data')
    assert expected_dims == actual_dims


def test_horizontal_dims_for_missing_variable(var_info_fxn, test_1D_dimensions_ncfile):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    expected_dims = None
    actual_dims = horizontal_dims_for_variable(var_info, '/missing')
    assert expected_dims == actual_dims


def test_is_column_dim_test_valid_x(test_1D_dimensions_ncfile, var_info_fxn):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    assert is_column_dim('/lon', var_info) is True


def test_is_column_dim_test_invalid_x(test_1D_dimensions_ncfile, var_info_fxn):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    assert is_column_dim('/lat', var_info) is False


def test_is_row_dim_test_valid_y(test_1D_dimensions_ncfile, var_info_fxn):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    assert is_row_dim('/lat', var_info) is True


def test_is_row_dim_test_invalid_y(test_1D_dimensions_ncfile, var_info_fxn):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    assert is_row_dim('/lon', var_info) is False


def test_get_column_dims_x_dims(test_1D_dimensions_ncfile, var_info_fxn):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    dims = ('/lat', '/lon')
    expected_dim = ['/lon']

    actual = get_column_dims(dims, var_info)
    assert expected_dim == actual


def test_get_row_dims_y_dims(test_1D_dimensions_ncfile, var_info_fxn):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    dims = ('/lat', '/lon')
    expected_dim = ['/lat']

    actual = get_row_dims(dims, var_info)
    assert expected_dim == actual


def test_get_row_dims_y_dims_no_variables(test_1D_dimensions_ncfile, var_info_fxn):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    dims = ('/baddim1', '/baddim2')

    expected_dims = []
    actual_dims = get_row_dims(dims, var_info)
    assert expected_dims == actual_dims


def test_get_column_dims_x_dims_no_variables(test_1D_dimensions_ncfile, var_info_fxn):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    dims = ('/baddim1', '/baddim2')
    expected_dims = []
    actual_dims = get_column_dims(dims, var_info)
    assert expected_dims == actual_dims


def test_get_column_dims_x_dims_with_bad_variable(
    test_1D_dimensions_ncfile, var_info_fxn
):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    dims = ('/baddim1', '/lon')
    expected_dim = ['/lon']

    actual_dim = get_column_dims(dims, var_info)
    assert expected_dim == actual_dim


def test__get_row_dims_y_dims_multiple_values(test_1D_dimensions_ncfile, var_info_fxn):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    dims = ('/lat', '/lon', '/lat', '/ba')
    expected_dim = ['/lat', '/lat']

    actual = get_row_dims(dims, var_info)
    assert expected_dim == actual


def test_create_dimension(test_file):
    name = '/somedim'
    size = 1000
    with Dataset(test_file, mode='w') as target_ds:
        dim = create_dimension(target_ds, name, size)
        assert isinstance(dim, Dimension)
        assert dim.size == size
        assert dim.name == 'somedim'


def test_create_nested_dimension(test_file):
    name = '/some/deeply/nested/dimname'
    size = 2000
    with Dataset(test_file, mode='w') as target_ds:
        dim = create_dimension(target_ds, name, size)
        assert isinstance(dim, Dimension)
        assert dim.size == size
        assert dim.name == 'dimname'


def test_all_dimensions(var_info_fxn, test_1D_dimensions_ncfile):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    expected_dimensions = {'/time', '/lon', '/lat', '/bnds'}
    actual_dimensions = get_all_dimensions(var_info)
    assert expected_dimensions == actual_dimensions


def test_copy_dimension(test_file, test_1D_dimensions_ncfile, longitudes):
    with (
        Dataset(test_file, mode='w') as target_ds,
        Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds,
    ):
        time_dimension = copy_dimension('/time', source_ds, target_ds)
        assert time_dimension.isunlimited() is True
        assert time_dimension.size == 0

        lon_dimension = copy_dimension('/lon', source_ds, target_ds)
        assert lon_dimension.isunlimited() is False
        assert lon_dimension.size == len(longitudes)


def test_copy_dimensions(test_file, test_1D_dimensions_ncfile, latitudes, longitudes):
    test_target = test_file
    with (
        Dataset(test_target, mode='w') as target_ds,
        Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds,
    ):
        copy_dimensions({'/lat', '/lon', '/time', '/bnds'}, source_ds, target_ds)

    with Dataset(test_target, mode='r') as validate:
        assert validate.dimensions['time'].isunlimited() is True
        assert validate.dimensions['time'].size == 0
        assert validate.dimensions['lat'].size == len(latitudes)
        assert validate.dimensions['lon'].size == len(longitudes)
        assert validate.dimensions['bnds'].size == 2


def test_copy_dimensions_with_groups(test_file, test_IMERG_ncfile):
    with (
        Dataset(test_file, mode='w') as target_ds,
        Dataset(test_IMERG_ncfile, mode='r') as source_ds,
    ):
        copy_dimensions(
            {'/Grid/latv', '/Grid/lonv', '/Grid/nv', '/Grid/time'},
            source_ds,
            target_ds,
        )

    with Dataset(test_file, mode='r') as validate:
        assert validate['Grid'].dimensions['time'].isunlimited() is True
        assert validate['Grid'].dimensions['time'].size == 0
        assert validate['Grid'].dimensions['lonv'].size == 2
        assert validate['Grid'].dimensions['latv'].size == 2
        assert validate['Grid'].dimensions['nv'].size == 2


def test_get_flat_dimension(test_1D_dimensions_ncfile, latitudes):
    with Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds:
        lat_dim = get_dimension(source_ds, '/lat')
        assert isinstance(lat_dim, Dimension)
        assert lat_dim.size == len(latitudes)
        assert lat_dim.name == 'lat'


def test_get_nested_dimension(test_IMERG_ncfile):
    with Dataset(test_IMERG_ncfile, mode='r') as source_ds:
        lat_dim = get_dimension(source_ds, '/Grid/lat')
        assert isinstance(lat_dim, Dimension)
        assert lat_dim.name == 'lat'
        assert lat_dim.size == 1800


def test_copy_1d_dimension_variables(
    test_file, test_area_fxn, var_info_fxn, test_1D_dimensions_ncfile
):
    target_file = test_file
    target_area = test_area_fxn()
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    dim_var_names = {'/lon', '/lat'}
    expected_attributes = {'long_name', 'standard_name', 'units'}
    vars_copied = []
    with (
        Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds,
        Dataset(target_file, mode='w') as target_ds,
    ):
        transfer_resampled_dimensions(source_ds, target_ds, target_area, var_info)
        vars_copied = copy_1d_dimension_variables(
            source_ds, target_ds, dim_var_names, target_area, var_info
        )

    assert dim_var_names == vars_copied
    with Dataset(target_file, mode='r') as validate:
        assert_array_equal(validate['/lon'][:], target_area.projection_x_coords)
        assert_array_equal(validate['/lat'][:], target_area.projection_y_coords)
        assert expected_attributes == set(validate['/lat'].ncattrs())
        with pytest.raises(AttributeError):
            validate['/lat'].getncattr('non-standard-attribute')
