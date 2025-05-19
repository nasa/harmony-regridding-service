"""Tests the dimensions module."""

from harmony_regridding_service.dimensions import (
    get_column_dims,
    get_row_dims,
    horizontal_dims_for_variable,
    is_column_dim,
    is_row_dim,
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
