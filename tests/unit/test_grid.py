"""Tests the grid module."""

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from harmony_service_lib.message import Message as HarmonyMessage
from pyresample.geometry import AreaDefinition, SwathDefinition

from harmony_regridding_service.exceptions import (
    InvalidSourceDimensions,
    SourceDataError,
)
from harmony_regridding_service.grid import (
    _compute_area_extent_from_regular_x_y_coords,
    _compute_array_bounds,
    _compute_horizontal_source_grids,
    _compute_num_elements,
    _compute_projected_horizontal_source_grids,
    _compute_source_swath,
    _compute_target_area,
    _grid_height,
    _grid_width,
)


@patch('harmony_regridding_service.grid.AreaDefinition', wraps=AreaDefinition)
def test__compute_target_area(mock_area):
    """Ensure Area Definition correctly generated."""
    crs = '+datum=WGS84 +no_defs +proj=longlat +type=crs'
    xmin = -180
    xmax = 180
    ymin = -90
    ymax = 90

    message = HarmonyMessage(
        {
            'format': {
                'crs': crs,
                'scaleSize': {'x': 1.0, 'y': 2.0},
                'scaleExtent': {
                    'x': {'min': xmin, 'max': xmax},
                    'y': {'min': ymin, 'max': ymax},
                },
            }
        }
    )

    expected_height = 90
    expected_width = 360

    actual_area = _compute_target_area(message)

    assert actual_area.shape == (expected_height, expected_width)
    assert actual_area.area_extent == (xmin, ymin, xmax, ymax)
    assert actual_area.proj_str == crs
    mock_area.assert_called_once_with(
        'target_area_id',
        'target area definition',
        None,
        crs,
        expected_width,
        expected_height,
        (xmin, ymin, xmax, ymax),
    )


def test__grid_height_message_with_scale_size(test_message_with_scale_size):
    expected_grid_height = 50
    actual_grid_height = _grid_height(test_message_with_scale_size)
    assert expected_grid_height == actual_grid_height


def test__grid_height_mesage_includes_height(test_message_with_height_width):
    expected_grid_height = 80
    actual_grid_height = _grid_height(test_message_with_height_width)
    assert expected_grid_height == actual_grid_height


def test__grid_width_message_with_scale_size(test_message_with_scale_size):
    expected_grid_width = 100
    actual_grid_width = _grid_width(test_message_with_scale_size)
    assert expected_grid_width == actual_grid_width


def test__grid_width_message_with_width(test_message_with_height_width):
    expected_grid_width = 40
    actual_grid_width = _grid_width(test_message_with_height_width)
    assert expected_grid_width == actual_grid_width


def test__compute_num_elements():
    xmin = 0
    xmax = 1000
    ymin = 0
    ymax = 500

    message = HarmonyMessage(
        {
            'format': {
                'scaleSize': {'x': 10, 'y': 10},
                'scaleExtent': {
                    'x': {'min': xmin, 'max': xmax},
                    'y': {'min': ymin, 'max': ymax},
                },
            }
        }
    )

    expected_x_elements = 100
    expected_y_elements = 50
    actual_x_elements = _compute_num_elements(message, 'x')
    actual_y_elements = _compute_num_elements(message, 'y')

    assert expected_x_elements == actual_x_elements
    assert expected_y_elements == actual_y_elements


@patch('harmony_regridding_service.grid._compute_projected_horizontal_source_grids')
@patch('harmony_regridding_service.grid._compute_horizontal_source_grids')
@patch('harmony_regridding_service.grid._dims_are_projected_x_y')
@patch('harmony_regridding_service.grid._dims_are_lon_lat')
def test__compute_source_swath_lon_lat(
    mock_dims_are_lon_lat,
    mock_dims_are_projected_x_y,
    mock_compute_horizontal_source_grids,
    mock_compute_projected_horizontal_source_grids,
):
    """Test _compute_source_swath with longitude/latitude dimensions."""
    mock_dims_are_lon_lat.return_value = True
    mock_dims_are_projected_x_y.return_value = False

    mock_lons = np.array([[1, 2], [3, 4]])
    mock_lats = np.array([[5, 6], [7, 8]])

    mock_compute_horizontal_source_grids.return_value = (mock_lons, mock_lats)
    mock_compute_projected_horizontal_source_grids.return_value = (mock_lons, mock_lats)

    grid_dimensions = ('/longitude', '/latitude')
    filepath = 'fake_filepath.nc'
    var_info = MagicMock()
    variable_set = {'variable'}

    swath_def = _compute_source_swath(grid_dimensions, filepath, var_info, variable_set)

    mock_dims_are_lon_lat.assert_called_once_with(grid_dimensions, var_info)
    mock_compute_horizontal_source_grids.assert_called_once_with(
        grid_dimensions, filepath, var_info
    )

    mock_dims_are_projected_x_y.assert_not_called()
    mock_compute_projected_horizontal_source_grids.assert_not_called()

    assert isinstance(swath_def, SwathDefinition)
    np.testing.assert_array_equal(swath_def.lons, mock_lons)
    np.testing.assert_array_equal(swath_def.lats, mock_lats)


@patch('harmony_regridding_service.grid._compute_projected_horizontal_source_grids')
@patch('harmony_regridding_service.grid._compute_horizontal_source_grids')
@patch('harmony_regridding_service.grid._dims_are_projected_x_y')
@patch('harmony_regridding_service.grid._dims_are_lon_lat')
def test__compute_source_swath_projected_xy(
    mock_dims_are_lon_lat,
    mock_dims_are_projected_x_y,
    mock_compute_horizontal_source_grids,
    mock_compute_projected_horizontal_source_grids,
):
    """Test _compute_source_swath with projected x/y dimensions."""
    mock_dims_are_lon_lat.return_value = False
    mock_dims_are_projected_x_y.return_value = True

    mock_lons = np.array([[1, 2], [3, 4]])
    mock_lats = np.array([[5, 6], [7, 8]])
    mock_compute_horizontal_source_grids.return_value = (mock_lons, mock_lats)
    mock_compute_projected_horizontal_source_grids.return_value = (mock_lons, mock_lats)

    grid_dimensions = ('/y', '/x')
    filepath = 'fake_filepath.nc'
    var_info = MagicMock()
    variable_set = {'variable'}

    swath_def = _compute_source_swath(grid_dimensions, filepath, var_info, variable_set)

    mock_dims_are_lon_lat.assert_called_once_with(grid_dimensions, var_info)

    mock_dims_are_projected_x_y.assert_called_once_with(grid_dimensions, var_info)
    mock_compute_projected_horizontal_source_grids.assert_called_once_with(
        grid_dimensions, filepath, var_info, variable_set
    )

    mock_compute_horizontal_source_grids.assert_not_called()

    assert isinstance(swath_def, SwathDefinition)
    np.testing.assert_array_equal(swath_def.lons, mock_lons)
    np.testing.assert_array_equal(swath_def.lats, mock_lats)


@patch('harmony_regridding_service.grid._dims_are_projected_x_y')
@patch('harmony_regridding_service.grid._dims_are_lon_lat')
def test__compute_source_swath_invalid_dimensions(
    mock_dims_are_lon_lat, mock_dims_are_projected_x_y
):
    """Test _compute_source_swath with invalid dimensions."""
    mock_dims_are_lon_lat.return_value = False
    mock_dims_are_projected_x_y.return_value = False

    grid_dimensions = ('time', 'depth')
    filepath = 'fake_filepath.nc'
    var_info = MagicMock()
    variable_set = {'variable'}

    with pytest.raises(
        SourceDataError, match='Cannot determine correct dimension type from source'
    ):
        _compute_source_swath(grid_dimensions, filepath, var_info, variable_set)

    mock_dims_are_lon_lat.assert_called_once_with(grid_dimensions, var_info)
    mock_dims_are_projected_x_y.assert_called_once_with(grid_dimensions, var_info)


@pytest.mark.parametrize('test_arg', [('/lon', '/lat'), ('/lat', '/lon')])
def test__compute_horizontal_source_grids_expected_result(
    test_arg, test_1D_dimensions_ncfile, var_info_fxn
):
    """Exercises the single function for computing horizontal grids."""
    var_info = var_info_fxn(test_1D_dimensions_ncfile)

    expected_longitudes = np.array(
        [
            [-180, -80, -45, 45, 80, 180],
            [-180, -80, -45, 45, 80, 180],
            [-180, -80, -45, 45, 80, 180],
            [-180, -80, -45, 45, 80, 180],
            [-180, -80, -45, 45, 80, 180],
        ]
    )

    expected_latitudes = np.array(
        [
            [90, 90, 90, 90, 90, 90],
            [45, 45, 45, 45, 45, 45],
            [0, 0, 0, 0, 0, 0],
            [-46, -46, -46, -46, -46, -46],
            [-89, -89, -89, -89, -89, -89],
        ]
    )

    longitudes, latitudes = _compute_horizontal_source_grids(
        test_arg, test_1D_dimensions_ncfile, var_info
    )

    np.testing.assert_array_equal(expected_latitudes, latitudes)
    np.testing.assert_array_equal(expected_longitudes, longitudes)


@pytest.mark.parametrize('grid_dimensions', [('/y', '/x'), ('/x', '/y')])
def test__compute_projected_horizontal_source_grids(
    grid_dimensions, var_info_fxn, smap_projected_netcdf_file
):
    """Test source grid generation."""
    var_info = var_info_fxn(smap_projected_netcdf_file)

    expected_longitudes = np.array(
        [
            [-161.28112846, -161.18776746, -161.09440646, -161.00104546, -160.90768446],
            [-161.28112846, -161.18776746, -161.09440646, -161.00104546, -160.90768446],
            [-161.28112846, -161.18776746, -161.09440646, -161.00104546, -160.90768446],
            [-161.28112846, -161.18776746, -161.09440646, -161.00104546, -160.90768446],
            [-161.28112846, -161.18776746, -161.09440646, -161.00104546, -160.90768446],
            [-161.28112846, -161.18776746, -161.09440646, -161.00104546, -160.90768446],
        ]
    )
    expected_latitudes = np.array(
        [
            [58.95624444, 58.95624444, 58.95624444, 58.95624444, 58.95624444],
            [58.82092601, 58.82092601, 58.82092601, 58.82092601, 58.82092601],
            [58.6861299, 58.6861299, 58.6861299, 58.6861299, 58.6861299],
            [58.55184932, 58.55184932, 58.55184932, 58.55184932, 58.55184932],
            [58.41807764, 58.41807764, 58.41807764, 58.41807764, 58.41807764],
            [58.28480835, 58.28480835, 58.28480835, 58.28480835, 58.28480835],
        ]
    )

    longitudes, latitudes = _compute_projected_horizontal_source_grids(
        grid_dimensions,
        smap_projected_netcdf_file,
        var_info,
        set({'/Forecast_Data/sm_profile_forecast'}),
    )

    np.testing.assert_array_almost_equal(latitudes, expected_latitudes)
    np.testing.assert_array_almost_equal(longitudes, expected_longitudes)


def test__compute_horizontal_source_grids_2D_lat_lon_input(
    test_2D_dimensions_ncfile, var_info_fxn
):
    var_info = var_info_fxn(test_2D_dimensions_ncfile)
    grid_dimensions = ('/lat', '/lon')

    expected_regex = re.escape(
        'Incorrect source data dimensions. rows:(6, 5), columns:(6, 5)'
    )
    with pytest.raises(InvalidSourceDimensions, match=expected_regex):
        _compute_horizontal_source_grids(
            grid_dimensions, test_2D_dimensions_ncfile, var_info
        )


@pytest.mark.parametrize(
    'x_values, y_values, expected',
    [
        ([1, 2, 3], [-1, -2, -3], (0.5, -3.5, 3.5, -0.5)),
        ([-1, -2, -3], [-1, 0, 1, 2, 3], (-3.5, -1.5, -0.5, 3.5)),
        ([1, 2, 3, 4], [3, 2, 1, 0, -1], (0.5, -1.5, 4.5, 3.5)),
        ([9, 10], [2, 1, 0, -1, -2, -3], (8.5, -3.5, 10.5, 2.5)),
    ],
)
def test_compute_area_extent_from_regular_x_y_coords(x_values, y_values, expected):
    actual = _compute_area_extent_from_regular_x_y_coords(x_values, y_values)
    assert actual == expected


@pytest.mark.parametrize(
    'input_values, expected',
    [
        ([1, 2, 3], (0.5, 3.5)),
        ([-1, -2, -3], (-0.5, -3.5)),
        ([-1, 0, 1, 2, 3], (-1.5, 3.5)),
        ([3, 2, 1, 0, -1], (3.5, -1.5)),
        ([2, 1, 0, -1, -2, -3], (2.5, -3.5)),
    ],
)
def test__compute_array_bounds(input_values, expected):
    """Test expected cases."""
    actual = _compute_array_bounds(input_values)
    assert actual == expected


@pytest.mark.parametrize(
    'input_values, expected_error, expected_message',
    [
        (
            [10, 20, 30, 43.5, 50],
            SourceDataError,
            'coordinates are not regularly spaced',
        ),
        ([1], SourceDataError, 'coordinates must have at least 2 values'),
    ],
)
def test__compute_array_bounds_failures(input_values, expected_error, expected_message):
    """Test expected cases."""
    with pytest.raises(expected_error, match=expected_message):
        _compute_array_bounds(input_values)
