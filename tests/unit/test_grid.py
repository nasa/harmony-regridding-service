"""Tests the grid module."""

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from harmony_service_lib.message import Message as HarmonyMessage
from pyproj import CRS
from pyresample.geometry import AreaDefinition, SwathDefinition

from harmony_regridding_service.exceptions import (
    InvalidSourceCRS,
    InvalidSourceDimensions,
    SourceDataError,
)
from harmony_regridding_service.grid import (
    calculate_source_resolution,
    compute_area_extent_from_regular_x_y_coords,
    compute_array_bounds,
    compute_horizontal_source_grids,
    compute_num_elements,
    compute_projected_horizontal_source_grids,
    compute_source_swath,
    compute_target_area,
    create_grid_parameters_from_source,
    crs_from_source_data,
    dims_are_lon_lat,
    dims_are_projected_x_y,
    get_source_area_extent,
    get_target_grid_parameters,
    get_x_y_grid_values,
    grid_height,
    grid_width,
    transform_area_extent_to_crs,
)


@patch('harmony_regridding_service.grid.AreaDefinition', wraps=AreaDefinition)
@patch('harmony_regridding_service.grid.get_target_grid_parameters')
def test_compute_target_area(
    mock_get_target_grid_parameters, mock_area, test_2D_dimensions_ncfile, var_info_fxn
):
    """Ensure Area Definition correctly generated."""
    mock_area_extent = (-180, -90, 180, 90)
    mock_height = 90
    mock_width = 360
    mock_get_target_grid_parameters.return_value = (
        mock_area_extent,
        mock_height,
        mock_width,
    )

    crs = '+datum=WGS84 +no_defs +proj=longlat +type=crs'
    message = HarmonyMessage(
        {
            'format': {
                'crs': crs,
            }
        }
    )
    var_info = var_info_fxn(test_2D_dimensions_ncfile)

    actual_area = compute_target_area(message, test_2D_dimensions_ncfile, var_info)

    assert actual_area.shape == (mock_height, mock_width)
    assert actual_area.area_extent == mock_area_extent
    assert actual_area.proj_str == crs

    mock_area.assert_called_once_with(
        'target_area_id',
        'target area definition',
        None,
        crs,
        mock_width,
        mock_height,
        mock_area_extent,
    )
    mock_get_target_grid_parameters.assert_called_once_with(
        message,
        test_2D_dimensions_ncfile,
        var_info,
    )


@pytest.mark.parametrize(
    'scale_extent, scale_size, dimensions, expected_params',
    [
        ({}, {}, [1, 1], 'get params from message'),
        ({}, {}, [None, None], 'get params from message'),
        ({}, None, [1, 1], 'get params from message'),
        ({}, None, [None, None], 'create params'),
        (None, {}, [1, 1], 'create params'),
        (None, {}, [None, None], 'create params'),
        (None, None, [1, 1], 'create params'),
        (None, None, [None, None], 'create params'),
    ],
)
@patch('harmony_regridding_service.grid.get_grid_parameters_from_message')
@patch('harmony_regridding_service.grid.create_grid_parameters_from_source')
def test_get_target_grid_parameters_implicitly_or_explicitly(
    mock_create_grid_parameters_from_source,
    mock_get_grid_parameters_from_message,
    scale_extent,
    scale_size,
    dimensions,
    expected_params,
    test_2D_dimensions_ncfile,
    var_info_fxn,
):
    """Test get_target_grid_parameters.

    Grid parameters are either extracted from the message or implicitly
    created depending on which parameters are include in the requeset. This
    tests that the grid parameters are either extracted from the message or
    implicitly created for each possible combination of parameters included
    in the request.

    """
    crs = '+datum=WGS84 +no_defs +proj=longlat +type=crs'

    if scale_extent is not None:
        scale_extent = {
            'x': {'min': -180, 'max': 180},
            'y': {'min': -90, 'max': 90},
        }
    if scale_size is not None:
        scale_size = {'x': 1.0, 'y': 2.0}

    message = HarmonyMessage(
        {
            'format': {
                'crs': crs,
                'height': dimensions[0],
                'width': dimensions[1],
                'scaleSize': scale_size,
                'scaleExtent': scale_extent,
            }
        }
    )
    var_info = var_info_fxn(test_2D_dimensions_ncfile)

    get_target_grid_parameters(message, test_2D_dimensions_ncfile, var_info)

    if expected_params == 'get params from message':
        mock_get_grid_parameters_from_message.assert_called_once_with(message)
        mock_create_grid_parameters_from_source.assert_not_called()
    elif expected_params == 'create params':
        mock_create_grid_parameters_from_source.assert_called_once_with(
            test_2D_dimensions_ncfile, var_info, crs
        )
        mock_get_grid_parameters_from_message.assert_not_called()


def test_get_grid_parameters_from_message(var_info_fxn, test_2D_dimensions_ncfile):
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

    var_info = var_info_fxn(test_2D_dimensions_ncfile)

    expected_height = 90
    expected_width = 360

    actual_area_extent, actual_height, actual_width = get_target_grid_parameters(
        message, test_2D_dimensions_ncfile, var_info
    )

    assert actual_area_extent == (xmin, ymin, xmax, ymax)
    assert actual_height == expected_height
    assert actual_width == expected_width


def test_get_x_y_grid_values(smap_projected_netcdf_file, var_info_fxn):
    var_info = var_info_fxn(smap_projected_netcdf_file)

    expected_xvalues = np.array(
        [
            -15561416.159668,
            -15552408.104004,
            -15543400.04834,
            -15534391.992676,
            -15525383.937012,
        ]
    )
    expected_yvalues = np.array(
        [
            6283118.82568359,
            6274110.77001953,
            6265102.71435547,
            6256094.65869141,
            6247086.60302734,
            6238078.54736328,
        ]
    )

    actual_xvalues, actual_yvalues = get_x_y_grid_values(
        smap_projected_netcdf_file, var_info
    )

    np.testing.assert_array_almost_equal(actual_xvalues, expected_xvalues)
    np.testing.assert_array_almost_equal(actual_yvalues, expected_yvalues)


def test_get_x_y_grid_values_invalid_grid(tmp_path, var_info_fxn):
    expected_regex = re.escape('Cannot retrieve source grid.')
    invalid_grid_file = tmp_path / 'bad_grid_data.nc'

    x_coords = xr.DataArray(
        np.array(
            [
                1,
                2,
                3,
                4,
            ]
        ),
    )
    y_coords = xr.DataArray(np.array([5, 6, 7, 8]))
    bad_datatree = xr.DataTree(xr.Dataset(coords={'y': y_coords, 'x': x_coords}))
    bad_datatree.to_netcdf(invalid_grid_file)
    var_info = var_info_fxn(invalid_grid_file)

    with pytest.raises(SourceDataError, match=expected_regex):
        get_x_y_grid_values(invalid_grid_file, var_info)


@patch('harmony_regridding_service.grid.dims_are_projected_x_y')
@patch('harmony_regridding_service.grid.get_x_y_grid_values')
def test_calculate_source_resolution_geographic(
    mock_get_x_y_grid_values,
    mock_dims_are_projected_x_y,
    test_2D_dimensions_ncfile,
    var_info_fxn,
):
    mock_dims_are_projected_x_y.return_value = False
    mock_get_x_y_grid_values.return_value = (
        np.array([-10, -5, 0, 5, 10]),
        np.array([-20, -10, 0, 10, 20]),
    )

    var_info = var_info_fxn(test_2D_dimensions_ncfile)

    expected_x_res = 5
    expected_y_res = 10

    actual_x_res, actual_y_res = calculate_source_resolution(
        test_2D_dimensions_ncfile, var_info
    )

    mock_get_x_y_grid_values.assert_called_once_with(
        test_2D_dimensions_ncfile, var_info
    )
    mock_dims_are_projected_x_y.assert_called_once()

    assert expected_x_res == actual_x_res
    assert expected_y_res == actual_y_res


@patch('harmony_regridding_service.grid.dims_are_projected_x_y')
@patch('harmony_regridding_service.grid.get_x_y_grid_values')
def test_calculate_source_resolution_projection_gridded(
    mock_get_x_y_grid_values,
    mock_dims_are_projected_x_y,
    var_info_fxn,
    test_2D_dimensions_ncfile,
):
    mock_dims_are_projected_x_y.return_value = True
    mock_get_x_y_grid_values.return_value = (
        np.array([-1000000, -500000, 0, 500000, 1000000]),
        np.array([-2000000, -1000000, 0, 1000000, 2000000]),
    )

    var_info = var_info_fxn(test_2D_dimensions_ncfile)

    expected_x_res = 4.5
    expected_y_res = 9

    actual_x_res, actual_y_res = calculate_source_resolution(
        test_2D_dimensions_ncfile, var_info
    )

    mock_get_x_y_grid_values.assert_called_once_with(
        test_2D_dimensions_ncfile, var_info
    )
    mock_dims_are_projected_x_y.assert_called_once()

    # The conversion from meters to degrees uses the conversion factor
    # 111319.444444, so values must be rounded.
    assert expected_x_res == round(actual_x_res, 1)
    assert expected_y_res == round(actual_y_res, 1)


@patch('harmony_regridding_service.grid.transform_area_extent_to_crs')
@patch('harmony_regridding_service.grid.dims_are_projected_x_y')
@patch('harmony_regridding_service.grid.compute_area_extent_from_regular_x_y_coords')
def test_get_source_area_extent_geographic(
    mock_compute_area_extent_from_regular_x_y_coords,
    mock_dims_are_projected_x_y,
    mock_transform_area_extent_to_geographic,
    test_2D_dimensions_ncfile,
    var_info_fxn,
):
    mock_dims_are_projected_x_y.return_value = False
    expected_area_extent = (-10, -5, 10, 5)
    mock_compute_area_extent_from_regular_x_y_coords.return_value = expected_area_extent

    var_info = var_info_fxn(test_2D_dimensions_ncfile)
    crs = '+datum=WGS84 +no_defs +proj=longlat +type=crs'

    actual_area_extent = get_source_area_extent(
        test_2D_dimensions_ncfile, var_info, crs
    )

    mock_dims_are_projected_x_y.assert_called_once()
    mock_transform_area_extent_to_geographic.assert_not_called()

    assert actual_area_extent == expected_area_extent


@patch('harmony_regridding_service.grid.transform_area_extent_to_crs')
@patch('harmony_regridding_service.grid.dims_are_projected_x_y')
@patch('harmony_regridding_service.grid.compute_area_extent_from_regular_x_y_coords')
def test_get_source_area_extent_projection_gridded(
    mock_compute_area_extent_from_regular_x_y_coords,
    mock_dims_are_projected_x_y,
    mock_transform_area_extent_to_geographic,
    test_2D_dimensions_ncfile,
    var_info_fxn,
):
    area_extent = (-10000, -5000, 10000, 5000)
    transformed_area_extent = (-10, -5, 10, 5)

    mock_dims_are_projected_x_y.return_value = True
    mock_compute_area_extent_from_regular_x_y_coords.return_value = area_extent
    mock_transform_area_extent_to_geographic.return_value = transformed_area_extent

    var_info = var_info_fxn(test_2D_dimensions_ncfile)
    crs = '+datum=WGS84 +no_defs +proj=longlat +type=crs'

    actual_area_extent = get_source_area_extent(
        test_2D_dimensions_ncfile, var_info, crs
    )

    mock_dims_are_projected_x_y.assert_called_once()
    mock_transform_area_extent_to_geographic.assert_called_once_with(
        test_2D_dimensions_ncfile, var_info, area_extent, crs
    )

    assert actual_area_extent == transformed_area_extent


@pytest.mark.parametrize(
    'input_crs, output_crs, expected_area, description',
    [
        (
            'EPSG:6933',
            'EPSG:3413',
            (-8719079.050670, -8718511.282842, 8719079.050670, 8718511.282842),
            'projection-gridded to projection-gridded',
        ),
        (
            'EPSG:6933',
            'EPSG:4326',
            (-0.001866, -0.000705, 0.001866, 0.000705),
            'projection-gridded to geo',
        ),
        (
            'EPSG:4326',
            'EPSG:6933',
            (-17367530.445161, -7342230.136499, 17367530.445161, 7342230.136499),
            'geo to projection-gridded',
        ),
        (
            'EPSG:4326',
            '+datum=WGS84 +no_defs +proj=longlat +type=crs',
            (-180, -90, 180, 90),
            'geo to geo',
        ),
    ],
)
@patch('harmony_regridding_service.grid.crs_from_source_data')
def test_transform_area_extent_to_crs(
    mock_crs_from_source_data,
    input_crs,
    output_crs,
    expected_area,
    description,
    test_2D_dimensions_ncfile,
    var_info_fxn,
):
    input_crs = CRS(input_crs)
    output_crs = CRS(output_crs)
    mock_crs_from_source_data.return_value = input_crs
    var_info = var_info_fxn(test_2D_dimensions_ncfile)
    input_area_extent = (-180, -90, 180, 90)

    actual_area_extent = transform_area_extent_to_crs(
        test_2D_dimensions_ncfile, var_info, input_area_extent, output_crs
    )

    assert expected_area == pytest.approx(actual_area_extent, abs=0.000001), (
        f'Failed for {description}'
    )


@pytest.mark.parametrize(
    'x_res, y_res, expected_width, expected_height',
    [
        (2.5, 2.5, 144, 72),
        (2.5, 7, 144, 26),
        (7, 7, 51, 26),
    ],
)
@patch('harmony_regridding_service.grid.calculate_source_resolution')
@patch('harmony_regridding_service.grid.get_source_area_extent')
def test_create_grid_parameters_from_source(
    mock_get_source_area_extent,
    mock_calculate_source_resolution,
    x_res,
    y_res,
    expected_height,
    expected_width,
    test_2D_dimensions_ncfile,
    var_info_fxn,
):
    crs = 'EPSG:4326'
    var_info = var_info_fxn(test_2D_dimensions_ncfile)
    expected_area_extent = (-180, -90, 180, 90)
    mock_get_source_area_extent.return_value = expected_area_extent
    mock_calculate_source_resolution.return_value = (x_res, y_res)

    actual_area_extent, actual_height, actual_width = (
        create_grid_parameters_from_source(test_2D_dimensions_ncfile, var_info, crs)
    )

    assert expected_area_extent == actual_area_extent
    assert expected_height == actual_height
    assert expected_width == actual_width


def test_grid_height_message_with_scale_size(test_message_with_scale_size):
    expected_grid_height = 50
    actual_grid_height = grid_height(test_message_with_scale_size)
    assert expected_grid_height == actual_grid_height


def test_grid_height_mesage_includes_height(test_message_with_height_width):
    expected_grid_height = 80
    actual_grid_height = grid_height(test_message_with_height_width)
    assert expected_grid_height == actual_grid_height


def test_grid_width_message_with_scale_size(test_message_with_scale_size):
    expected_grid_width = 100
    actual_grid_width = grid_width(test_message_with_scale_size)
    assert expected_grid_width == actual_grid_width


def test_grid_width_message_with_width(test_message_with_height_width):
    expected_grid_width = 40
    actual_grid_width = grid_width(test_message_with_height_width)
    assert expected_grid_width == actual_grid_width


def test_compute_num_elements():
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
    actual_x_elements = compute_num_elements(message, 'x')
    actual_y_elements = compute_num_elements(message, 'y')

    assert expected_x_elements == actual_x_elements
    assert expected_y_elements == actual_y_elements


@patch('harmony_regridding_service.grid.compute_projected_horizontal_source_grids')
@patch('harmony_regridding_service.grid.compute_horizontal_source_grids')
@patch('harmony_regridding_service.grid.dims_are_projected_x_y')
@patch('harmony_regridding_service.grid.dims_are_lon_lat')
def test_compute_source_swath_lon_lat(
    mock_dims_are_lon_lat,
    mock_dims_are_projected_x_y,
    mock_compute_horizontal_source_grids,
    mock_compute_projected_horizontal_source_grids,
):
    """Test compute_source_swath with longitude/latitude dimensions."""
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

    swath_def = compute_source_swath(grid_dimensions, filepath, var_info, variable_set)

    mock_dims_are_lon_lat.assert_called_once_with(grid_dimensions, var_info)
    mock_compute_horizontal_source_grids.assert_called_once_with(
        grid_dimensions, filepath, var_info
    )

    mock_dims_are_projected_x_y.assert_not_called()
    mock_compute_projected_horizontal_source_grids.assert_not_called()

    assert isinstance(swath_def, SwathDefinition)
    np.testing.assert_array_equal(swath_def.lons, mock_lons)
    np.testing.assert_array_equal(swath_def.lats, mock_lats)


@patch('harmony_regridding_service.grid.compute_projected_horizontal_source_grids')
@patch('harmony_regridding_service.grid.compute_horizontal_source_grids')
@patch('harmony_regridding_service.grid.dims_are_projected_x_y')
@patch('harmony_regridding_service.grid.dims_are_lon_lat')
def test_compute_source_swath_projected_xy(
    mock_dims_are_lon_lat,
    mock_dims_are_projected_x_y,
    mock_compute_horizontal_source_grids,
    mock_compute_projected_horizontal_source_grids,
):
    """Test compute_source_swath with projected x/y dimensions."""
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

    swath_def = compute_source_swath(grid_dimensions, filepath, var_info, variable_set)

    mock_dims_are_lon_lat.assert_called_once_with(grid_dimensions, var_info)

    mock_dims_are_projected_x_y.assert_called_once_with(grid_dimensions, var_info)
    mock_compute_projected_horizontal_source_grids.assert_called_once_with(
        grid_dimensions, filepath, var_info, variable_set
    )

    mock_compute_horizontal_source_grids.assert_not_called()

    assert isinstance(swath_def, SwathDefinition)
    np.testing.assert_array_equal(swath_def.lons, mock_lons)
    np.testing.assert_array_equal(swath_def.lats, mock_lats)


@patch('harmony_regridding_service.grid.dims_are_projected_x_y')
@patch('harmony_regridding_service.grid.dims_are_lon_lat')
def test_compute_source_swath_invalid_dimensions(
    mock_dims_are_lon_lat, mock_dims_are_projected_x_y
):
    """Test compute_source_swath with invalid dimensions."""
    mock_dims_are_lon_lat.return_value = False
    mock_dims_are_projected_x_y.return_value = False

    grid_dimensions = ('time', 'depth')
    filepath = 'fake_filepath.nc'
    var_info = MagicMock()
    variable_set = {'variable'}

    with pytest.raises(
        SourceDataError, match='Cannot determine correct dimension type from source'
    ):
        compute_source_swath(grid_dimensions, filepath, var_info, variable_set)

    mock_dims_are_lon_lat.assert_called_once_with(grid_dimensions, var_info)
    mock_dims_are_projected_x_y.assert_called_once_with(grid_dimensions, var_info)


@pytest.mark.parametrize('test_arg', [('/lon', '/lat'), ('/lat', '/lon')])
def test_compute_horizontal_source_grids_expected_result(
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

    longitudes, latitudes = compute_horizontal_source_grids(
        test_arg, test_1D_dimensions_ncfile, var_info
    )

    np.testing.assert_array_equal(expected_latitudes, latitudes)
    np.testing.assert_array_equal(expected_longitudes, longitudes)


@pytest.mark.parametrize('grid_dimensions', [('/y', '/x'), ('/x', '/y')])
def test_compute_projected_horizontal_source_grids(
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

    longitudes, latitudes = compute_projected_horizontal_source_grids(
        grid_dimensions,
        smap_projected_netcdf_file,
        var_info,
        set({'/Forecast_Data/sm_profile_forecast'}),
    )

    np.testing.assert_array_almost_equal(latitudes, expected_latitudes)
    np.testing.assert_array_almost_equal(longitudes, expected_longitudes)


def test_compute_horizontal_source_grids_2D_lat_lon_input(
    test_2D_dimensions_ncfile, var_info_fxn
):
    var_info = var_info_fxn(test_2D_dimensions_ncfile)
    grid_dimensions = ('/lat', '/lon')

    expected_regex = re.escape(
        'Incorrect source data dimensions. rows:(6, 5), columns:(6, 5)'
    )
    with pytest.raises(InvalidSourceDimensions, match=expected_regex):
        compute_horizontal_source_grids(
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
    actual = compute_area_extent_from_regular_x_y_coords(x_values, y_values)
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
def test_compute_array_bounds(input_values, expected):
    """Test expected cases."""
    actual = compute_array_bounds(input_values)
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
def test_compute_array_bounds_failures(input_values, expected_error, expected_message):
    """Test expected cases."""
    with pytest.raises(expected_error, match=expected_message):
        compute_array_bounds(input_values)


def test_crs_from_source_data_expected_case(smap_projected_netcdf_file):
    dt = xr.open_datatree(smap_projected_netcdf_file)
    expected_crs = CRS('epsg:6933')
    crs = crs_from_source_data(dt, set({'/Forecast_Data/sm_profile_forecast'}))
    assert crs.to_epsg() == expected_crs


def test_crs_from_source_data_missing(smap_projected_netcdf_file):
    dt = xr.open_datatree(smap_projected_netcdf_file)
    dt['/Forecast_Data/sm_profile_forecast'].attrs.pop('grid_mapping')
    with pytest.raises(InvalidSourceCRS, match='No grid_mapping metadata found'):
        crs_from_source_data(dt, set({'/Forecast_Data/sm_profile_forecast'}))


def test_crs_from_source_data_bad(smap_projected_netcdf_file):
    dt = xr.open_datatree(smap_projected_netcdf_file)
    dt['EASE2_global_projection'].attrs['grid_mapping_name'] = 'nonsense projection'
    dt['/Forecast_Data/sm_profile_forecast'].attrs['grid_mapping']
    with pytest.raises(
        InvalidSourceCRS, match='Could not create a CRS from grid_mapping metadata'
    ):
        crs_from_source_data(dt, set({'/Forecast_Data/sm_profile_forecast'}))


@pytest.mark.parametrize(
    'file_fixture_name, dimensions, expected_result',
    [
        ('test_2D_dimensions_ncfile', ('/lon', '/lat'), True),
        ('smap_projected_netcdf_file', ('/y', '/x'), False),
    ],
)
def test_dims_are_lon_lat(
    var_info_fxn, request, file_fixture_name, dimensions, expected_result
):
    """Test if dimensions are lon/lat coordinates."""
    file_fixture = request.getfixturevalue(file_fixture_name)
    var_info = var_info_fxn(file_fixture)
    assert dims_are_lon_lat(dimensions, var_info) is expected_result


@pytest.mark.parametrize(
    'file_fixture_name, dimensions, expected_result',
    [
        ('test_2D_dimensions_ncfile', ('/lon', '/lat'), False),
        ('smap_projected_netcdf_file', ('/y', '/x'), True),
    ],
)
def test_dims_are_projected_x_y(
    var_info_fxn, request, file_fixture_name, dimensions, expected_result
):
    """Test if dimensions are projected x/y coordinates."""
    file_fixture = request.getfixturevalue(file_fixture_name)
    var_info = var_info_fxn(file_fixture)
    assert dims_are_projected_x_y(dimensions, var_info) is expected_result
