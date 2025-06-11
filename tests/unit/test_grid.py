"""Tests the grid module."""

import re
from unittest.mock import Mock, patch

import numpy as np
import pytest
from harmony_service_lib.message import Message as HarmonyMessage
from pyproj import CRS
from pyresample import create_area_def
from pyresample.geometry import AreaDefinition
from pytest import approx

from harmony_regridding_service.dimensions import GridDimensionPair
from harmony_regridding_service.exceptions import (
    InvalidSourceDimensions,
    SourceDataError,
)
from harmony_regridding_service.grid import (
    compute_area_extent_from_regular_x_y_coords,
    compute_array_bounds,
    compute_horizontal_source_grids,
    compute_num_elements,
    compute_projected_horizontal_source_grids,
    compute_target_areas,
    convert_projected_area_to_geographic,
    create_area_definition_for_projected_source_grid,
    create_target_areas_from_source,
    dims_are_lon_lat,
    dims_are_projected_x_y,
    get_area_definition_from_message,
    get_geographic_area_extent,
    get_geographic_resolution,
    grid_height,
    grid_width,
    reorder_extents,
)


@patch(
    'harmony_regridding_service.grid.create_target_areas_from_source',
    wraps=create_target_areas_from_source,
)
@patch(
    'harmony_regridding_service.grid.get_area_definition_from_message',
    wraps=get_area_definition_from_message,
)
def test_compute_target_areas_with_parameters(
    mock_get_area_definition_from_message,
    mock_create_target_areas_from_source,
    smap_projected_netcdf_file,
    var_info_fxn,
):
    """Ensure Area Definition correctly generated from message parameters.

    This test defines a valid grid in the Harmony Message, and verifies that
    the resulting target area from compute_target_areas is pulled from the
    message not computed from the source.

    """
    var_info = var_info_fxn(smap_projected_netcdf_file)
    crs = '+datum=WGS84 +no_defs +proj=longlat +type=crs'
    xmin = -180
    xmax = 180
    ymin = -90
    ymax = 90
    scale_y = 2.0
    scale_x = 1.0

    message = HarmonyMessage(
        {
            'format': {
                'crs': crs,
                'scaleSize': {'x': scale_x, 'y': scale_y},
                'scaleExtent': {
                    'x': {'min': xmin, 'max': xmax},
                    'y': {'min': ymin, 'max': ymax},
                },
            },
        }
    )
    expected_height = (ymax - ymin) / scale_y
    expected_width = (xmax - xmin) / scale_x

    actual_area_definitions = compute_target_areas(
        message, smap_projected_netcdf_file, var_info
    )

    # We pulled the grid from the message.
    mock_get_area_definition_from_message.assert_called_once_with(message)
    # We did not compute the source grid area.
    mock_create_target_areas_from_source.assert_not_called()

    expected_grid = GridDimensionPair('/y', '/x')

    assert expected_grid in actual_area_definitions.keys()
    actual_area = actual_area_definitions[expected_grid]

    assert actual_area.shape == (expected_height, expected_width)
    assert actual_area.area_extent == (xmin, ymin, xmax, ymax)
    assert actual_area.proj_str == crs


@patch(
    'harmony_regridding_service.grid.create_target_areas_from_source',
    wraps=create_target_areas_from_source,
)
@patch(
    'harmony_regridding_service.grid.get_area_definition_from_message',
    wraps=get_area_definition_from_message,
)
def test_compute_target_areas_without_parameters(
    mock_get_area_definition_from_message,
    mock_create_target_areas_from_source,
    smap_projected_netcdf_file,
    var_info_fxn,
):
    """Ensure Area Definition generated from source when no grid params.

    This test uses a Harmony Message with no grid parameters, and verifies that
    the resulting target from compute_target_areas is generated from the source
    grid.

    """
    var_info = var_info_fxn(smap_projected_netcdf_file)

    # Default CRS should be 4326 when the message does not have one.
    crs = CRS.from_epsg(4326)

    message = HarmonyMessage({})

    # values from our SMAP fixture.
    expected_width = 5
    expected_height = 9
    expected_area_extent = (-161.2811, 58.2848, -160.9077, 58.9562)

    actual_areas = compute_target_areas(message, smap_projected_netcdf_file, var_info)
    actual_area_definition = actual_areas[('/y', '/x')]

    # We called the function that pulls a grid from the source
    mock_create_target_areas_from_source.assert_called_once_with(
        smap_projected_netcdf_file, var_info, crs
    )
    # we did not get any information from the message.
    mock_get_area_definition_from_message.assert_not_called()

    assert actual_area_definition.shape == (expected_height, expected_width)
    assert actual_area_definition.area_extent == approx(expected_area_extent, abs=1e-4)
    assert CRS.from_proj4(actual_area_definition.proj_str).equals(
        crs, ignore_axis_order=True
    )
    assert CRS.from_proj4(actual_area_definition.proj_str).equals(
        crs, ignore_axis_order=True
    )


@patch(
    'harmony_regridding_service.grid.create_target_areas_from_source',
    wraps=create_target_areas_from_source,
)
@patch(
    'harmony_regridding_service.grid.get_area_definition_from_message',
    wraps=get_area_definition_from_message,
)
def test_compute_target_areas_with_only_CRS_parameter(
    mock_get_area_definition_from_message,
    mock_create_target_areas_from_source,
    smap_projected_netcdf_file,
    var_info_fxn,
):
    """Ensure Area Definition generated from source if only CRS in message.

    This test uses a Harmony Message with just a CRS parameter, and verifies
    that the resulting target from compute_target_areas is generated from the
    source grid.

    """
    var_info = var_info_fxn(smap_projected_netcdf_file)

    crs = '+datum=WGS84 +no_defs +proj=longlat +type=crs'
    target_crs = CRS(crs)

    message = HarmonyMessage(
        {
            'format': {
                'crs': crs,
            }
        }
    )

    expected_width = 5
    expected_height = 9
    expected_area_extent = (-161.2811, 58.2848, -160.9077, 58.9562)

    actual_area_definitions = compute_target_areas(
        message, smap_projected_netcdf_file, var_info
    )

    mock_create_target_areas_from_source.assert_called_once_with(
        smap_projected_netcdf_file, var_info, target_crs
    )

    expected_grid = GridDimensionPair('/y', '/x')
    assert expected_grid in actual_area_definitions
    actual_area = actual_area_definitions[expected_grid]

    assert actual_area.shape == (expected_height, expected_width)
    assert actual_area.area_extent == approx(expected_area_extent, abs=1e-4)
    assert CRS.from_proj4(actual_area.proj_str).equals(crs, ignore_axis_order=True)


@patch(
    'harmony_regridding_service.grid.create_target_areas_from_source',
    wraps=create_target_areas_from_source,
)
@patch(
    'harmony_regridding_service.grid.get_area_definition_from_message',
    wraps=get_area_definition_from_message,
)
def test_compute_target_areas_with_mulitple_grids_no_params(
    mock_get_area_definition_from_message,
    mock_create_target_areas_from_source,
    test_spl3ftp_ncfile,
    var_info_fxn,
):
    """Ensure Area Definition generated from source for both grids.

    This test uses an empty Harmony Message, and verifies
    that the resulting target areas from compute_target_areas are generated from the
    source grid.

    """
    var_info = var_info_fxn(test_spl3ftp_ncfile)
    crs = CRS.from_epsg(4326)

    message = HarmonyMessage({})

    expected_global_grid = GridDimensionPair(
        '/Freeze_Thaw_Retrieval_Data_Global/y', '/Freeze_Thaw_Retrieval_Data_Global/x'
    )
    expected_global = {
        'width': 186,
        'height': 73,
        'extent': (-70.02074, 60.128553, -9.89626, 83.63197),
    }
    expected_polar_grid = GridDimensionPair(
        '/Freeze_Thaw_Retrieval_Data_Polar/y', '/Freeze_Thaw_Retrieval_Data_Polar/x'
    )
    expected_polar = {
        'width': 263,
        'height': 122,
        'extent': (-86.361813, 48.6998, -1.5823, 88.0526),
    }

    actual_area_definitions = compute_target_areas(
        message, test_spl3ftp_ncfile, var_info
    )

    mock_create_target_areas_from_source.assert_called_once_with(
        test_spl3ftp_ncfile, var_info, crs
    )
    mock_get_area_definition_from_message.assert_not_called()

    # Assert both grids present in the area definitions.
    assert expected_polar_grid in actual_area_definitions
    actual_polar_area = actual_area_definitions[expected_polar_grid]
    assert actual_polar_area.shape == (
        expected_polar['height'],
        expected_polar['width'],
    )
    assert actual_polar_area.area_extent == approx(expected_polar['extent'], abs=1e-4)
    assert CRS.from_proj4(actual_polar_area.proj_str).equals(
        crs, ignore_axis_order=True
    )

    assert expected_global_grid in actual_area_definitions
    actual_global_area = actual_area_definitions[expected_global_grid]
    assert actual_global_area.shape == (
        expected_global['height'],
        expected_global['width'],
    )
    assert actual_global_area.area_extent == approx(expected_global['extent'], abs=1e-4)
    assert CRS.from_proj4(actual_global_area.proj_str).equals(
        crs, ignore_axis_order=True
    )


@patch('harmony_regridding_service.grid.AreaDefinition', wraps=AreaDefinition)
def test_get_area_definition_from_message(mock_area_definition):
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
    expected_area_extent = (xmin, ymin, xmax, ymax)

    actual_area_definition = get_area_definition_from_message(message)

    assert actual_area_definition.area_extent == expected_area_extent
    assert actual_area_definition.shape == (expected_height, expected_width)

    mock_area_definition.assert_called_once_with(
        'target_area_id',
        'target area definition',
        None,
        crs,
        expected_width,
        expected_height,
        expected_area_extent,
    )


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


def test_get_geographic_area_extent():
    """Verify extent behavior.

    extent is just defined:
    (np.min(lons), np.min(lats), np.max(lons), np.max(lats))

    """
    mock_area_def = Mock()
    lats = np.array([-10, 0, 10, -20])
    lons = np.array([-170, -175, -160, -170])

    mock_area_def.get_lonlats.return_value = (lons, lats)

    expected_area_extent = (-175, -20, -160, 10)

    actual_area_extent = get_geographic_area_extent(mock_area_def)

    assert expected_area_extent == actual_area_extent


def test_get_geographic_resolution():
    """Ensure conversion from meters to degrees is correct."""
    meters_per_degree = 111320.0

    ease_resolution = (36000.0, 36000.0)
    ease_upper_left = (-9000000.0, 9000000.0)
    ease_width = 500
    ease_height = 500

    projected_crs = CRS.from_epsg(6931)
    projected_area = create_area_def(
        'test_ease_grid_polar',
        projected_crs,
        width=ease_width,
        height=ease_height,
        upper_left_extent=ease_upper_left,
        resolution=ease_resolution,
    )
    expected_resolution = (36000.0 / meters_per_degree, 36000.0 / meters_per_degree)

    actual_geographic_resolution = get_geographic_resolution(projected_area)

    assert expected_resolution == actual_geographic_resolution


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


def test_compute_projected_horizontal_source_grids(
    var_info_fxn,
    smap_projected_netcdf_file,
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
        ('/y', '/x'),
        smap_projected_netcdf_file,
        var_info,
    )

    np.testing.assert_array_almost_equal(latitudes, expected_latitudes)
    np.testing.assert_array_almost_equal(longitudes, expected_longitudes)


@patch('harmony_regridding_service.grid.crs_from_source_data')
@patch('harmony_regridding_service.grid.compute_area_extent_from_regular_x_y_coords')
def test_create_area_definition_for_projected_source_grid(
    mock_compute_area_extent_from_regular_x_y_coords,
    mock_crs_from_source_data,
    smap_projected_netcdf_file,
    var_info_fxn,
):
    var_info = var_info_fxn(smap_projected_netcdf_file)

    mock_area_extent = (-180, -90, 180, 90)
    mock_compute_area_extent_from_regular_x_y_coords.return_value = mock_area_extent

    test_crs = 'epsg:6933'
    mock_crs_from_source_data.return_value = test_crs
    expected_width = 5
    expected_height = 6

    actual_area_definition = create_area_definition_for_projected_source_grid(
        smap_projected_netcdf_file, ('/y', '/x'), var_info
    )

    mock_crs_from_source_data.assert_called_once()
    assert actual_area_definition.area_extent == mock_area_extent
    assert actual_area_definition.shape == (expected_height, expected_width)
    assert actual_area_definition.crs == test_crs


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


@pytest.mark.parametrize(
    'test_extent, expected, description',
    [
        ((1, 2, 3, 4), (1, 2, 3, 4), 'no reordering necessary'),
        ((3, 2, 1, 4), (1, 2, 3, 4), 'x needs swapped'),
        ((1, 4, 3, 2), (1, 2, 3, 4), 'y needs swapped'),
        ((3, 4, 1, 2), (1, 2, 3, 4), 'x and y need swapped'),
    ],
)
def test_reorder_extents(test_extent, expected, description):
    assert expected == reorder_extents(*test_extent)


@patch('harmony_regridding_service.grid.create_area_def', wraps=create_area_def)
def test_convert_projected_area_to_geographic_ease_grid_global(mock_create_area_def):
    """Test converting projected area.

    Create a projected AreaDefinition using EPSG:6933
    (NSIDC EASE-Grid 2.0 Global)
    """
    ease_resolution = (36032.220840584, 36032.220840584)
    ease_upper_left = (-17367530.4451615, 7314540.8306386)
    ease_width = 964
    ease_height = 406

    projected_crs = CRS.from_epsg(6933)
    projected_area = create_area_def(
        'test_ease_grid_global',
        projected_crs,
        width=ease_width,
        height=ease_height,
        upper_left_extent=ease_upper_left,
        resolution=ease_resolution,
    )

    lons, lats = projected_area.get_lonlats()
    expected_geo_extent = (np.min(lons), np.min(lats), np.max(lons), np.max(lats))
    expected_resolution = (ease_resolution[0] / 111320.0, ease_resolution[1] / 111320.0)

    target_crs = CRS.from_epsg(4326)
    actual_geographic_area = convert_projected_area_to_geographic(
        projected_area, target_crs
    )

    mock_create_area_def.assert_called_once_with(
        'Geographic Area',
        target_crs,
        area_extent=expected_geo_extent,
        resolution=expected_resolution,
    )

    assert isinstance(actual_geographic_area, AreaDefinition)
    assert actual_geographic_area.crs == target_crs
    assert actual_geographic_area.width == 1112
    assert actual_geographic_area.height == 517

    # creating an area definition by extent and resolution does not ensure
    # whole numbers of grid cells and pyresample adjusts the resolution
    # accordingly.
    assert actual_geographic_area.resolution != expected_resolution

    lon_min, lat_min, lon_max, lat_max = actual_geographic_area.area_extent
    assert -180 < lon_min < lon_max < 180
    assert -90 < lat_min < lat_max < 90


@patch('harmony_regridding_service.grid.create_area_def', wraps=create_area_def)
def test_convert_projected_area_to_geographic_ease_grid_polar(mock_create_area_def):
    """Test converting projected area.

    Create a projected AreaDefinition using EPSG:6931
    (NSIDC EASE-Grid 2.0 Northern)
    """
    ease_resolution = (36000.0, 36000.0)
    ease_upper_left = (-9000000.0, 9000000.0)
    ease_width = 500
    ease_height = 500

    projected_crs = CRS.from_epsg(6931)
    projected_area = create_area_def(
        'test_ease_grid_polar',
        projected_crs,
        width=ease_width,
        height=ease_height,
        upper_left_extent=ease_upper_left,
        resolution=ease_resolution,
    )

    lons, lats = projected_area.get_lonlats()
    expected_geo_extent = (np.min(lons), np.min(lats), np.max(lons), np.max(lats))
    expected_resolution = (ease_resolution[0] / 111320.0, ease_resolution[1] / 111320.0)

    target_crs = CRS.from_epsg(4326)
    actual_geographic_area = convert_projected_area_to_geographic(
        projected_area, target_crs
    )

    mock_create_area_def.assert_called_once_with(
        'Geographic Area',
        target_crs,
        area_extent=expected_geo_extent,
        resolution=expected_resolution,
    )

    assert isinstance(actual_geographic_area, AreaDefinition)
    assert actual_geographic_area.crs == target_crs
    assert actual_geographic_area.width == 1113
    assert actual_geographic_area.height == 529

    # creating an area definition by extent and resolution does not ensure
    # whole numbers of grid cells and pyresample adjusts the resolution
    # accordingly.
    assert actual_geographic_area.resolution != expected_resolution

    lon_min, lat_min, lon_max, lat_max = actual_geographic_area.area_extent
    assert -180 < lon_min < lon_max < 180
    assert -90 < lat_min < lat_max < 90
