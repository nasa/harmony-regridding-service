from unittest import TestCase
from unittest.mock import patch

from netCDF4 import Dataset
import numpy as np
from logging import getLogger
from pathlib import Path
from pyresample.geometry import AreaDefinition
import re
from tempfile import mkdtemp
from shutil import rmtree

from harmony.message import Message
from varinfo import VarInfoFromNetCDF4

from harmony_regridding_service.regridding_service import (
    _compute_horizontal_source_grids, _compute_num_elements,
    _compute_source_swath, _compute_target_area, _get_projection_x_dim,
    _get_projection_y_dim, _grid_height, _grid_width)
from harmony_regridding_service.exceptions import InvalidSourceDimensions


class TestRegriddingService(TestCase):
    """Test the regridding_service module."""

    @classmethod
    def setUpClass(cls):
        """fixtures for all class tests."""
        cls.tmp_dir = mkdtemp()
        cls.logger = getLogger()

        cls.test_1D_dimensions_ncfile = Path(cls.tmp_dir, '1D_test.nc')
        cls.test_2D_dimensions_ncfile = Path(cls.tmp_dir, '2D_test.nc')

        longitudes = np.array([-180, -80, -45, 45, 80, 180])
        latitudes = np.array([90, 45, 0, -46, -89])

        # Set up a file with one dimensional /lon and /lat variables.
        dataset = Dataset(cls.test_1D_dimensions_ncfile, 'w')
        dataset.createDimension('lon', size=len(longitudes))
        dataset.createDimension('lat', size=len(latitudes))
        dataset.createVariable('/lon', longitudes.dtype, dimensions=('lon'))
        dataset.createVariable('/lat', latitudes.dtype, dimensions=('lat'))
        dataset['lon'].setncattr('units', 'degrees_east')
        dataset['lat'].setncattr('units', 'degrees_north')
        dataset['lat'][:] = latitudes
        dataset['lon'][:] = longitudes
        dataset.close()

        # Set up a file with two dimensional /lon and /lat variables.
        dataset = Dataset(cls.test_2D_dimensions_ncfile, 'w')
        dataset.createDimension('lon', size=(len(longitudes)))
        dataset.createDimension('lat', size=(len(latitudes)))
        dataset.createVariable('/lon',
                               longitudes.dtype,
                               dimensions=('lon', 'lat'))
        dataset.createVariable('/lat',
                               latitudes.dtype,
                               dimensions=('lon', 'lat'))
        dataset['lon'].setncattr('units', 'degrees_east')
        dataset['lat'].setncattr('units', 'degrees_north')
        dataset['lat'][:] = np.broadcast_to(latitudes, (6, 5))
        dataset['lon'][:] = np.broadcast_to(longitudes, (5, 6)).T
        dataset.close()

        # Set up test Harmony messages
        cls.test_message_with_scale_size = Message({
            'format': {
                'scaleSize': {
                    'x': 10,
                    'y': 10
                },
                'scaleExtent': {
                    'x': {
                        'min': 0,
                        'max': 1000
                    },
                    'y': {
                        'min': 0,
                        'max': 500
                    }
                }
            }
        })

        cls.test_message_with_height_width = Message({
            'format': {
                'height': 80,
                'width': 40,
                'scaleExtent': {
                    'x': {
                        'min': 0,
                        'max': 1000
                    },
                    'y': {
                        'min': 0,
                        'max': 500
                    }
                }
            }
        })

    @classmethod
    def tearDownCass(cls):
        rmtree(cls.tmp_dir)

    @patch('harmony_regridding_service.regridding_service.AreaDefinition', wraps=AreaDefinition)
    def test_compute_target_area(self, mock_area):
        """Ensure Area Definition correctly generated"""
        crs = '+proj=longlat +datum=WGS84 +no_defs +type=crs'
        xmin = -180
        xmax = 180
        ymin = -90
        ymax = 90

        message = Message({
            'format': {
                'crs': crs,
                'scaleSize': {
                    'x': 1.0,
                    'y': 2.0
                },
                'scaleExtent': {
                    'x': {
                        'min': xmin,
                        'max': xmax
                    },
                    'y': {
                        'min': ymin,
                        'max': ymax
                    }
                }
            }
        })

        expected_height = 90
        expected_width = 360

        actual_area = _compute_target_area(message)

        self.assertEqual(actual_area.shape, (expected_height, expected_width))
        self.assertEqual(actual_area.shape, (expected_height, expected_width))
        self.assertEqual(actual_area.area_extent, (xmin, ymin, xmax, ymax))
        self.assertEqual(actual_area.proj4_string, crs)
        mock_area.assert_called_once_with(
            'target_area_id', 'target area definition', None, crs,
            expected_width, expected_height,
            (xmin, ymin, xmax, ymax))

    def test_grid_height(self):
        expected_grid_height = 50
        actual_grid_height = _grid_height(self.test_message_with_scale_size)
        self.assertEqual(expected_grid_height, actual_grid_height)

    def test_grid_height_message_includes_height(self):
        expected_grid_height = 80
        actual_grid_height = _grid_height(self.test_message_with_height_width)
        self.assertEqual(expected_grid_height, actual_grid_height)

    def test_grid_width(self):
        expected_grid_width = 100
        actual_grid_width = _grid_width(self.test_message_with_scale_size)
        self.assertEqual(expected_grid_width, actual_grid_width)

    def test_grid_width_message_includes_width(self):
        expected_grid_width = 40
        actual_grid_width = _grid_width(self.test_message_with_height_width)
        self.assertEqual(expected_grid_width, actual_grid_width)

    def test_compute_num_elements(self):
        xmin = 0
        xmax = 1000
        ymin = 0
        ymax = 500

        message = Message({
            'format': {
                'scaleSize': {
                    'x': 10,
                    'y': 10
                },
                'scaleExtent': {
                    'x': {
                        'min': xmin,
                        'max': xmax
                    },
                    'y': {
                        'min': ymin,
                        'max': ymax
                    }
                }
            }
        })

        expected_x_elements = 100
        expected_y_elements = 50
        actual_x_elements = _compute_num_elements(message, 'x')
        actual_y_elements = _compute_num_elements(message, 'y')

        self.assertEqual(expected_x_elements, actual_x_elements)
        self.assertEqual(expected_y_elements, actual_y_elements)

    def test_get_projection_x_dim(self):
        var_info = VarInfoFromNetCDF4(self.test_1D_dimensions_ncfile,
                                      self.logger)
        dims = ('/lat', '/lon')
        expected_dim = '/lat'

        actual = _get_projection_x_dim(dims, var_info)
        self.assertEqual(expected_dim, actual)

    def test_get_projection_y_dim(self):
        var_info = VarInfoFromNetCDF4(self.test_1D_dimensions_ncfile,
                                      self.logger)
        dims = ('/lat', '/lon')
        expected_dim = '/lon'

        actual = _get_projection_y_dim(dims, var_info)
        self.assertEqual(expected_dim, actual)

    def test_get_projection_y_dim_no_variable(self):
        var_info = VarInfoFromNetCDF4(self.test_1D_dimensions_ncfile,
                                      self.logger)
        dims = ('/baddim1', '/baddim2')

        with self.assertRaisesRegex(InvalidSourceDimensions,
                                    "No longitude dimension found"):
            _get_projection_y_dim(dims, var_info)

    def test_get_projection_x_dim_no_variable(self):
        var_info = VarInfoFromNetCDF4(self.test_1D_dimensions_ncfile,
                                      self.logger)
        dims = ('/baddim1', '/baddim2')

        with self.assertRaisesRegex(InvalidSourceDimensions,
                                    "No latitude dimension found"):
            _get_projection_x_dim(dims, var_info)

    @patch('harmony_regridding_service.regridding_service.SwathDefinition')
    @patch(
        'harmony_regridding_service.regridding_service._compute_horizontal_source_grids'
    )
    def test_compute_source_swath(self, mock_horiz_source_grids, mock_swath):
        """Ensure source swaths are correctly generated."""
        grid_dims = ('/lon', '/lat')
        filepath = 'path to a file'
        var_info = {"fake": "varinfo object"}
        lons = np.array([[1, 1], [1, 1], [1, 1]])
        lats = np.array([[2, 2], [2, 2], [2, 2]])

        mock_horiz_source_grids.return_value = (lons, lats)

        _compute_source_swath(grid_dims, filepath, var_info)

        # horizontal grids were called successfully
        mock_horiz_source_grids.assert_called_with(grid_dims, filepath,
                                                   var_info)
        # swath was called with the horizontal 2d grids.
        mock_swath.assert_called_with(lons=lons, lats=lats)

    def test_expected_result_compute_horizontal_source_grids(self):
        """Exercises the single function for computing horizontal grids."""
        var_info = VarInfoFromNetCDF4(self.test_1D_dimensions_ncfile,
                                      self.logger)

        expected_longitudes = np.array([[-180, -80, -45, 45, 80, 180],
                                        [-180, -80, -45, 45, 80, 180],
                                        [-180, -80, -45, 45, 80, 180],
                                        [-180, -80, -45, 45, 80, 180],
                                        [-180, -80, -45, 45, 80, 180]])

        expected_latitudes = np.array([[90, 90, 90, 90, 90, 90],
                                       [45, 45, 45, 45, 45, 45],
                                       [0, 0, 0, 0, 0, 0],
                                       [-46, -46, -46, -46, -46, -46],
                                       [-89, -89, -89, -89, -89, -89]])

        test_args = [('/lon', '/lat'), ('/lat', '/lon')]

        for grid_dimensions in test_args:
            with self.subTest(
                    f'independent grid_dimension order {grid_dimensions}'):
                longitudes, latitudes = _compute_horizontal_source_grids(
                    grid_dimensions, self.test_1D_dimensions_ncfile, var_info)

                np.testing.assert_array_equal(expected_latitudes, latitudes)
                np.testing.assert_array_equal(expected_longitudes, longitudes)

    def test_2D_lat_lon_input_compute_horizontal_source_grids(self):
        var_info = VarInfoFromNetCDF4(self.test_2D_dimensions_ncfile,
                                      self.logger)
        grid_dimensions = ('/lat', '/lon')

        expected_regex = re.escape('Incorrect source data dimensions. '
                                   'rows:(6, 5), columns:(6, 5)')
        with self.assertRaisesRegex(InvalidSourceDimensions, expected_regex):
            _compute_horizontal_source_grids(grid_dimensions,
                                             self.test_2D_dimensions_ncfile,
                                             var_info)
