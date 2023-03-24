import re
from logging import getLogger
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from harmony.message import Message
from netCDF4 import Dataset, Variable
from numpy.testing import assert_array_equal
from pyresample.geometry import AreaDefinition
from varinfo import VarInfoFromNetCDF4

from harmony_regridding_service.exceptions import InvalidSourceDimensions
from harmony_regridding_service.regridding_service import (
    HRS_VARINFO_CONFIG_FILENAME, _all_dimension_variables, _all_dimensions,
    _clone_variables, _compute_horizontal_source_grids, _compute_num_elements,
    _compute_source_swath, _compute_target_area, _copy_1d_dimension_variables,
    _copy_dimension, _copy_dimension_variables, _copy_dimensions,
    _copy_resampled_bounds_variable, _create_dimension,
    _create_resampled_dimensions, _get_bounds_var, _get_dimension,
    _get_projection_x_dim, _get_projection_y_dim, _get_variable, _grid_height,
    _grid_width, _resampled_dimension_variable_names, _resampled_dimensions,
    _transfer_dimensions, _unresampled_variables)


class TestRegriddingService(TestCase):
    """Test the regridding_service module."""

    @classmethod
    def setUpClass(cls):
        """fixtures for all class tests."""
        cls.tmp_dir = mkdtemp()
        cls.logger = getLogger()
        cls.counter = 0
        # Test fixtures representing typical input data
        # ATL14 data includes two X,Y input grids.
        cls.test_ATL14_ncfile = Path(
            Path(__file__).parent, 'fixtures', 'empty-ATL14.nc')
        # MERRA2 has 4 dim variables that are flat at the root
        cls.test_MERRA2_ncfile = Path(
            Path(__file__).parent, 'fixtures', 'empty-MERRA2.nc')
        # IMERG data variables are contained in netcdf groups
        cls.test_IMERG_ncfile = Path(
            Path(__file__).parent, 'fixtures', 'empty-IMERG.nc')

        cls.longitudes = np.array([-180, -80, -45, 45, 80, 180],
                                  dtype=np.dtype('f8'))
        cls.latitudes = np.array([90, 45, 0, -46, -89], dtype=np.dtype('f8'))

        cls.test_1D_dimensions_ncfile = Path(cls.tmp_dir, '1D_test.nc')
        cls.test_2D_dimensions_ncfile = Path(cls.tmp_dir, '2D_test.nc')

        # Set up a file with one dimensional /lon and /lat variables.
        dataset = Dataset(cls.test_1D_dimensions_ncfile, 'w')
        dataset.createDimension('time', size=None)
        dataset.createDimension('lon', size=len(cls.longitudes))
        dataset.createDimension('lat', size=len(cls.latitudes))
        dataset.createDimension('bnds', size=2)
        dataset.createVariable('/lon', cls.longitudes.dtype, dimensions=('lon'))
        dataset.createVariable('/lat', cls.latitudes.dtype, dimensions=('lat'))
        dataset.createVariable('/data',
                               np.dtype('f8'),
                               dimensions=('lon', 'lat'))
        dataset.createVariable('/time', np.dtype('f8'), dimensions=('time'))
        dataset.createVariable('/time_bnds',
                               np.dtype('u2'),
                               dimensions=('time', 'bnds'))

        dataset['lon'].setncattr('units', 'degrees_east')
        dataset['lat'].setncattr('units', 'degrees_north')
        dataset['lat'].setncattr('dropme', 'dont get copied')
        dataset['lat'][:] = cls.latitudes
        dataset['lon'][:] = cls.longitudes
        dataset['time'][:] = [1., 2., 3., 4.]
        dataset['data'][:] = np.ones((len(cls.longitudes), len(cls.latitudes)))
        dataset['time_bnds'][:] = np.array([[.5, 1.5, 2.5, 3.5],
                                            [1.5, 2.5, 3.5, 4.5]]).T
        dataset.close()

        # Set up a file with two dimensional /lon and /lat variables.
        dataset = Dataset(cls.test_2D_dimensions_ncfile, 'w')
        dataset.createDimension('lon', size=(len(cls.longitudes)))
        dataset.createDimension('lat', size=(len(cls.latitudes)))
        dataset.createVariable('/lon',
                               cls.longitudes.dtype,
                               dimensions=('lon', 'lat'))
        dataset.createVariable('/lat',
                               cls.latitudes.dtype,
                               dimensions=('lon', 'lat'))
        dataset['lon'].setncattr('units', 'degrees_east')
        dataset['lat'].setncattr('units', 'degrees_north')
        dataset['lat'][:] = np.broadcast_to(cls.latitudes, (6, 5))
        dataset['lon'][:] = np.broadcast_to(cls.longitudes, (5, 6)).T
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
    def setUp(self):
        self.target_tmp_dir = mkdtemp()

    @classmethod
    def tearDown(self):
        rmtree(self.target_tmp_dir)

    @classmethod
    def tearDownCass(cls):
        rmtree(cls.tmp_dir)

    @classmethod
    def target_file(self):
        """return a temporary target netcdf filename"""
        return Path(self.target_tmp_dir, f'target_{uuid4()}.nc')

    @classmethod
    def var_info(cls, source_filename):
        return VarInfoFromNetCDF4(source_filename,
                                  cls.logger,
                                  config_file=HRS_VARINFO_CONFIG_FILENAME)

    @classmethod
    def test_area(cls, width=360, height=180, area_extent=(-180, -90, 180, 90)):
        projection = '+proj=longlat +datum=WGS84 +no_defs +type=crs'
        return AreaDefinition('test_id', 'test area definition', None,
                              projection, width, height, area_extent)

    def test_copy_1d_dimension_variables(self):
        target_file = self.target_file()
        target_area = self.test_area()
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        dim_var_names = {'/lon', '/lat'}
        expected_attributes = {'long_name', 'standard_name', 'units'}
        vars_copied = []
        with Dataset(self.test_1D_dimensions_ncfile, mode='r') as source_ds, \
             Dataset(target_file, mode='w') as target_ds:
            _transfer_dimensions(source_ds, target_ds, target_area, var_info)

            vars_copied = _copy_1d_dimension_variables(source_ds, target_ds,
                                                       dim_var_names,
                                                       target_area, var_info)

        self.assertEqual(dim_var_names, vars_copied)
        with Dataset(target_file, mode='r') as validate:
            assert_array_equal(validate['/lon'][:],
                               target_area.projection_x_coords)
            assert_array_equal(validate['/lat'][:],
                               target_area.projection_y_coords)
            self.assertSetEqual(expected_attributes,
                                set(validate['/lat'].ncattrs()))
            with self.assertRaises(AttributeError):
                validate['/lat'].getncattr('dropme')

    def test_copy_dimension_variables(self):
        target_file = self.target_file()
        width = 300
        height = 150
        target_area = self.test_area(width=width, height=height)
        var_info = self.var_info(self.test_MERRA2_ncfile)
        expected_vars_copied = {'/lon', '/lat'}

        with Dataset(self.test_MERRA2_ncfile, mode='r') as source_ds, \
             Dataset(target_file, mode='w') as target_ds:
            _transfer_dimensions(source_ds, target_ds, target_area, var_info)

            vars_copied = _copy_dimension_variables(source_ds, target_ds,
                                                    target_area, var_info)

            self.assertSetEqual(expected_vars_copied, vars_copied)

        with Dataset(target_file, mode='r') as validate:
            self.assertEqual(validate.dimensions['lon'].size, width)
            self.assertEqual(validate.dimensions['lat'].size, height)
            self.assertEqual(validate.dimensions['lev'].size, 42)

    def test_get_bound_var(self):
        var_info = self.var_info(self.test_IMERG_ncfile)
        expected_bounds = 'lon_bnds'

        actual_bounds = _get_bounds_var(var_info, '/Grid/lon')
        self.assertEqual(expected_bounds, actual_bounds)

    def test_copy_resampled_bounds_variable(self):
        target_file = self.target_file()
        target_area = self.test_area()
        var_info = self.var_info(self.test_IMERG_ncfile)
        bnds_var = '/Grid/lat_bnds'
        var_copied = None

        expected_lat_bnds = np.array([
            target_area.projection_y_coords + .5,
            target_area.projection_y_coords - .5
        ]).T

        with Dataset(self.test_IMERG_ncfile, mode='r') as source_ds, \
             Dataset(target_file, mode='w') as target_ds:
            _transfer_dimensions(source_ds, target_ds, target_area, var_info)

            var_copied = _copy_resampled_bounds_variable(
                source_ds, target_ds, bnds_var, target_area, var_info)

        self.assertEqual({bnds_var}, var_copied)
        with Dataset(target_file, mode='r') as validate:
            assert_array_equal(expected_lat_bnds,
                               validate['Grid']['lat_bnds'][:])

    def test_resampled_dimension_variable_names(self):
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        expected_resampled = {'/lon', '/lat'}

        actual_resampled = _resampled_dimension_variable_names(var_info)
        self.assertEqual(expected_resampled, actual_resampled)

    def test_grouped_resampled_dimension_variable_names(self):
        var_info = self.var_info(self.test_IMERG_ncfile)
        expected_resampled = {
            '/Grid/lon', '/Grid/lat', '/Grid/lon_bnds', '/Grid/lat_bnds'
        }

        actual_resampled = _resampled_dimension_variable_names(var_info)
        self.assertEqual(expected_resampled, actual_resampled)

    def test_multiple_resampled_dimension_variable_names(self):
        var_info = self.var_info(self.test_ATL14_ncfile)
        expected_resampled = {'/x', '/y', '/tile_stats/x', '/tile_stats/y'}

        actual_resampled = _resampled_dimension_variable_names(var_info)
        self.assertEqual(expected_resampled, actual_resampled)

    def test_transfer_dimensions(self):
        """Tests transfer of all dimensions.

        test transfer of dimensions from source to target including resizing
        for the target's area definition.  The internal functions of
        _transfer_dimensions are tested further down in this file.

        """
        projection = '+proj=longlat +datum=WGS84 +no_defs +type=crs'
        width = 36
        height = 18
        area_extent = (-180, -90, 180, 90)
        test_area = AreaDefinition('test_id', 'test area definition', None,
                                   projection, width, height, area_extent)
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        target_file = self.target_file()

        with Dataset(self.test_1D_dimensions_ncfile, mode='r') as source_ds, \
             Dataset(target_file, mode='w') as target_ds:
            _transfer_dimensions(source_ds, target_ds, test_area, var_info)

        with Dataset(target_file, mode='r') as validate:
            self.assertEqual(validate.dimensions['bnds'].size, 2)
            self.assertEqual(validate.dimensions['time'].size, 0)
            self.assertTrue(validate.dimensions['time'].isunlimited())
            self.assertEqual(validate.dimensions['lon'].size, width)
            self.assertEqual(validate.dimensions['lat'].size, height)

    def test_clone_variables(self):
        target_file = self.target_file()
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        projection = '+proj=longlat +datum=WGS84 +no_defs +type=crs'
        width = 36
        height = 18
        area_extent = (-180, -90, 180, 90)
        test_area = AreaDefinition('test_id', 'test area definition', None,
                                   projection, width, height, area_extent)
        copy_vars = {'/time', '/time_bnds'}
        with Dataset(self.test_1D_dimensions_ncfile, mode='r') as source_ds, \
             Dataset(target_file, mode='w') as target_ds:

            _transfer_dimensions(source_ds, target_ds, test_area, var_info)

            copied = _clone_variables(source_ds, target_ds, copy_vars)

            self.assertEqual(copy_vars, copied)

            with Dataset(target_file, mode='r') as validate:
                assert_array_equal(validate['time_bnds'],
                                   source_ds['time_bnds'])
                assert_array_equal(validate['time'], source_ds['time'])

    def test_create_resampled_dimensions(self):
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        projection = '+proj=longlat +datum=WGS84 +no_defs +type=crs'
        width = 36
        height = 18
        area_extent = (-180, -90, 180, 90)
        test_area = AreaDefinition('test_id', 'test area definition', None,
                                   projection, width, height, area_extent)
        target_file = self.target_file()

        with Dataset(target_file, mode='w') as target_ds:
            _create_resampled_dimensions({'/lat', '/lon'}, target_ds, test_area,
                                         var_info)

        with Dataset(target_file, mode='r') as validate:
            self.assertEqual(validate.dimensions['lat'].size, 18)
            self.assertEqual(validate.dimensions['lon'].size, 36)

    def test_create_nested_resampled_dimensions(self):
        var_info = self.var_info(self.test_IMERG_ncfile)
        projection = '+proj=longlat +datum=WGS84 +no_defs +type=crs'
        width = 360
        height = 180
        area_extent = (-180, -90, 180, 90)
        test_area = AreaDefinition('test_id', 'test area definition', None,
                                   projection, width, height, area_extent)
        target_file = self.target_file()
        with Dataset(target_file, mode='w') as target_ds:
            _create_resampled_dimensions({'/Grid/lon', '/Grid/lat'}, target_ds,
                                         test_area, var_info)

        with Dataset(target_file, mode='r') as validate:
            self.assertEqual(validate['Grid'].dimensions['lat'].size, 180)
            self.assertEqual(validate['Grid'].dimensions['lon'].size, 360)

    def test_get_variable(self):
        with Dataset(self.test_ATL14_ncfile, mode='r') as source_ds:
            var_grouped = _get_variable(source_ds, '/tile_stats/RMS_data')
            expected_grouped = source_ds['tile_stats'].variables['RMS_data']
            self.assertEqual(expected_grouped, var_grouped)

            var_flat = _get_variable(source_ds, '/ice_area')
            expected_flat = source_ds.variables['ice_area']
            self.assertEqual(expected_flat, var_flat)

    def test_create_dimension(self):
        name = '/somedim'
        size = 1000
        with Dataset(self.target_file(), mode='w') as target_ds:
            dim = _create_dimension(target_ds, name, size)
            self.assertTrue(type(dim), Variable)
            self.assertEqual(dim.size, size)
            self.assertEqual(dim.name, 'somedim')

    def test_create_nested_dimension(self):
        name = '/some/deeply/nested/dimname'
        size = 2000
        with Dataset(self.target_file(), mode='w') as target_ds:
            dim = _create_dimension(target_ds, name, size)
            self.assertTrue(type(dim), Variable)
            self.assertEqual(dim.size, size)
            self.assertEqual(dim.name, 'dimname')

    def test_get_flat_dimension(self):
        with Dataset(self.test_1D_dimensions_ncfile, mode='r') as source_ds:
            lat_dim = _get_dimension(source_ds, '/lat')
            self.assertTrue(type(lat_dim), Variable)
            self.assertTrue(lat_dim.size, len(self.latitudes))
            self.assertTrue(lat_dim.name, 'lat')

    def test_get_nested_dimension(self):
        with Dataset(self.test_IMERG_ncfile, mode='r') as source_ds:
            lat_dim = _get_dimension(source_ds, '/Grid/lat')
            self.assertTrue(type(lat_dim), Variable)
            self.assertTrue(lat_dim.name, 'lat')
            self.assertTrue(lat_dim.size, 1800)

    def test_copy_dimension(self):
        with Dataset(self.target_file(), mode='w' ) as target_ds, \
             Dataset(self.test_1D_dimensions_ncfile, mode='r') as source_ds:
            time_dimension = _copy_dimension('/time', source_ds, target_ds)
            self.assertTrue(time_dimension.isunlimited())
            self.assertEqual(time_dimension.size, 0)

            lon_dimension = _copy_dimension('/lon', source_ds, target_ds)
            self.assertFalse(lon_dimension.isunlimited())
            self.assertEqual(lon_dimension.size, len(self.longitudes))

    def test_copy_dimensions(self):
        test_target = self.target_file()
        with Dataset(test_target, mode='w' ) as target_ds, \
             Dataset(self.test_1D_dimensions_ncfile, mode='r') as source_ds:
            _copy_dimensions({'/lat', '/lon', '/time', '/bnds'}, source_ds,
                             target_ds)

        with Dataset(test_target, mode='r') as validate:
            self.assertTrue(validate.dimensions['time'].isunlimited())
            self.assertEqual(validate.dimensions['time'].size, 0)
            self.assertEqual(validate.dimensions['lat'].size,
                             len(self.latitudes))
            self.assertEqual(validate.dimensions['lon'].size,
                             len(self.longitudes))
            self.assertEqual(validate.dimensions['bnds'].size, 2)

    def test_copy_dimensions_with_groups(self):
        test_target = self.target_file()
        with Dataset(test_target, mode='w' ) as target_ds, \
             Dataset(self.test_IMERG_ncfile, mode='r') as source_ds:
            _copy_dimensions(
                {'/Grid/latv', '/Grid/lonv', '/Grid/nv', '/Grid/time'},
                source_ds, target_ds)

        with Dataset(test_target, mode='r') as validate:
            self.assertTrue(validate['Grid'].dimensions['time'].isunlimited())
            self.assertEqual(validate['Grid'].dimensions['time'].size, 0)
            self.assertEqual(validate['Grid'].dimensions['lonv'].size, 2)
            self.assertEqual(validate['Grid'].dimensions['latv'].size, 2)
            self.assertEqual(validate['Grid'].dimensions['nv'].size, 2)

    def test_all_dimensions(self):
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        expected_dimensions = {'/time', '/lon', '/lat', '/bnds'}
        actual_dimensions = _all_dimensions(var_info)
        self.assertEqual(expected_dimensions, actual_dimensions)

    def test_unresampled_variables(self):
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        expected_vars = {'/time', '/time_bnds'}
        actual_vars = _unresampled_variables(var_info)
        self.assertEqual(expected_vars, actual_vars)

    def test_unresampled_variables_IMERG(self):
        var_info = self.var_info(self.test_IMERG_ncfile)

        expected_vars = {'/Grid/time', '/Grid/time_bnds'}
        actual_vars = _unresampled_variables(var_info)
        self.assertEqual(expected_vars, actual_vars)

    def test_unresampled_variables_MERRA2(self):
        var_info = self.var_info(self.test_MERRA2_ncfile)

        expected_vars = {'/lev', '/time'}
        actual_vars = _unresampled_variables(var_info)
        self.assertEqual(expected_vars, actual_vars)

    def test_unresampled_variables_ATL14(self):
        var_info = self.var_info(self.test_ATL14_ncfile)

        expected_vars = {
            '/Polar_Stereographic', '/orbit_info/bounding_polygon_dim1',
            '/orbit_info/bounding_polygon_lat1',
            '/orbit_info/bounding_polygon_lon1',
            '/quality_assessment/qa_granule_fail_reason',
            '/quality_assessment/qa_granule_pass_fail'
        }
        actual_vars = _unresampled_variables(var_info)
        self.assertEqual(expected_vars, actual_vars)

    def test_all_dimension_variables(self):
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        expected_vars = {'/lat', '/lon', '/time'}
        actual_vars = _all_dimension_variables(var_info)
        self.assertEqual(expected_vars, actual_vars)

    def test_all_dimension_variables_2D(self):
        var_info = self.var_info(self.test_2D_dimensions_ncfile)
        expected_vars = {'/lat', '/lon'}
        actual_vars = _all_dimension_variables(var_info)
        self.assertEqual(expected_vars, actual_vars)

    def test_resampled_dimensions(self):
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        expected_dimensions = {'/lat', '/lon'}
        actual_dimensions = _resampled_dimensions(var_info)
        self.assertEqual(expected_dimensions, actual_dimensions)

    def test_resampled_dimensions_with_multiple_grids(self):
        """Return the dimension variables that are involved in resampling."""
        var_info = self.var_info(self.test_ATL14_ncfile)

        expected_dimensions = {'/x', '/y', '/tile_stats/x', '/tile_stats/y'}
        actual_dimensions = _resampled_dimensions(var_info)
        self.assertEqual(expected_dimensions, actual_dimensions)

    @patch('harmony_regridding_service.regridding_service.AreaDefinition',
           wraps=AreaDefinition)
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
        mock_area.assert_called_once_with('target_area_id',
                                          'target area definition', None, crs,
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
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        dims = ('/lat', '/lon')
        expected_dim = ['/lon']

        actual = _get_projection_x_dim(dims, var_info)
        self.assertEqual(expected_dim, actual)

    def test_get_projection_y_dim(self):
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        dims = ('/lat', '/lon')
        expected_dim = ['/lat']

        actual = _get_projection_y_dim(dims, var_info)
        self.assertEqual(expected_dim, actual)

    def test_get_projection_y_dim_no_variable(self):
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        dims = ('/baddim1', '/baddim2')

        expected_dims = []
        actual_dims = _get_projection_y_dim(dims, var_info)
        self.assertEqual(expected_dims, actual_dims)

    def test_get_projection_x_dim_no_variable(self):
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        dims = ('/baddim1', '/baddim2')
        expected_dims = []
        actual_dims = _get_projection_x_dim(dims, var_info)
        self.assertEqual(expected_dims, actual_dims)

    def test_get_projection_x_dim_with_bad_variable(self):
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        dims = ('/baddim1', '/lon')
        expected_dim = ['/lon']

        actual_dim = _get_projection_x_dim(dims, var_info)
        self.assertEqual(expected_dim, actual_dim)

    def test_get_projection_y_dim_multiple_values(self):
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        dims = ('/lat', '/lon', '/lat', '/ba')
        expected_dim = ['/lat', '/lat']

        actual = _get_projection_y_dim(dims, var_info)
        self.assertEqual(expected_dim, actual)

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
        var_info = self.var_info(self.test_1D_dimensions_ncfile)

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
        var_info = self.var_info(self.test_2D_dimensions_ncfile)
        grid_dimensions = ('/lat', '/lon')

        expected_regex = re.escape('Incorrect source data dimensions. '
                                   'rows:(6, 5), columns:(6, 5)')
        with self.assertRaisesRegex(InvalidSourceDimensions, expected_regex):
            _compute_horizontal_source_grids(grid_dimensions,
                                             self.test_2D_dimensions_ncfile,
                                             var_info)
