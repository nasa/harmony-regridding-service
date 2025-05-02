"""Tests regridding service."""

import re
from logging import getLogger
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
import xarray as xr
from harmony_service_lib.message import Message as HarmonyMessage
from harmony_service_lib.message import Source as HarmonySource
from netCDF4 import Dataset, Variable
from numpy.testing import assert_array_equal
from pyresample.geometry import AreaDefinition
from varinfo import VarInfoFromNetCDF4

import harmony_regridding_service.regridding_service as rs
from harmony_regridding_service.exceptions import (
    InvalidSourceDimensions,
    RegridderException,
    SourceDataError,
)
from harmony_regridding_service.regridding_service import _compute_array_bounds


class TestRegriddingService(TestCase):
    """Test the regridding_service module."""

    @classmethod
    def setUpClass(cls):
        """Fixtures for all class tests."""
        cls.tmp_dir = mkdtemp()
        cls.logger = getLogger()

        # Test fixtures representing typical input data
        # ATL14 data includes two X,Y input grids.
        cls.test_ATL14_ncfile = Path(
            Path(__file__).parent, 'fixtures', 'empty-ATL14.nc'
        )
        # MERRA2 has 4 dim variables that are flat at the root
        cls.test_MERRA2_ncfile = Path(
            Path(__file__).parent, 'fixtures', 'empty-MERRA2.nc'
        )
        # IMERG data variables are contained in netcdf groups
        cls.test_IMERG_ncfile = Path(
            Path(__file__).parent, 'fixtures', 'empty-IMERG.nc'
        )

        cls.longitudes = np.array([-180, -80, -45, 45, 80, 180], dtype=np.dtype('f8'))
        cls.latitudes = np.array([90, 45, 0, -46, -89], dtype=np.dtype('f8'))

        cls.test_1D_dimensions_ncfile = Path(cls.tmp_dir, '1D_test.nc')
        cls.test_2D_dimensions_ncfile = Path(cls.tmp_dir, '2D_test.nc')

        # Set up file with one dimensional /lon and /lat root variables
        dataset = Dataset(cls.test_1D_dimensions_ncfile, 'w')
        dataset.setncatts({'root-attribute1': 'value1', 'root-attribute2': 'value2'})

        # Set up some groups and metadata
        group1 = dataset.createGroup('/level1-nested1')
        group2 = dataset.createGroup('/level1-nested2')
        group2.setncatts({'level1-nested2': 'level1-nested2-value1'})
        group1.setncatts({'level1-nested1': 'level1-nested1-value1'})
        group3 = group1.createGroup('/level2-nested1')
        group3.setncatts({'level2-nested1': 'level2-nested1-value1'})

        dataset.createDimension('time', size=None)
        dataset.createDimension('lon', size=len(cls.longitudes))
        dataset.createDimension('lat', size=len(cls.latitudes))
        dataset.createDimension('bnds', size=2)

        dataset.createVariable('/lon', cls.longitudes.dtype, dimensions=('lon'))
        dataset.createVariable('/lat', cls.latitudes.dtype, dimensions=('lat'))
        dataset.createVariable('/data', np.dtype('f8'), dimensions=('lon', 'lat'))
        dataset.createVariable('/time', np.dtype('f8'), dimensions=('time'))
        dataset.createVariable(
            '/time_bnds', np.dtype('u2'), dimensions=('time', 'bnds')
        )

        dataset['lat'][:] = cls.latitudes
        dataset['lon'][:] = cls.longitudes
        dataset['time'][:] = [1.0, 2.0, 3.0, 4.0]
        dataset['data'][:] = np.arange(
            len(cls.longitudes) * len(cls.latitudes)
        ).reshape((len(cls.longitudes), len(cls.latitudes)))
        dataset['time_bnds'][:] = np.array(
            [[0.5, 1.5, 2.5, 3.5], [1.5, 2.5, 3.5, 4.5]]
        ).T

        dataset['lon'].setncattr('units', 'degrees_east')
        dataset['lat'].setncattr('units', 'degrees_north')
        dataset['lat'].setncattr('non-standard-attribute', 'Wont get copied')
        dataset['data'].setncattr('units', 'widgets per month')
        dataset.close()

        # Set up a file with two dimensional /lon and /lat variables.
        dataset = Dataset(cls.test_2D_dimensions_ncfile, 'w')
        dataset.createDimension('lon', size=(len(cls.longitudes)))
        dataset.createDimension('lat', size=(len(cls.latitudes)))
        dataset.createVariable('/lon', cls.longitudes.dtype, dimensions=('lon', 'lat'))
        dataset.createVariable('/lat', cls.latitudes.dtype, dimensions=('lon', 'lat'))
        dataset['lon'].setncattr('units', 'degrees_east')
        dataset['lat'].setncattr('units', 'degrees_north')
        dataset['lat'][:] = np.broadcast_to(cls.latitudes, (6, 5))
        dataset['lon'][:] = np.broadcast_to(cls.longitudes, (5, 6)).T
        dataset.close()

        # Set up test Harmony messages
        cls.test_message_with_scale_size = HarmonyMessage(
            {
                'format': {
                    'scaleSize': {'x': 10, 'y': 10},
                    'scaleExtent': {
                        'x': {'min': 0, 'max': 1000},
                        'y': {'min': 0, 'max': 500},
                    },
                }
            }
        )

        cls.test_message_with_height_width = HarmonyMessage(
            {
                'format': {
                    'height': 80,
                    'width': 40,
                    'scaleExtent': {
                        'x': {'min': 0, 'max': 1000},
                        'y': {'min': 0, 'max': 500},
                    },
                }
            }
        )

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
    def _generate_test_file(self):
        """Return a temporary target netcdf filename."""
        return Path(self.target_tmp_dir, f'target_{uuid4()}.nc')

    @classmethod
    def var_info(cls, source_filename):
        return VarInfoFromNetCDF4(
            source_filename, config_file=rs.HRS_VARINFO_CONFIG_FILENAME
        )

    @classmethod
    def _generate_test_area(
        cls, width=360, height=180, area_extent=(-180, -90, 180, 90)
    ):
        projection = '+proj=longlat +datum=WGS84 +no_defs +type=crs'
        return AreaDefinition(
            'test_id',
            'test area definition',
            None,
            projection,
            width,
            height,
            area_extent,
        )

    def test_group_by_ndim(self):
        with self.subTest('one var'):
            var_info = self.var_info(self.test_1D_dimensions_ncfile)
            variables = {'/data'}
            expected_sorted = {2: {'/data'}}
            actual_sorted = rs._group_by_ndim(var_info, variables)
            self.assertDictEqual(expected_sorted, actual_sorted)

        with self.subTest('MERRA2'):
            var_info = self.var_info(self.test_MERRA2_ncfile)
            variables = {'/OMEGA', '/RH', '/PHIS', '/PS', '/lat'}
            expected_sorted = {4: {'/OMEGA', '/RH'}, 3: {'/PHIS', '/PS'}, 1: {'/lat'}}
            actual_sorted = rs._group_by_ndim(var_info, variables)
            self.assertDictEqual(expected_sorted, actual_sorted)

    def test_walk_groups(self):
        """Demonstrate traversing all groups."""
        target_path = self._generate_test_file()
        groups = ['/a/nested/group', '/b/another/deeper/group2']
        expected_visited = {'a', 'nested', 'group', 'b', 'another', 'deeper', 'group2'}

        with Dataset(target_path, mode='w') as target_ds:
            for group in groups:
                target_ds.createGroup(group)

        actual_visited = set()
        with Dataset(target_path, mode='r') as validate:
            for groups in rs._walk_groups(validate):
                for group in groups:
                    actual_visited.update([group.name])

        self.assertSetEqual(expected_visited, actual_visited)

    def test_copy_1d_dimension_variables(self):
        target_file = self._generate_test_file()
        target_area = self._generate_test_area()
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        dim_var_names = {'/lon', '/lat'}
        expected_attributes = {'long_name', 'standard_name', 'units'}
        vars_copied = []
        with (
            Dataset(self.test_1D_dimensions_ncfile, mode='r') as source_ds,
            Dataset(target_file, mode='w') as target_ds,
        ):
            rs._transfer_dimensions(source_ds, target_ds, target_area, var_info)
            vars_copied = rs._copy_1d_dimension_variables(
                source_ds, target_ds, dim_var_names, target_area, var_info
            )

        self.assertEqual(dim_var_names, vars_copied)
        with Dataset(target_file, mode='r') as validate:
            assert_array_equal(validate['/lon'][:], target_area.projection_x_coords)
            assert_array_equal(validate['/lat'][:], target_area.projection_y_coords)
            self.assertSetEqual(expected_attributes, set(validate['/lat'].ncattrs()))
            with self.assertRaises(AttributeError):
                validate['/lat'].getncattr('non-standard-attribute')

    def test_copy_vars_without_metadata(self):
        target_file = self._generate_test_file()
        target_area = self._generate_test_area()
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        with (
            Dataset(self.test_1D_dimensions_ncfile, mode='r') as source_ds,
            Dataset(target_file, mode='w') as target_ds,
        ):
            rs._transfer_dimensions(source_ds, target_ds, target_area, var_info)
            rs._copy_var_without_metadata(source_ds, target_ds, '/data')

        with Dataset(target_file, mode='r') as validate:
            actual_metadata = {
                attr: validate['/data'].getncattr(attr)
                for attr in validate['/data'].ncattrs()
            }
            self.assertDictEqual({}, actual_metadata)

    def test_copy_var_with_attrs(self):
        target_file = self._generate_test_file()
        target_area = self._generate_test_area()
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        expected_metadata = {'units': 'widgets per month'}
        with (
            Dataset(self.test_1D_dimensions_ncfile, mode='r') as source_ds,
            Dataset(target_file, mode='w') as target_ds,
        ):
            rs._transfer_dimensions(source_ds, target_ds, target_area, var_info)
            rs._copy_var_with_attrs(source_ds, target_ds, '/data')

        with Dataset(target_file, mode='r') as validate:
            actual_metadata = {
                attr: validate['/data'].getncattr(attr)
                for attr in validate['/data'].ncattrs()
            }
            self.assertDictEqual(actual_metadata, expected_metadata)

    def test_copy_dimension_variables(self):
        target_file = self._generate_test_file()
        width = 300
        height = 150
        target_area = self._generate_test_area(width=width, height=height)
        var_info = self.var_info(self.test_MERRA2_ncfile)
        expected_vars_copied = {'/lon', '/lat'}

        with (
            Dataset(self.test_MERRA2_ncfile, mode='r') as source_ds,
            Dataset(target_file, mode='w') as target_ds,
        ):
            rs._transfer_dimensions(source_ds, target_ds, target_area, var_info)

            vars_copied = rs._copy_dimension_variables(
                source_ds, target_ds, target_area, var_info
            )

            self.assertSetEqual(expected_vars_copied, vars_copied)

        with Dataset(target_file, mode='r') as validate:
            self.assertEqual(validate.dimensions['lon'].size, width)
            self.assertEqual(validate.dimensions['lat'].size, height)
            self.assertEqual(validate.dimensions['lev'].size, 42)

    def test_prepare_data_plane(self):
        with self.subTest('floating point data without rotation'):
            var_info = self.var_info(self.test_MERRA2_ncfile)
            test_data = np.ma.array(
                np.arange(12).reshape(4, 3), fill_value=-9999.9, dtype=np.float32
            )
            var_name = '/T'
            expected_data = np.ma.copy(test_data)
            actual_data = rs._prepare_data_plane(
                test_data, var_info, var_name, cast_to=np.float64
            )

            self.assertEqual(np.float64, actual_data.dtype)
            np.testing.assert_equal(expected_data, actual_data)

        with self.subTest('floating point data with rotation'):
            var_info = self.var_info(self.test_IMERG_ncfile)
            test_data = np.array(np.arange(12).reshape(4, 3), dtype=np.float16)
            var_name = '/Grid/HQprecipitation'
            expected_data = np.copy(test_data.T)
            actual_data = rs._prepare_data_plane(
                test_data, var_info, var_name, cast_to=np.float64
            )

            self.assertEqual(np.float64, actual_data.dtype)
            np.testing.assert_equal(expected_data, actual_data)

        with self.subTest('integer data without rotation'):
            var_info = self.var_info(self.test_MERRA2_ncfile)
            test_data = np.array(np.arange(12).reshape(4, 3), dtype=np.int8)
            var_name = '/T'
            expected_data = np.copy(test_data)
            actual_data = rs._prepare_data_plane(
                test_data, var_info, var_name, cast_to=np.float64
            )

            self.assertEqual(np.float64, actual_data.dtype)
            np.testing.assert_equal(expected_data, actual_data)

        with self.subTest('integer data with rotation'):
            var_info = self.var_info(self.test_IMERG_ncfile)
            test_data = np.array(np.arange(12).reshape(4, 3), dtype=np.int64)
            test_data[0, 0] = -99999999
            var_name = '/Grid/HQprecipitation'
            expected_data = np.copy(test_data.T).astype(np.float64)

            actual_data = rs._prepare_data_plane(
                test_data, var_info, var_name, cast_to=np.float64
            )

            self.assertEqual(np.float64, actual_data.dtype)
            np.testing.assert_equal(expected_data, actual_data)

    def test_get_bound_var(self):
        var_info = self.var_info(self.test_IMERG_ncfile)
        expected_bounds = 'lon_bnds'

        actual_bounds = rs._get_bounds_var(var_info, '/Grid/lon')
        self.assertEqual(expected_bounds, actual_bounds)

    def test_copy_resampled_bounds_variable(self):
        target_file = self._generate_test_file()
        target_area = self._generate_test_area()
        var_info = self.var_info(self.test_IMERG_ncfile)
        bnds_var = '/Grid/lat_bnds'
        var_copied = None

        expected_lat_bnds = np.array(
            [
                target_area.projection_y_coords + 0.5,
                target_area.projection_y_coords - 0.5,
            ]
        ).T

        with (
            Dataset(self.test_IMERG_ncfile, mode='r') as source_ds,
            Dataset(target_file, mode='w') as target_ds,
        ):
            rs._transfer_dimensions(source_ds, target_ds, target_area, var_info)

            var_copied = rs._copy_resampled_bounds_variable(
                source_ds, target_ds, bnds_var, target_area, var_info
            )

        self.assertEqual({bnds_var}, var_copied)
        with Dataset(target_file, mode='r') as validate:
            assert_array_equal(expected_lat_bnds, validate['Grid']['lat_bnds'][:])

    def test_resampled_dimension_variable_names(self):
        with self.subTest('root level dimensions'):
            var_info = self.var_info(self.test_1D_dimensions_ncfile)
            expected_resampled = {'/lon', '/lat'}

            actual_resampled = rs._resampled_dimension_variable_names(var_info)
            self.assertEqual(expected_resampled, actual_resampled)

        with self.subTest('grouped dimensions'):
            var_info = self.var_info(self.test_IMERG_ncfile)
            expected_resampled = {
                '/Grid/lon',
                '/Grid/lat',
                '/Grid/lon_bnds',
                '/Grid/lat_bnds',
            }

            actual_resampled = rs._resampled_dimension_variable_names(var_info)
            self.assertEqual(expected_resampled, actual_resampled)

    def test_multiple_resampled_dimension_variable_names(self):
        var_info = self.var_info(self.test_ATL14_ncfile)
        expected_resampled = {'/x', '/y', '/tile_stats/x', '/tile_stats/y'}

        actual_resampled = rs._resampled_dimension_variable_names(var_info)
        self.assertEqual(expected_resampled, actual_resampled)

    def test_crs_variable_name(self):
        with self.subTest('multiple grids, separate groups'):
            dim_pair = ('/Grid/lat', '/Grid/lon')
            dim_pairs = [
                ('/Grid/lat', '/Grid/lon'),
                ('/Grid2/lat', '/Grid2/lon'),
                ('/Grid3/lat', '/Grid3/lon'),
            ]

            expected_crs_name = '/Grid/crs'
            actual_crs_name = rs._crs_variable_name(dim_pair, dim_pairs)
            self.assertEqual(expected_crs_name, actual_crs_name)

        with self.subTest('single grid'):
            dim_pair = ('/lat', '/lon')
            dim_pairs = [('/lat', '/lon')]
            expected_crs_name = '/crs'

            actual_crs_name = rs._crs_variable_name(dim_pair, dim_pairs)
            self.assertEqual(expected_crs_name, actual_crs_name)

        with self.subTest('multiple grids share group'):
            dim_pair = ('/global_grid_lat', '/global_grid_lon')
            dim_pairs = [
                ('/npolar_grid_lat', '/npolar_grid_lon'),
                ('/global_grid_lat', '/global_grid_lon'),
                ('/spolar_grid_lat', '/spolar_grid_lon'),
            ]

            expected_crs_name = '/crs_global_grid_lat_global_grid_lon'
            actual_crs_name = rs._crs_variable_name(dim_pair, dim_pairs)
            self.assertEqual(expected_crs_name, actual_crs_name)

    def test_transfer_metadata(self):
        """Tests to ensure root and group level metadata is transfered to target."""
        _generate_test_file = self._generate_test_file()

        # metadata Set in the test 1D file
        expected_root_metadata = {
            'root-attribute1': 'value1',
            'root-attribute2': 'value2',
        }
        expected_root_groups = {'level1-nested1', 'level1-nested2'}
        expected_nested_metadata = {'level2-nested1': 'level2-nested1-value1'}

        with (
            Dataset(self.test_1D_dimensions_ncfile, mode='r') as source_ds,
            Dataset(_generate_test_file, mode='w') as target_ds,
        ):
            rs._transfer_metadata(source_ds, target_ds)

        with Dataset(_generate_test_file, mode='r') as validate:
            root_metadata = {
                attr: validate.getncattr(attr) for attr in validate.ncattrs()
            }
            root_groups = set(validate.groups.keys())
            nested_group = validate['/level1-nested1/level2-nested1']
            nested_metadata = {
                attr: nested_group.getncattr(attr) for attr in nested_group.ncattrs()
            }

            self.assertSetEqual(expected_root_groups, root_groups)
            self.assertDictEqual(expected_root_metadata, root_metadata)
            self.assertDictEqual(expected_nested_metadata, nested_metadata)

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
        _generate_test_area = AreaDefinition(
            'test_id',
            'test area definition',
            None,
            projection,
            width,
            height,
            area_extent,
        )
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        target_file = self._generate_test_file()

        with (
            Dataset(self.test_1D_dimensions_ncfile, mode='r') as source_ds,
            Dataset(target_file, mode='w') as target_ds,
        ):
            rs._transfer_dimensions(source_ds, target_ds, _generate_test_area, var_info)

        with Dataset(target_file, mode='r') as validate:
            self.assertEqual(validate.dimensions['bnds'].size, 2)
            self.assertEqual(validate.dimensions['time'].size, 0)
            self.assertTrue(validate.dimensions['time'].isunlimited())
            self.assertEqual(validate.dimensions['lon'].size, width)
            self.assertEqual(validate.dimensions['lat'].size, height)

    def test_clone_dimensions(self):
        target_file = self._generate_test_file()
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        projection = '+proj=longlat +datum=WGS84 +no_defs +type=crs'
        width = 36
        height = 18
        area_extent = (-180, -90, 180, 90)
        _generate_test_area = AreaDefinition(
            'test_id',
            'test area definition',
            None,
            projection,
            width,
            height,
            area_extent,
        )
        copy_vars = {'/time', '/time_bnds'}
        with (
            Dataset(self.test_1D_dimensions_ncfile, mode='r') as source_ds,
            Dataset(target_file, mode='w') as target_ds,
        ):
            rs._transfer_dimensions(source_ds, target_ds, _generate_test_area, var_info)

            copied = rs._clone_dimensions(source_ds, target_ds, copy_vars)

            self.assertEqual(copy_vars, copied)

            with Dataset(target_file, mode='r') as validate:
                assert_array_equal(validate['time_bnds'], source_ds['time_bnds'])
                assert_array_equal(validate['time'], source_ds['time'])

    def test_create_resampled_dimensions(self):
        with self.subTest('root level dimensions'):
            var_info = self.var_info(self.test_1D_dimensions_ncfile)
            projection = '+proj=longlat +datum=WGS84 +no_defs +type=crs'
            width = 36
            height = 18
            area_extent = (-180, -90, 180, 90)
            _generate_test_area = AreaDefinition(
                'test_id',
                'test area definition',
                None,
                projection,
                width,
                height,
                area_extent,
            )
            target_file = self._generate_test_file()

            with Dataset(target_file, mode='w') as target_ds:
                rs._create_resampled_dimensions(
                    [('/lat', '/lon')], target_ds, _generate_test_area, var_info
                )

            with Dataset(target_file, mode='r') as validate:
                self.assertEqual(validate.dimensions['lat'].size, 18)
                self.assertEqual(validate.dimensions['lon'].size, 36)

        with self.subTest('Group level dimensions'):
            var_info = self.var_info(self.test_IMERG_ncfile)
            projection = '+proj=longlat +datum=WGS84 +no_defs +type=crs'
            width = 360
            height = 180
            area_extent = (-180, -90, 180, 90)
            _generate_test_area = AreaDefinition(
                'test_id',
                'test area definition',
                None,
                projection,
                width,
                height,
                area_extent,
            )
            target_file = self._generate_test_file()
            with Dataset(target_file, mode='w') as target_ds:
                rs._create_resampled_dimensions(
                    [('/Grid/lon', '/Grid/lat')],
                    target_ds,
                    _generate_test_area,
                    var_info,
                )

            with Dataset(target_file, mode='r') as validate:
                self.assertEqual(validate['Grid'].dimensions['lat'].size, 180)
                self.assertEqual(validate['Grid'].dimensions['lon'].size, 360)

    def test_resampler_kwargs(self):
        with self.subTest('floating data'):
            data = np.array([1.0, 2.0, 3.0, 4.0], dtype='float')
            expected_args = {'rows_per_scan': 2}
            actual_args = rs._resampler_kwargs(data, 'float')
            self.assertDictEqual(expected_args, actual_args)

        with self.subTest('all rows needed'):
            data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype='float')
            expected_args = {'rows_per_scan': 7}
            actual_args = rs._resampler_kwargs(data, 'float')
            self.assertDictEqual(expected_args, actual_args)

        with self.subTest('integer data'):
            data = np.ma.array([1, 2, 3], dtype='int16')
            expected_args = {
                'rows_per_scan': 3,
                'maximum_weight_mode': True,
            }
            actual_args = rs._resampler_kwargs(data, 'int16')
            self.assertDictEqual(expected_args, actual_args)

    def test_write_grid_mappings(self):
        target_file = self._generate_test_file()
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        _generate_test_area = self._generate_test_area()
        expected_crs_map = {('/lon', '/lat'): '/crs'}

        with (
            Dataset(self.test_1D_dimensions_ncfile, mode='r') as source_ds,
            Dataset(target_file, mode='w') as target_ds,
        ):
            rs._transfer_metadata(source_ds, target_ds)
            rs._transfer_dimensions(source_ds, target_ds, _generate_test_area, var_info)

            actual_crs_map = rs._write_grid_mappings(
                target_ds, rs._resampled_dimension_pairs(var_info), _generate_test_area
            )
            self.assertDictEqual(expected_crs_map, actual_crs_map)

        with Dataset(target_file, mode='r') as validate:
            crs = rs._get_variable(validate, '/crs')
            expected_crs_metadata = _generate_test_area.crs.to_cf()

            actual_crs_metadata = {attr: crs.getncattr(attr) for attr in crs.ncattrs()}

            self.assertDictEqual(expected_crs_metadata, actual_crs_metadata)

    def test_get_variable(self):
        with Dataset(self.test_ATL14_ncfile, mode='r') as source_ds:
            var_grouped = rs._get_variable(source_ds, '/tile_stats/RMS_data')
            expected_grouped = source_ds['tile_stats'].variables['RMS_data']
            self.assertEqual(expected_grouped, var_grouped)

            var_flat = rs._get_variable(source_ds, '/ice_area')
            expected_flat = source_ds.variables['ice_area']
            self.assertEqual(expected_flat, var_flat)

    def test_create_dimension(self):
        name = '/somedim'
        size = 1000
        with Dataset(self._generate_test_file(), mode='w') as target_ds:
            dim = rs._create_dimension(target_ds, name, size)
            self.assertTrue(type(dim), Variable)
            self.assertEqual(dim.size, size)
            self.assertEqual(dim.name, 'somedim')

    def test_create_nested_dimension(self):
        name = '/some/deeply/nested/dimname'
        size = 2000
        with Dataset(self._generate_test_file(), mode='w') as target_ds:
            dim = rs._create_dimension(target_ds, name, size)
            self.assertTrue(type(dim), Variable)
            self.assertEqual(dim.size, size)
            self.assertEqual(dim.name, 'dimname')

    def test_get_flat_dimension(self):
        with Dataset(self.test_1D_dimensions_ncfile, mode='r') as source_ds:
            lat_dim = rs._get_dimension(source_ds, '/lat')
            self.assertTrue(type(lat_dim), Variable)
            self.assertTrue(lat_dim.size, len(self.latitudes))
            self.assertTrue(lat_dim.name, 'lat')

    def test_get_nested_dimension(self):
        with Dataset(self.test_IMERG_ncfile, mode='r') as source_ds:
            lat_dim = rs._get_dimension(source_ds, '/Grid/lat')
            self.assertTrue(type(lat_dim), Variable)
            self.assertTrue(lat_dim.name, 'lat')
            self.assertTrue(lat_dim.size, 1800)

    def test_copy_dimension(self):
        with (
            Dataset(self._generate_test_file(), mode='w') as target_ds,
            Dataset(self.test_1D_dimensions_ncfile, mode='r') as source_ds,
        ):
            time_dimension = rs._copy_dimension('/time', source_ds, target_ds)
            self.assertTrue(time_dimension.isunlimited())
            self.assertEqual(time_dimension.size, 0)

            lon_dimension = rs._copy_dimension('/lon', source_ds, target_ds)
            self.assertFalse(lon_dimension.isunlimited())
            self.assertEqual(lon_dimension.size, len(self.longitudes))

    def test_copy_dimensions(self):
        test_target = self._generate_test_file()
        with (
            Dataset(test_target, mode='w') as target_ds,
            Dataset(self.test_1D_dimensions_ncfile, mode='r') as source_ds,
        ):
            rs._copy_dimensions(
                {'/lat', '/lon', '/time', '/bnds'}, source_ds, target_ds
            )

        with Dataset(test_target, mode='r') as validate:
            self.assertTrue(validate.dimensions['time'].isunlimited())
            self.assertEqual(validate.dimensions['time'].size, 0)
            self.assertEqual(validate.dimensions['lat'].size, len(self.latitudes))
            self.assertEqual(validate.dimensions['lon'].size, len(self.longitudes))
            self.assertEqual(validate.dimensions['bnds'].size, 2)

    def test_copy_dimensions_with_groups(self):
        test_target = self._generate_test_file()
        with (
            Dataset(test_target, mode='w') as target_ds,
            Dataset(self.test_IMERG_ncfile, mode='r') as source_ds,
        ):
            rs._copy_dimensions(
                {'/Grid/latv', '/Grid/lonv', '/Grid/nv', '/Grid/time'},
                source_ds,
                target_ds,
            )

        with Dataset(test_target, mode='r') as validate:
            self.assertTrue(validate['Grid'].dimensions['time'].isunlimited())
            self.assertEqual(validate['Grid'].dimensions['time'].size, 0)
            self.assertEqual(validate['Grid'].dimensions['lonv'].size, 2)
            self.assertEqual(validate['Grid'].dimensions['latv'].size, 2)
            self.assertEqual(validate['Grid'].dimensions['nv'].size, 2)

    def test_horizontal_dims_for_variable_grouped(self):
        var_info = self.var_info(self.test_IMERG_ncfile)
        expected_dims = ('/Grid/lon', '/Grid/lat')
        actual_dims = rs._horizontal_dims_for_variable(
            var_info, '/Grid/IRkalmanFilterWeight'
        )
        self.assertEqual(expected_dims, actual_dims)

    def test_horizontal_dims_for_variable(self):
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        expected_dims = ('/lon', '/lat')
        actual_dims = rs._horizontal_dims_for_variable(var_info, '/data')
        self.assertEqual(expected_dims, actual_dims)

    def test_horizontal_dims_for_missing_variable(self):
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        expected_dims = None
        actual_dims = rs._horizontal_dims_for_variable(var_info, '/missing')
        self.assertEqual(expected_dims, actual_dims)

    def test_resampled_dimenension_pairs(self):
        with self.subTest('1d file'):
            var_info = self.var_info(self.test_1D_dimensions_ncfile)
            expected_pairs = [('/lon', '/lat')]
            actual_pairs = rs._resampled_dimension_pairs(var_info)
            self.assertEqual(expected_pairs, actual_pairs)

        with self.subTest('multiple horizontal pairs.'):
            var_info = self.var_info(self.test_ATL14_ncfile)
            expected_pairs = [('/y', '/x'), ('/tile_stats/y', '/tile_stats/x')]
            actual_pairs = rs._resampled_dimension_pairs(var_info)
            self.assertEqual(set(expected_pairs), set(actual_pairs))

    def test_all_dimensions(self):
        var_info = self.var_info(self.test_1D_dimensions_ncfile)
        expected_dimensions = {'/time', '/lon', '/lat', '/bnds'}
        actual_dimensions = rs._all_dimensions(var_info)
        self.assertEqual(expected_dimensions, actual_dimensions)

    def test_unresampled_variables(self):
        with self.subTest('flat ungrouped'):
            var_info = self.var_info(self.test_1D_dimensions_ncfile)
            expected_vars = {'/time', '/time_bnds'}
            actual_vars = rs._unresampled_variables(var_info)
            self.assertEqual(expected_vars, actual_vars)

        with self.subTest('IMERG grouped'):
            var_info = self.var_info(self.test_IMERG_ncfile)

            expected_vars = {'/Grid/time', '/Grid/time_bnds'}
            actual_vars = rs._unresampled_variables(var_info)
            self.assertEqual(expected_vars, actual_vars)

        with self.subTest('MERRA2 includes levels'):
            var_info = self.var_info(self.test_MERRA2_ncfile)

            expected_vars = {'/lev', '/time'}
            actual_vars = rs._unresampled_variables(var_info)
            self.assertEqual(expected_vars, actual_vars)

        with self.subTest('ATL14 lots of deep group vars'):
            var_info = self.var_info(self.test_ATL14_ncfile)

            expected_vars = {
                '/Polar_Stereographic',
                '/orbit_info/bounding_polygon_dim1',
                '/orbit_info/bounding_polygon_lat1',
                '/orbit_info/bounding_polygon_lon1',
                '/quality_assessment/qa_granule_fail_reason',
                '/quality_assessment/qa_granule_pass_fail',
            }
            actual_vars = rs._unresampled_variables(var_info)
            self.assertEqual(expected_vars, actual_vars)

    def test_all_dimension_variables(self):
        with self.subTest('1D file'):
            var_info = self.var_info(self.test_1D_dimensions_ncfile)
            expected_vars = {'/lat', '/lon', '/time'}
            actual_vars = rs._all_dimension_variables(var_info)
            self.assertEqual(expected_vars, actual_vars)

        with self.subTest('2D file'):
            var_info = self.var_info(self.test_2D_dimensions_ncfile)
            expected_vars = {'/lat', '/lon'}
            actual_vars = rs._all_dimension_variables(var_info)
            self.assertEqual(expected_vars, actual_vars)

    def test_resampled_dimensions(self):
        with self.subTest('1D file'):
            var_info = self.var_info(self.test_1D_dimensions_ncfile)
            expected_dimensions = {'/lat', '/lon'}
            actual_dimensions = rs._resampled_dimensions(var_info)
            self.assertEqual(expected_dimensions, actual_dimensions)

        with self.subTest('ATL14: multiple grids'):
            var_info = self.var_info(self.test_ATL14_ncfile)

            expected_dimensions = {'/x', '/y', '/tile_stats/x', '/tile_stats/y'}
            actual_dimensions = rs._resampled_dimensions(var_info)
            self.assertEqual(expected_dimensions, actual_dimensions)

    def test_needs_rotation(self):
        with self.subTest('needs rotation'):
            var_info = self.var_info(self.test_1D_dimensions_ncfile)
            self.assertTrue(rs._needs_rotation(var_info, '/data'))

        with self.subTest('no rotation'):
            var_info = self.var_info(self.test_MERRA2_ncfile)
            self.assertFalse(rs._needs_rotation(var_info, '/PHIS'))
            self.assertFalse(rs._needs_rotation(var_info, '/OMEGA'))

    def test_validate_remaining_variables(self):
        with self.subTest('success'):
            test_vars = {
                2: {'some', '2d', 'vars'},
                3: {'more', 'cubes'},
                4: {'hypercube', 'data'},
            }
            self.assertEqual(rs._validate_remaining_variables(test_vars), None)

        with self.subTest('failure'):
            test_vars = {
                1: {'1d', 'should', 'have been', 'processed'},
                2: {'some', '2d', 'vars'},
                3: {'more', 'cubes'},
                4: {'hypercube', 'data'},
            }
            with self.assertRaisesRegex(
                RegridderException, 'Variables with dimensions.*cannot be handled.'
            ):
                rs._validate_remaining_variables(test_vars)

    def test_integer_like(self):
        int_types = [
            np.byte,
            np.ubyte,
            np.short,
            np.ushort,
            np.intc,
            np.uintc,
            np.int_,
            np.uint,
            np.longlong,
            np.ulonglong,
        ]

        other_types = [np.float16, np.float32, np.float64]

        for _type in int_types:
            with self.subTest(_type):
                self.assertTrue(rs._integer_like(_type))

        for _type in other_types:
            with self.subTest(_type):
                self.assertFalse(rs._integer_like(_type))

        with self.subTest('string'):
            self.assertFalse(rs._integer_like(str))

    @patch(
        'harmony_regridding_service.regridding_service.AreaDefinition',
        wraps=AreaDefinition,
    )
    def test_compute_target_area(self, mock_area):
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

        actual_area = rs._compute_target_area(message)

        self.assertEqual(actual_area.shape, (expected_height, expected_width))
        self.assertEqual(actual_area.shape, (expected_height, expected_width))
        self.assertEqual(actual_area.area_extent, (xmin, ymin, xmax, ymax))
        self.assertEqual(actual_area.proj_str, crs)
        mock_area.assert_called_once_with(
            'target_area_id',
            'target area definition',
            None,
            crs,
            expected_width,
            expected_height,
            (xmin, ymin, xmax, ymax),
        )

    def test_grid_height(self):
        with self.subTest('message with scale size'):
            expected_grid_height = 50
            actual_grid_height = rs._grid_height(self.test_message_with_scale_size)
            self.assertEqual(expected_grid_height, actual_grid_height)

        with self.subTest('mesage includes height'):
            expected_grid_height = 80
            actual_grid_height = rs._grid_height(self.test_message_with_height_width)
            self.assertEqual(expected_grid_height, actual_grid_height)

    def test_grid_width(self):
        with self.subTest('message with scale size'):
            expected_grid_width = 100
            actual_grid_width = rs._grid_width(self.test_message_with_scale_size)
            self.assertEqual(expected_grid_width, actual_grid_width)

        with self.subTest('message with width'):
            expected_grid_width = 40
            actual_grid_width = rs._grid_width(self.test_message_with_height_width)
            self.assertEqual(expected_grid_width, actual_grid_width)

    def test_compute_num_elements(self):
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
        actual_x_elements = rs._compute_num_elements(message, 'x')
        actual_y_elements = rs._compute_num_elements(message, 'y')

        self.assertEqual(expected_x_elements, actual_x_elements)
        self.assertEqual(expected_y_elements, actual_y_elements)

    def test_is_projection_dim(self):
        with self.subTest('test valid x'):
            var_info = self.var_info(self.test_1D_dimensions_ncfile)
            self.assertTrue(rs._is_horizontal_dim('/lon', var_info))

        with self.subTest('test invalid x'):
            var_info = self.var_info(self.test_1D_dimensions_ncfile)
            self.assertFalse(rs._is_horizontal_dim('/lat', var_info))

        with self.subTest('test valid y'):
            var_info = self.var_info(self.test_1D_dimensions_ncfile)
            self.assertTrue(rs._is_vertical_dim('/lat', var_info))

        with self.subTest('test invalid y'):
            var_info = self.var_info(self.test_1D_dimensions_ncfile)
            self.assertFalse(rs._is_vertical_dim('/lon', var_info))

    def test_get_projection_dims(self):
        with self.subTest('x dims'):
            var_info = self.var_info(self.test_1D_dimensions_ncfile)
            dims = ('/lat', '/lon')
            expected_dim = ['/lon']

            actual = rs._get_horizontal_dims(dims, var_info)
            self.assertEqual(expected_dim, actual)

        with self.subTest('y dims'):
            var_info = self.var_info(self.test_1D_dimensions_ncfile)
            dims = ('/lat', '/lon')
            expected_dim = ['/lat']

            actual = rs._get_vertical_dims(dims, var_info)
            self.assertEqual(expected_dim, actual)

        with self.subTest('y dims no variables'):
            var_info = self.var_info(self.test_1D_dimensions_ncfile)
            dims = ('/baddim1', '/baddim2')

            expected_dims = []
            actual_dims = rs._get_vertical_dims(dims, var_info)
            self.assertEqual(expected_dims, actual_dims)

        with self.subTest('x dims no variables'):
            var_info = self.var_info(self.test_1D_dimensions_ncfile)
            dims = ('/baddim1', '/baddim2')
            expected_dims = []
            actual_dims = rs._get_horizontal_dims(dims, var_info)
            self.assertEqual(expected_dims, actual_dims)

        with self.subTest('x dims with bad variable'):
            var_info = self.var_info(self.test_1D_dimensions_ncfile)
            dims = ('/baddim1', '/lon')
            expected_dim = ['/lon']

            actual_dim = rs._get_horizontal_dims(dims, var_info)
            self.assertEqual(expected_dim, actual_dim)

        with self.subTest('y dims multiple values'):
            var_info = self.var_info(self.test_1D_dimensions_ncfile)
            dims = ('/lat', '/lon', '/lat', '/ba')
            expected_dim = ['/lat', '/lat']

            actual = rs._get_vertical_dims(dims, var_info)
            self.assertEqual(expected_dim, actual)

    def test_expected_result_compute_horizontal_source_grids(self):
        """Exercises the single function for computing horizontal grids."""
        var_info = self.var_info(self.test_1D_dimensions_ncfile)

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

        test_args = [('/lon', '/lat'), ('/lat', '/lon')]

        for grid_dimensions in test_args:
            with self.subTest(f'independent grid_dimension order {grid_dimensions}'):
                longitudes, latitudes = rs._compute_horizontal_source_grids(
                    grid_dimensions, self.test_1D_dimensions_ncfile, var_info
                )

                np.testing.assert_array_equal(expected_latitudes, latitudes)
                np.testing.assert_array_equal(expected_longitudes, longitudes)

    def test_2D_lat_lon_input_compute_horizontal_source_grids(self):
        var_info = self.var_info(self.test_2D_dimensions_ncfile)
        grid_dimensions = ('/lat', '/lon')

        expected_regex = re.escape(
            'Incorrect source data dimensions. rows:(6, 5), columns:(6, 5)'
        )
        with self.assertRaisesRegex(InvalidSourceDimensions, expected_regex):
            rs._compute_horizontal_source_grids(
                grid_dimensions, self.test_2D_dimensions_ncfile, var_info
            )


@pytest.mark.parametrize(
    'input_value, expected, description',
    [
        (1, 1, 'number less than 2'),
        (4, 2, 'even composite number'),
        (9, 3, 'odd composite number'),
        (7, 7, 'prime number'),
    ],
)
def test__get_rows_per_scan(input_value, expected, description):
    """Test _get_rows_per_scan with various input types."""
    assert rs._get_rows_per_scan(input_value) == expected, f'Failed for {description}'


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


# Use the fixture from conftest.py
def test_regrid_projected_data_end_to_end(smap_projected_netcdf_file, tmp_path):
    """Test the full regrid process for projected input data."""
    input_filename = str(smap_projected_netcdf_file)
    output_filename = str(tmp_path / 'regridded_output.nc')
    logger_mock = MagicMock()

    # Define a target CRS and grid (example: Geographic WGS84)
    params = {
        'format': {
            'mime': 'application/x-netcdf',
            'crs': 'EPSG:4326',
            'width': 100,
            'height': 50,
            'scaleExtent': {
                'x': {'min': -180, 'max': 180},
                'y': {'min': -90, 'max': 90},
            },
        },
        'sources': [{'collection': 'C123-TEST', 'shortName': 'SPL4SMAU'}],
    }
    message = HarmonyMessage(params)
    source = HarmonySource({'collection': 'C123-TEST', 'shortName': 'SPL4SMAU'})

    # Mock generate_output_filename to control the output path
    with patch(
        'harmony_regridding_service.regridding_service.generate_output_filename',
        return_value=output_filename,
    ):
        result_filename = rs.regrid(message, input_filename, source, logger_mock)

    assert result_filename == output_filename
    assert Path(output_filename).exists()

    with xr.open_datatree(output_filename) as ds_out:
        assert 'crs' in ds_out

        assert ds_out.dims['y'] == 50
        assert ds_out.dims['x'] == 100

        assert 'sm_profile_forecast' in ds_out['Forecast_Data']
        assert 'sm_profile_analysis' in ds_out['Analysis_Data']
        assert 'tb_v_obs' in ds_out['Observations_Data']

        assert (
            ds_out['/Metadata/DatasetIdentification'].attrs['shortName'] == 'SPL4SMAU'
        )

        assert (
            ds_out['Observations_Data/tb_v_obs'].attrs['long_name']
            == 'Composite resolution observed (L2_SM_AP or L1C_TB) V-pol ...'
        )
