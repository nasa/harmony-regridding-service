from unittest import TestCase

from netCDF4 import Dataset
import numpy as np
from logging import getLogger
from pathlib import Path
import re
from tempfile import mkdtemp
from shutil import rmtree

from varinfo import VarInfoFromNetCDF4

from harmony_regridding_service.regridding_service import _compute_horizontal_source_grids
from harmony_regridding_service.exceptions import InvalidSourceDimensions


class TestRegriddingService(TestCase):
    """Test the regridding_service module."""

    @classmethod
    def setUpClass(cls):
        pass


class Test_ComputeHorizontalSourceGrids(TestCase):
    """Exercises the single function for computing horizontal grids."""

    @classmethod
    def setUpClass(cls):
        """fixtures for all class tests."""
        cls.tmp_dir = mkdtemp()
        cls.test_ncfile = Path(cls.tmp_dir, 'valid_test.nc')
        cls.bad_ncfile = Path(cls.tmp_dir, 'invalid_test.nc')
        cls.logger = getLogger()

        longitudes = np.array([-180, -80, -45, 45, 80, 180])
        latitudes = np.array([90, 45, 0, -46, -89])

        dataset = Dataset(cls.test_ncfile, 'w')
        dataset.createDimension('lon', size=len(longitudes))
        dataset.createDimension('lat', size=len(latitudes))
        dataset.createVariable('/lon', longitudes.dtype, dimensions=('lon'))
        dataset.createVariable('/lat', latitudes.dtype, dimensions=('lat'))
        dataset['lon'].setncattr('units', 'degrees_east')
        dataset['lat'].setncattr('units', 'degrees_north')
        dataset['lat'][:] = latitudes
        dataset['lon'][:] = longitudes
        dataset.close()

        dataset = Dataset(cls.bad_ncfile, 'w')
        dataset.createDimension('lon', size=(len(longitudes)))
        dataset.createDimension('lat', size=(len(latitudes)))
        dataset.createVariable('/lon', longitudes.dtype, dimensions=('lon', 'lat'))
        dataset.createVariable('/lat', latitudes.dtype, dimensions=('lon', 'lat'))
        dataset['lon'].setncattr('units', 'degrees_east')
        dataset['lat'].setncattr('units', 'degrees_north')
        dataset['lat'][:] = np.broadcast_to(latitudes, (6, 5))
        dataset['lon'][:] = np.broadcast_to(longitudes, (5,6)).T
        dataset.close()


    @classmethod
    def tearDownCass(cls):
        rmtree(cls.tmp_dir)

    def test_expected_result(self):
        var_info = VarInfoFromNetCDF4(self.test_ncfile, self.logger)

        expected_longitudes = np.array([
            [-180,  -80,  -45,   45,   80,  180],
            [-180,  -80,  -45,   45,   80,  180],
            [-180,  -80,  -45,   45,   80,  180],
            [-180,  -80,  -45,   45,   80,  180],
            [-180,  -80,  -45,   45,   80,  180]])

        expected_latitudes = np.array([
            [ 90,  90,  90,  90,  90,  90],
            [ 45,  45,  45,  45,  45,  45],
            [  0,   0,   0,   0,   0,   0],
            [-46, -46, -46, -46, -46, -46],
            [-89, -89, -89, -89, -89, -89]])


        test_args = [
            ('/lon', '/lat'),
            ('/lat', '/lon')]

        for grid_dimensions in test_args:
            with self.subTest(f'independent grid_dimension order {grid_dimensions}'):
                longitudes, latitudes = _compute_horizontal_source_grids(
                    grid_dimensions, self.test_ncfile, var_info)

                np.testing.assert_array_equal(expected_latitudes, latitudes)
                np.testing.assert_array_equal(expected_longitudes, longitudes)

    def test_gridded_lat_lons(self):
        var_info = VarInfoFromNetCDF4(self.bad_ncfile, self.logger)
        grid_dimensions = ('/lat', '/lon')

        expected_regex = re.escape('rows:(6, 5), columns:(6, 5)')
        with self.assertRaisesRegex(InvalidSourceDimensions, expected_regex):
            _compute_horizontal_source_grids(
                grid_dimensions, self.bad_ncfile, var_info)
