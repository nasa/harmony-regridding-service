"""Test the variable utitity module."""

import numpy as np
import pytest
from netCDF4 import Dataset
from varinfo import VarInfoFromNetCDF4

from harmony_regridding_service.regridding_service import HRS_VARINFO_CONFIG_FILENAME
from harmony_regridding_service.var_utilitities import (
    get_unprocessable_variables,
    is_excluded_science_variable,
    is_string_variable,
)


@pytest.fixture
def test_variables_nc_file(tmp_path):
    test_nc_filename = tmp_path / 'test.nc4'

    with Dataset(str(test_nc_filename), 'w') as dataset:
        dataset.createDimension('lat', size=2)
        dataset.createDimension('lon', size=2)

        dataset.createVariable(
            '/numeric_variable', np.float64, dimensions=('lat', 'lon')
        )
        dataset.createVariable('/string_variable', str, dimensions=('lat', 'lon'))

        dataset.createVariable('/nuked_time_utc_str', 'S24', dimensions=('lat', 'lon'))

        dataset.createVariable(
            '/nuked_time_utc_float', np.float64, dimensions=('lat', 'lon')
        )

    yield str(test_nc_filename)


def test_is_string_variable(test_variables_nc_file):
    var_info = VarInfoFromNetCDF4(test_variables_nc_file)

    assert is_string_variable(var_info, '/string_variable')
    assert is_string_variable(var_info, '/nuked_time_utc_str')
    assert not is_string_variable(var_info, '/numeric_variable')
    assert not is_string_variable(var_info, '/nuked_time_utc_float')


def test_is_excluded_science_var(test_variables_nc_file):
    var_info = VarInfoFromNetCDF4(
        test_variables_nc_file,
        short_name='SPL3TEST',
        config_file=HRS_VARINFO_CONFIG_FILENAME,
    )

    assert not is_excluded_science_variable(var_info, '/string_variable')
    assert not is_excluded_science_variable(var_info, '/numeric_variable')
    assert is_excluded_science_variable(var_info, '/nuked_time_utc_str')
    assert is_excluded_science_variable(var_info, '/nuked_time_utc_float')


def test_get_unprocessed_variables(test_variables_nc_file):
    var_info = VarInfoFromNetCDF4(
        test_variables_nc_file,
        short_name='SPL3TEST',
        config_file=HRS_VARINFO_CONFIG_FILENAME,
    )

    assert {
        '/string_variable',
        '/nuked_time_utc_str',
        '/nuked_time_utc_float',
    } == get_unprocessable_variables(var_info, var_info.get_all_variables())
