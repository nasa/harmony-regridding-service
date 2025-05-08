"""Tests regridding service."""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
import xarray as xr
from harmony_service_lib.message import Message as HarmonyMessage
from harmony_service_lib.message import Source as HarmonySource
from netCDF4 import Dataset, Dimension
from numpy.testing import assert_array_equal
from pyproj import CRS
from pyresample.geometry import AreaDefinition
from varinfo import VarInfoFromNetCDF4

from harmony_regridding_service.exceptions import (
    InvalidSourceCRS,
    InvalidSourceDimensions,
    SourceDataError,
)
from harmony_regridding_service.regridding_service import (
    HRS_VARINFO_CONFIG_FILENAME,
    _all_dimension_variables,
    _all_dimensions,
    _clone_variables,
    _compute_area_extent_from_regular_x_y_coords,
    _compute_array_bounds,
    _compute_horizontal_source_grids,
    _compute_num_elements,
    _compute_projected_horizontal_source_grids,
    _compute_target_area,
    _copy_1d_dimension_variables,
    _copy_dimension,
    _copy_dimension_variables,
    _copy_dimensions,
    _copy_resampled_bounds_variable,
    _copy_var_with_attrs,
    _copy_var_without_metadata,
    _create_dimension,
    _create_resampled_dimensions,
    _crs_from_source_data,
    _crs_variable_name,
    _dims_are_lon_lat,
    _dims_are_projected_x_y,
    _get_bounds_var,
    _get_dimension,
    _get_horizontal_dims,
    _get_rows_per_scan,
    _get_variable,
    _get_vertical_dims,
    _grid_height,
    _grid_width,
    _horizontal_dims_for_variable,
    _integer_like,
    _is_horizontal_dim,
    _is_vertical_dim,
    _needs_rotation,
    _prepare_data_plane,
    _resample_layer,
    _resampled_dimension_pairs,
    _resampled_dimension_variable_names,
    _resampled_dimensions,
    _resampler_kwargs,
    _transfer_dimensions,
    _transfer_metadata,
    _unresampled_variables,
    _walk_groups,
    _write_grid_mappings,
    regrid,
)


## pytest fixtures
@pytest.fixture(scope='session')
def test_fixtures_dir():
    """Return path to the test fixtures directory."""
    return Path(Path(__file__).parent, 'fixtures')


@pytest.fixture(scope='session')
def test_ATL14_ncfile(test_fixtures_dir):
    """Return path to the ATL14 test file."""
    return Path(test_fixtures_dir, 'empty-ATL14.nc')


@pytest.fixture(scope='session')
def test_MERRA2_ncfile(test_fixtures_dir):
    """Return path to the MERRA2 test file."""
    return Path(test_fixtures_dir, 'empty-MERRA2.nc')


@pytest.fixture(scope='session')
def test_IMERG_ncfile(test_fixtures_dir):
    """Return path to the IMERG test file."""
    return Path(test_fixtures_dir, 'empty-IMERG.nc')


@pytest.fixture(scope='session')
def longitudes():
    """Return longitudes array used in tests."""
    return np.array([-180, -80, -45, 45, 80, 180], dtype=np.dtype('f8'))


@pytest.fixture(scope='session')
def latitudes():
    """Return latitudes array used in tests."""
    return np.array([90, 45, 0, -46, -89], dtype=np.dtype('f8'))


@pytest.fixture(scope='session')
def test_1D_dimensions_ncfile(tmp_path_factory, longitudes, latitudes):
    """Create and return a test file with 1D /lon and /lat root vars."""
    # overide xarray's import
    from netCDF4 import Dataset

    tmp_dir = tmp_path_factory.mktemp('1d_test')
    test_file = Path(tmp_dir, '1D_test.nc')

    # Set up file with one dimensional /lon and /lat root variables
    dataset = Dataset(test_file, 'w')
    dataset.setncatts({'root-attribute1': 'value1', 'root-attribute2': 'value2'})

    # Set up some groups and metadata
    group1 = dataset.createGroup('/level1-nested1')
    group2 = dataset.createGroup('/level1-nested2')
    group2.setncatts({'level1-nested2': 'level1-nested2-value1'})
    group1.setncatts({'level1-nested1': 'level1-nested1-value1'})
    group3 = group1.createGroup('/level2-nested1')
    group3.setncatts({'level2-nested1': 'level2-nested1-value1'})

    dataset.createDimension('time', size=None)
    dataset.createDimension('lon', size=len(longitudes))
    dataset.createDimension('lat', size=len(latitudes))
    dataset.createDimension('bnds', size=2)

    dataset.createVariable('/lon', longitudes.dtype, dimensions=('lon'))
    dataset.createVariable('/lat', latitudes.dtype, dimensions=('lat'))
    dataset.createVariable('/data', np.dtype('f8'), dimensions=('lon', 'lat'))
    dataset.createVariable('/time', np.dtype('f8'), dimensions=('time'))
    dataset.createVariable('/time_bnds', np.dtype('u2'), dimensions=('time', 'bnds'))

    dataset['lat'][:] = latitudes
    dataset['lon'][:] = longitudes
    dataset['time'][:] = [1.0, 2.0, 3.0, 4.0]
    dataset['data'][:] = np.arange(len(longitudes) * len(latitudes)).reshape(
        (len(longitudes), len(latitudes))
    )
    dataset['time_bnds'][:] = np.array([[0.5, 1.5, 2.5, 3.5], [1.5, 2.5, 3.5, 4.5]]).T

    dataset['lon'].setncattr('units', 'degrees_east')
    dataset['lat'].setncattr('units', 'degrees_north')
    dataset['lat'].setncattr('non-standard-attribute', 'Wont get copied')
    dataset['data'].setncattr('units', 'widgets per month')
    dataset.close()

    return test_file


@pytest.fixture(scope='session')
def test_2D_dimensions_ncfile(tmp_path_factory, longitudes, latitudes):
    """Create and return a test file with 2D dimensions."""
    from netCDF4 import Dataset

    tmp_dir = tmp_path_factory.mktemp('2d_test')
    test_file = Path(tmp_dir, '2D_test.nc')

    # Set up a file with two dimensional /lon and /lat variables.
    dataset = Dataset(test_file, 'w')
    dataset.createDimension('lon', size=(len(longitudes)))
    dataset.createDimension('lat', size=(len(latitudes)))
    dataset.createVariable('/lon', longitudes.dtype, dimensions=('lon', 'lat'))
    dataset.createVariable('/lat', latitudes.dtype, dimensions=('lon', 'lat'))
    dataset['lon'].setncattr('units', 'degrees_east')
    dataset['lat'].setncattr('units', 'degrees_north')
    dataset['lat'][:] = np.broadcast_to(latitudes, (6, 5))
    dataset['lon'][:] = np.broadcast_to(longitudes, (5, 6)).T
    dataset.close()

    return test_file


@pytest.fixture
def test_message_with_scale_size():
    """Create a test Harmony message with scale size."""
    return HarmonyMessage(
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


@pytest.fixture
def test_message_with_height_width():
    """Create a test Harmony message with height and width."""
    return HarmonyMessage(
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


@pytest.fixture
def var_info_fxn():
    """Varinfo fixture factory.

    Returns a function that will create a varinfo instance with the input
    NetCDF filename.
    """

    def _var_info(nc_file: str | Path, short_name: str | None = None):
        return VarInfoFromNetCDF4(
            str(nc_file),
            config_file=HRS_VARINFO_CONFIG_FILENAME,
            short_name=short_name,
        )

    return _var_info


@pytest.fixture
def test_file(tmp_path):
    """Return a temporary target netcdf filename."""
    return Path(tmp_path, f'target_{uuid4()}.nc')


@pytest.fixture
def test_area_fxn():
    """An AreaDefinition factory.

    Returns:
        An AreaDefinition function that can be called with overriden values.
    """

    def _test_area(width=360, height=180, area_extent=(-180, -90, 180, 90)):
        return AreaDefinition(
            'test_id',
            'test area definition',
            None,
            '+proj=longlat +datum=WGS84 +no_defs +type=crs',
            width,
            height,
            area_extent,
        )

    return _test_area


### TESTS
def test__walk_groups(test_file):
    """Demonstrate traversing all groups."""
    target_path = test_file
    groups = ['/a/nested/group', '/b/another/deeper/group2']
    expected_visited = {'a', 'nested', 'group', 'b', 'another', 'deeper', 'group2'}

    with Dataset(target_path, mode='w') as target_ds:
        for group in groups:
            target_ds.createGroup(group)

    actual_visited = set()
    with Dataset(target_path, mode='r') as validate:
        for groups in _walk_groups(validate):
            for group in groups:
                actual_visited.update([group.name])

    assert expected_visited == actual_visited


def test__copy_1d_dimension_variables(
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
        _transfer_dimensions(source_ds, target_ds, target_area, var_info)
        vars_copied = _copy_1d_dimension_variables(
            source_ds, target_ds, dim_var_names, target_area, var_info
        )

    assert dim_var_names == vars_copied
    with Dataset(target_file, mode='r') as validate:
        assert_array_equal(validate['/lon'][:], target_area.projection_x_coords)
        assert_array_equal(validate['/lat'][:], target_area.projection_y_coords)
        assert expected_attributes == set(validate['/lat'].ncattrs())
        with pytest.raises(AttributeError):
            validate['/lat'].getncattr('non-standard-attribute')


def test__copy_vars_without_metadata(
    test_file, test_area_fxn, test_1D_dimensions_ncfile, var_info_fxn
):
    target_file = test_file
    target_area = test_area_fxn()
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    with (
        Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds,
        Dataset(target_file, mode='w') as target_ds,
    ):
        _transfer_dimensions(source_ds, target_ds, target_area, var_info)
        _copy_var_without_metadata(source_ds, target_ds, '/data')

    with Dataset(target_file, mode='r') as validate:
        actual_metadata = {
            attr: validate['/data'].getncattr(attr)
            for attr in validate['/data'].ncattrs()
        }
        assert {} == actual_metadata


def test__copy_var_with_attrs(
    test_file, test_area_fxn, test_1D_dimensions_ncfile, var_info_fxn
):
    target_file = test_file
    target_area = test_area_fxn()
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    expected_metadata = {'units': 'widgets per month'}
    with (
        Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds,
        Dataset(target_file, mode='w') as target_ds,
    ):
        _transfer_dimensions(source_ds, target_ds, target_area, var_info)
        _copy_var_with_attrs(source_ds, target_ds, '/data')

    with Dataset(target_file, mode='r') as validate:
        actual_metadata = {
            attr: validate['/data'].getncattr(attr)
            for attr in validate['/data'].ncattrs()
        }
        assert actual_metadata == expected_metadata


def test__copy_dimension_variables(
    test_file,
    test_area_fxn,
    test_MERRA2_ncfile,
    var_info_fxn,
):
    target_file = test_file
    width = 300
    height = 150
    target_area = test_area_fxn(width=width, height=height)
    var_info = var_info_fxn(test_MERRA2_ncfile)
    expected_vars_copied = {'/lon', '/lat'}

    with (
        Dataset(test_MERRA2_ncfile, mode='r') as source_ds,
        Dataset(target_file, mode='w') as target_ds,
    ):
        _transfer_dimensions(source_ds, target_ds, target_area, var_info)

        vars_copied = _copy_dimension_variables(
            source_ds, target_ds, target_area, var_info
        )

        assert expected_vars_copied == vars_copied

    with Dataset(target_file, mode='r') as validate:
        assert validate.dimensions['lon'].size == width
        assert validate.dimensions['lat'].size == height
        assert validate.dimensions['lev'].size == 42


def test__resample_layer_compute_float_explicit_fill(var_info_fxn, test_MERRA2_ncfile):
    """Test resampler.compute with float input and explicit fill value."""
    var_info = var_info_fxn(test_MERRA2_ncfile)
    source_plane = np.array(np.arange(12).reshape(4, 3), dtype=np.float32)
    resampler_mock = MagicMock()
    var_name = '/SLP'
    eventual_fill_value = np.float64(-9999.0)

    expected_source = source_plane.astype(np.float64)
    expected_rps = _get_rows_per_scan(source_plane.shape[0])

    _resample_layer(
        source_plane, resampler_mock, var_info, var_name, eventual_fill_value
    )

    call_args, call_kwargs = resampler_mock.compute.call_args
    actual_source = call_args[0]
    actual_fill_value = call_kwargs['fill_value']
    actual_rps = call_kwargs['rows_per_scan']

    np.testing.assert_array_equal(expected_source, actual_source)
    assert actual_fill_value == eventual_fill_value
    assert actual_rps == expected_rps
    assert 'maximum_weight_mode' not in call_kwargs  # Default for float


def test__resample_layer_compute_int_explicit_fill(var_info_fxn, test_MERRA2_ncfile):
    """Test resampler.compute with int input and explicit fill value."""
    var_info = var_info_fxn(test_MERRA2_ncfile)
    source_plane = np.array(np.arange(12).reshape(4, 3), dtype=np.int32)
    resampler_mock = MagicMock()
    var_name = '/PS'
    eventual_fill_value = np.int32(9999.0)

    expected_source = source_plane.astype(np.float64)
    expected_rps = _get_rows_per_scan(source_plane.shape[0])

    _resample_layer(
        source_plane, resampler_mock, var_info, var_name, eventual_fill_value
    )

    call_args, call_kwargs = resampler_mock.compute.call_args
    actual_source = call_args[0]
    actual_fill_value = call_kwargs['fill_value']
    actual_rps = call_kwargs['rows_per_scan']

    np.testing.assert_array_equal(expected_source, actual_source)
    assert actual_fill_value == eventual_fill_value
    assert actual_rps == expected_rps
    assert call_kwargs['maximum_weight_mode'] is True


def test__resample_layer_compute_float_no_fill(var_info_fxn, test_MERRA2_ncfile):
    """Test resampler.compute with float input and no explicit fill value."""
    var_info = var_info_fxn(test_MERRA2_ncfile)
    source_plane = np.array(np.arange(12).reshape(4, 3), dtype=np.float32)
    resampler_mock = MagicMock()
    resampler_mock._get_default_fill.return_value = -999.0
    var_name = '/QI'

    expected_source = source_plane.astype(np.float64)
    expected_rps = _get_rows_per_scan(source_plane.shape[0])

    _resample_layer(source_plane, resampler_mock, var_info, var_name, None)

    call_args, call_kwargs = resampler_mock.compute.call_args
    actual_source = call_args[0]
    actual_fill_value = call_kwargs['fill_value']
    actual_rps = call_kwargs['rows_per_scan']

    np.testing.assert_array_equal(expected_source, actual_source)
    assert actual_fill_value == -999.0
    assert actual_rps == expected_rps
    assert 'maximum_weight_mode' not in call_kwargs

    assert resampler_mock._get_default_fill.call_count == 1


def test__prepare_data_plane_floating_without_rotation(
    var_info_fxn, test_MERRA2_ncfile
):
    var_info = var_info_fxn(test_MERRA2_ncfile)
    test_data = np.array(np.arange(12).reshape(4, 3), dtype=np.float32)
    var_name = '/T'
    expected_data = np.copy(test_data)
    actual_data = _prepare_data_plane(test_data, var_info, var_name, cast_to=np.float64)

    assert np.float64 == actual_data.dtype
    np.testing.assert_equal(expected_data, actual_data)


def test__prepare_data_plane_floating_with_rotation(var_info_fxn, test_IMERG_ncfile):
    var_info = var_info_fxn(test_IMERG_ncfile)
    test_data = np.array(np.arange(12).reshape(4, 3), dtype=np.float16)
    var_name = '/Grid/HQprecipitation'
    expected_data = np.copy(test_data.T)
    actual_data = _prepare_data_plane(test_data, var_info, var_name, cast_to=np.float64)

    assert np.float64 == actual_data.dtype
    np.testing.assert_equal(expected_data, actual_data)


def test__prepare_data_plane_int_without_rotation(var_info_fxn, test_MERRA2_ncfile):
    var_info = var_info_fxn(test_MERRA2_ncfile)
    test_data = np.array(np.arange(12).reshape(4, 3), dtype=np.int8)
    var_name = '/T'
    expected_data = np.copy(test_data)
    actual_data = _prepare_data_plane(test_data, var_info, var_name, cast_to=np.float64)

    assert np.float64 == actual_data.dtype
    np.testing.assert_equal(expected_data, actual_data)


def test__prepare_data_plane_int_with_rotation(var_info_fxn, test_IMERG_ncfile):
    var_info = var_info_fxn(test_IMERG_ncfile)
    test_data = np.array(np.arange(12).reshape(4, 3), dtype=np.int64)
    test_data[0, 0] = -99999999
    var_name = '/Grid/HQprecipitation'
    expected_data = np.copy(test_data.T).astype(np.float64)

    actual_data = _prepare_data_plane(test_data, var_info, var_name, cast_to=np.float64)

    assert np.float64 == actual_data.dtype
    np.testing.assert_equal(expected_data, actual_data)


def test__get_bounds_var(var_info_fxn, test_IMERG_ncfile):
    var_info = var_info_fxn(test_IMERG_ncfile)
    expected_bounds = 'lon_bnds'

    actual_bounds = _get_bounds_var(var_info, '/Grid/lon')
    assert expected_bounds == actual_bounds


def test__copy_resampled_bounds_variable(
    test_file, test_area_fxn, test_IMERG_ncfile, var_info_fxn
):
    target_file = test_file
    target_area = test_area_fxn()
    var_info = var_info_fxn(test_IMERG_ncfile)
    bnds_var = '/Grid/lat_bnds'
    var_copied = None

    expected_lat_bnds = np.array(
        [
            target_area.projection_y_coords + 0.5,
            target_area.projection_y_coords - 0.5,
        ]
    ).T

    with (
        Dataset(test_IMERG_ncfile, mode='r') as source_ds,
        Dataset(target_file, mode='w') as target_ds,
    ):
        _transfer_dimensions(source_ds, target_ds, target_area, var_info)

        var_copied = _copy_resampled_bounds_variable(
            source_ds, target_ds, bnds_var, target_area, var_info
        )

    assert {bnds_var} == var_copied
    with Dataset(target_file, mode='r') as validate:
        assert_array_equal(expected_lat_bnds, validate['Grid']['lat_bnds'][:])


def test__resampled_dimension_variable_names_root_level_dimensions(
    test_1D_dimensions_ncfile, var_info_fxn
):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    expected_resampled = {'/lon', '/lat'}

    actual_resampled = _resampled_dimension_variable_names(var_info)
    assert expected_resampled == actual_resampled


def test__resampled_dimension_variable_names_grouped_dimensions(
    test_IMERG_ncfile, var_info_fxn
):
    var_info = var_info_fxn(test_IMERG_ncfile)
    expected_resampled = {
        '/Grid/lon',
        '/Grid/lat',
        '/Grid/lon_bnds',
        '/Grid/lat_bnds',
    }

    actual_resampled = _resampled_dimension_variable_names(var_info)
    assert expected_resampled == actual_resampled


def test__multiple_resampled_dimension_variable_names(test_ATL14_ncfile, var_info_fxn):
    var_info = var_info_fxn(test_ATL14_ncfile)
    expected_resampled = {'/x', '/y', '/tile_stats/x', '/tile_stats/y'}

    actual_resampled = _resampled_dimension_variable_names(var_info)
    assert expected_resampled == actual_resampled


def test__crs_variable_name_multiple_grids_separate_groups():
    dim_pair = ('/Grid/lat', '/Grid/lon')
    dim_pairs = [
        ('/Grid/lat', '/Grid/lon'),
        ('/Grid2/lat', '/Grid2/lon'),
        ('/Grid3/lat', '/Grid3/lon'),
    ]

    expected_crs_name = '/Grid/crs'
    actual_crs_name = _crs_variable_name(dim_pair, dim_pairs)
    assert expected_crs_name == actual_crs_name


def test__crs_variable_name_single_grid():
    dim_pair = ('/lat', '/lon')
    dim_pairs = [('/lat', '/lon')]
    expected_crs_name = '/crs'

    actual_crs_name = _crs_variable_name(dim_pair, dim_pairs)
    assert expected_crs_name == actual_crs_name


def test__crs_variable_name_multiple_grids_share_group():
    dim_pair = ('/global_grid_lat', '/global_grid_lon')
    dim_pairs = [
        ('/npolar_grid_lat', '/npolar_grid_lon'),
        ('/global_grid_lat', '/global_grid_lon'),
        ('/spolar_grid_lat', '/spolar_grid_lon'),
    ]

    expected_crs_name = '/crs_global_grid_lat_global_grid_lon'
    actual_crs_name = _crs_variable_name(dim_pair, dim_pairs)
    assert expected_crs_name == actual_crs_name


def test__transfer_metadata(test_file, test_1D_dimensions_ncfile):
    """Tests to ensure root and group level metadata is transfered to target."""
    _generate_test_file = test_file

    # metadata Set in the test 1D file
    expected_root_metadata = {
        'root-attribute1': 'value1',
        'root-attribute2': 'value2',
    }
    expected_root_groups = {'level1-nested1', 'level1-nested2'}
    expected_nested_metadata = {'level2-nested1': 'level2-nested1-value1'}

    with (
        Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds,
        Dataset(_generate_test_file, mode='w') as target_ds,
    ):
        _transfer_metadata(source_ds, target_ds)

    with Dataset(_generate_test_file, mode='r') as validate:
        root_metadata = {attr: validate.getncattr(attr) for attr in validate.ncattrs()}
        root_groups = set(validate.groups.keys())
        nested_group = validate['/level1-nested1/level2-nested1']
        nested_metadata = {
            attr: nested_group.getncattr(attr) for attr in nested_group.ncattrs()
        }

        assert expected_root_groups == root_groups
        assert expected_root_metadata == root_metadata
        assert expected_nested_metadata == nested_metadata


def test__transfer_dimensions(test_area_fxn, var_info_fxn, test_1D_dimensions_ncfile):
    """Tests transfer of all dimensions.

    test transfer of dimensions from source to target including resizing
    for the target's area definition.  The internal functions of
    _transfer_dimensions are tested further down in this file.

    """
    width = 36
    height = 18
    _generate_test_area = test_area_fxn(width=width, height=height)
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    target_file = test_file
    with (
        Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds,
        Dataset(target_file, mode='w') as target_ds,
    ):
        _transfer_dimensions(source_ds, target_ds, _generate_test_area, var_info)

    with Dataset(target_file, mode='r') as validate:
        assert validate.dimensions['bnds'].size == 2
        assert validate.dimensions['time'].size == 0
        assert validate.dimensions['time'].isunlimited() is True
        assert validate.dimensions['lon'].size == width
        assert validate.dimensions['lat'].size == height


def test__clone_variables(
    test_file, var_info_fxn, test_area_fxn, test_1D_dimensions_ncfile
):
    target_file = test_file
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    width = 36
    height = 18

    _generate_test_area = test_area_fxn(width=width, height=height)

    copy_vars = {'/time', '/time_bnds'}
    with (
        Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds,
        Dataset(target_file, mode='w') as target_ds,
    ):
        _transfer_dimensions(source_ds, target_ds, _generate_test_area, var_info)

        copied = _clone_variables(source_ds, target_ds, copy_vars)

        assert copy_vars == copied

        with Dataset(target_file, mode='r') as validate:
            assert_array_equal(validate['time_bnds'], source_ds['time_bnds'])
            assert_array_equal(validate['time'], source_ds['time'])


def test__create_resampled_dimensions_root_dimensions(
    var_info_fxn,
    test_1D_dimensions_ncfile,
    test_area_fxn,
    test_file,
):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    width = 36
    height = 18
    _generate_test_area = test_area_fxn(width=width, height=height)
    target_file = test_file

    with Dataset(target_file, mode='w') as target_ds:
        _create_resampled_dimensions(
            [('/lat', '/lon')], target_ds, _generate_test_area, var_info
        )

    with Dataset(target_file, mode='r') as validate:
        assert validate.dimensions['lat'].size == 18
        assert validate.dimensions['lon'].size == 36


def test__create_resampled_dimensions_group_level_dimensions(
    var_info_fxn,
    test_IMERG_ncfile,
    test_area_fxn,
    test_file,
):
    var_info = var_info_fxn(test_IMERG_ncfile)
    _generate_test_area = test_area_fxn()
    target_file = test_file
    with Dataset(target_file, mode='w') as target_ds:
        _create_resampled_dimensions(
            [('/Grid/lon', '/Grid/lat')],
            target_ds,
            _generate_test_area,
            var_info,
        )

    with Dataset(target_file, mode='r') as validate:
        assert validate['Grid'].dimensions['lat'].size == 180
        assert validate['Grid'].dimensions['lon'].size == 360


def test__resampler_kwargs_floating_data():
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype='float')
    expected_args = {'rows_per_scan': 2}
    actual_args = _resampler_kwargs(data, 'float')
    assert expected_args == actual_args


def test__resampler_kwargs_all_rows_needed():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype='float')
    expected_args = {'rows_per_scan': 7}
    actual_args = _resampler_kwargs(data, 'float')
    assert expected_args == actual_args


def test__resampler_kwargs_integer_data():
    data = np.array([1, 2, 3], dtype='int16')
    expected_args = {
        'rows_per_scan': 3,
        'maximum_weight_mode': True,
    }
    actual_args = _resampler_kwargs(data, 'int16')
    assert expected_args == actual_args


def test__write_grid_mappings(
    test_file,
    var_info_fxn,
    test_1D_dimensions_ncfile,
    test_area_fxn,
):
    target_file = test_file
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    _generate_test_area = test_area_fxn()
    expected_crs_map = {('/lon', '/lat'): '/crs'}

    with (
        Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds,
        Dataset(target_file, mode='w') as target_ds,
    ):
        _transfer_metadata(source_ds, target_ds)
        _transfer_dimensions(source_ds, target_ds, _generate_test_area, var_info)

        actual_crs_map = _write_grid_mappings(
            target_ds, _resampled_dimension_pairs(var_info), _generate_test_area
        )
        assert expected_crs_map == actual_crs_map

    with Dataset(target_file, mode='r') as validate:
        crs = _get_variable(validate, '/crs')
        expected_crs_metadata = _generate_test_area.crs.to_cf()

        actual_crs_metadata = {attr: crs.getncattr(attr) for attr in crs.ncattrs()}

        assert expected_crs_metadata == actual_crs_metadata


def test__get_variable(test_ATL14_ncfile):
    with Dataset(test_ATL14_ncfile, mode='r') as source_ds:
        var_grouped = _get_variable(source_ds, '/tile_stats/RMS_data')
        expected_grouped = source_ds['tile_stats'].variables['RMS_data']
        assert expected_grouped == var_grouped

        var_flat = _get_variable(source_ds, '/ice_area')
        expected_flat = source_ds.variables['ice_area']
        assert expected_flat == var_flat


def test__create_dimension(test_file):
    name = '/somedim'
    size = 1000
    with Dataset(test_file, mode='w') as target_ds:
        dim = _create_dimension(target_ds, name, size)
        assert isinstance(dim, Dimension)
        assert dim.size == size
        assert dim.name == 'somedim'


def test__create_nested_dimension(test_file):
    name = '/some/deeply/nested/dimname'
    size = 2000
    with Dataset(test_file, mode='w') as target_ds:
        dim = _create_dimension(target_ds, name, size)
        assert isinstance(dim, Dimension)
        assert dim.size == size
        assert dim.name == 'dimname'


def test__get_flat_dimension(test_1D_dimensions_ncfile, latitudes):
    with Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds:
        lat_dim = _get_dimension(source_ds, '/lat')
        assert isinstance(lat_dim, Dimension)
        assert lat_dim.size == len(latitudes)
        assert lat_dim.name == 'lat'


def test__get_nested_dimension(test_IMERG_ncfile):
    with Dataset(test_IMERG_ncfile, mode='r') as source_ds:
        lat_dim = _get_dimension(source_ds, '/Grid/lat')
        assert isinstance(lat_dim, Dimension)
        assert lat_dim.name == 'lat'
        assert lat_dim.size == 1800


def test__copy_dimension(test_file, test_1D_dimensions_ncfile, longitudes):
    with (
        Dataset(test_file, mode='w') as target_ds,
        Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds,
    ):
        time_dimension = _copy_dimension('/time', source_ds, target_ds)
        assert time_dimension.isunlimited() is True
        assert time_dimension.size == 0

        lon_dimension = _copy_dimension('/lon', source_ds, target_ds)
        assert lon_dimension.isunlimited() is False
        assert lon_dimension.size == len(longitudes)


def test__copy_dimensions(test_file, test_1D_dimensions_ncfile, latitudes, longitudes):
    test_target = test_file
    with (
        Dataset(test_target, mode='w') as target_ds,
        Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds,
    ):
        _copy_dimensions({'/lat', '/lon', '/time', '/bnds'}, source_ds, target_ds)

    with Dataset(test_target, mode='r') as validate:
        assert validate.dimensions['time'].isunlimited() is True
        assert validate.dimensions['time'].size == 0
        assert validate.dimensions['lat'].size == len(latitudes)
        assert validate.dimensions['lon'].size == len(longitudes)
        assert validate.dimensions['bnds'].size == 2


def test__copy_dimensions_with_groups(test_file, test_IMERG_ncfile):
    with (
        Dataset(test_file, mode='w') as target_ds,
        Dataset(test_IMERG_ncfile, mode='r') as source_ds,
    ):
        _copy_dimensions(
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


def test__horizontal_dims_for_variable_grouped(test_IMERG_ncfile, var_info_fxn):
    var_info = var_info_fxn(test_IMERG_ncfile)
    expected_dims = ('/Grid/lon', '/Grid/lat')
    actual_dims = _horizontal_dims_for_variable(var_info, '/Grid/IRkalmanFilterWeight')
    assert expected_dims == actual_dims


def test__horizontal_dims_for_variable(var_info_fxn, test_1D_dimensions_ncfile):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    expected_dims = ('/lon', '/lat')
    actual_dims = _horizontal_dims_for_variable(var_info, '/data')
    assert expected_dims == actual_dims


def test__horizontal_dims_for_missing_variable(var_info_fxn, test_1D_dimensions_ncfile):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    expected_dims = None
    actual_dims = _horizontal_dims_for_variable(var_info, '/missing')
    assert expected_dims == actual_dims


def test__resampled_dimenension_pairs_1d_file(var_info_fxn, test_1D_dimensions_ncfile):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    expected_pairs = [('/lon', '/lat')]
    actual_pairs = _resampled_dimension_pairs(var_info)
    assert expected_pairs == actual_pairs


def test__resampled_dimenension_pairs_multiple_horizontal_pairs(
    var_info_fxn, test_ATL14_ncfile
):
    var_info = var_info_fxn(test_ATL14_ncfile)
    expected_pairs = [('/y', '/x'), ('/tile_stats/y', '/tile_stats/x')]
    actual_pairs = _resampled_dimension_pairs(var_info)
    assert set(expected_pairs) == set(actual_pairs)


def test__all_dimensions(var_info_fxn, test_1D_dimensions_ncfile):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    expected_dimensions = {'/time', '/lon', '/lat', '/bnds'}
    actual_dimensions = _all_dimensions(var_info)
    assert expected_dimensions == actual_dimensions


def test__unresampled_variables_flat_ungrouped(var_info_fxn, test_1D_dimensions_ncfile):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    expected_vars = {'/time', '/time_bnds'}
    actual_vars = _unresampled_variables(var_info)
    assert expected_vars == actual_vars


def test__unresampled_variables_IMERG_grouped(var_info_fxn, test_IMERG_ncfile):
    var_info = var_info_fxn(test_IMERG_ncfile)

    expected_vars = {'/Grid/time', '/Grid/time_bnds'}
    actual_vars = _unresampled_variables(var_info)
    assert expected_vars == actual_vars


def test__unresampled_variables_MERRA2_includes_levels(
    var_info_fxn, test_MERRA2_ncfile
):
    var_info = var_info_fxn(test_MERRA2_ncfile)

    expected_vars = {'/lev', '/time'}
    actual_vars = _unresampled_variables(var_info)
    assert expected_vars == actual_vars


def test__unresampled_variables_ATL14_lots_of_deep_group_vars(
    var_info_fxn, test_ATL14_ncfile
):
    var_info = var_info_fxn(test_ATL14_ncfile)

    expected_vars = {
        '/Polar_Stereographic',
        '/orbit_info/bounding_polygon_dim1',
        '/orbit_info/bounding_polygon_lat1',
        '/orbit_info/bounding_polygon_lon1',
        '/quality_assessment/qa_granule_fail_reason',
        '/quality_assessment/qa_granule_pass_fail',
    }
    actual_vars = _unresampled_variables(var_info)
    assert expected_vars == actual_vars


def test__all_dimension_variables_1d_file(var_info_fxn, test_1D_dimensions_ncfile):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    expected_vars = {'/lat', '/lon', '/time'}
    actual_vars = _all_dimension_variables(var_info)
    assert expected_vars == actual_vars


def test__all_dimension_variables_2D_file(var_info_fxn, test_2D_dimensions_ncfile):
    var_info = var_info_fxn(test_2D_dimensions_ncfile)
    expected_vars = {'/lat', '/lon'}
    actual_vars = _all_dimension_variables(var_info)
    assert expected_vars == actual_vars


def test__resampled_dimensions_1D_file(var_info_fxn, test_1D_dimensions_ncfile):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    expected_dimensions = {'/lat', '/lon'}
    actual_dimensions = _resampled_dimensions(var_info)
    assert expected_dimensions == actual_dimensions


def test__resampled_dimensions_ATL14_multiple_grids(var_info_fxn, test_ATL14_ncfile):
    var_info = var_info_fxn(test_ATL14_ncfile)

    expected_dimensions = {'/x', '/y', '/tile_stats/x', '/tile_stats/y'}
    actual_dimensions = _resampled_dimensions(var_info)
    assert expected_dimensions == actual_dimensions


def test__needs_rotation_needs_rotation(var_info_fxn, test_1D_dimensions_ncfile):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    assert _needs_rotation(var_info, '/data') is True


def test__needs_rotation_no_rotation(var_info_fxn, test_MERRA2_ncfile):
    var_info = var_info_fxn(test_MERRA2_ncfile)
    assert _needs_rotation(var_info, '/PHIS') is False
    assert _needs_rotation(var_info, '/OMEGA') is False


@pytest.mark.parametrize(
    'int_type',
    [
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
    ],
)
def test__integer_like(int_type):
    assert _integer_like(int_type) is True


@pytest.mark.parametrize('float_type', [np.float16, np.float32, np.float64])
def test__integer_like_false(float_type):
    assert _integer_like(float_type) is False


def test__integer_like_string():
    assert _integer_like(str) is False


@patch(
    'harmony_regridding_service.regridding_service.AreaDefinition', wraps=AreaDefinition
)
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


def test__is_horizontal_dim_test_valid_x(test_1D_dimensions_ncfile, var_info_fxn):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    assert _is_horizontal_dim('/lon', var_info) is True


def test__is_horizontal_dim_test_invalid_x(test_1D_dimensions_ncfile, var_info_fxn):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    assert _is_horizontal_dim('/lat', var_info) is False


def test__is_vertical_dim_test_valid_y(test_1D_dimensions_ncfile, var_info_fxn):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    assert _is_vertical_dim('/lat', var_info) is True


def test__is_vertical_dim_test_invalid_y(test_1D_dimensions_ncfile, var_info_fxn):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    assert _is_vertical_dim('/lon', var_info) is False


def test__get_horizontal_dims_x_dims(test_1D_dimensions_ncfile, var_info_fxn):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    dims = ('/lat', '/lon')
    expected_dim = ['/lon']

    actual = _get_horizontal_dims(dims, var_info)
    assert expected_dim == actual


def test__get_vertical_dims_y_dims(test_1D_dimensions_ncfile, var_info_fxn):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    dims = ('/lat', '/lon')
    expected_dim = ['/lat']

    actual = _get_vertical_dims(dims, var_info)
    assert expected_dim == actual


def test__get_vertical_dims_y_dims_no_variables(
    test_1D_dimensions_ncfile, var_info_fxn
):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    dims = ('/baddim1', '/baddim2')

    expected_dims = []
    actual_dims = _get_vertical_dims(dims, var_info)
    assert expected_dims == actual_dims


def test__get_horizontal_dims_x_dims_no_variables(
    test_1D_dimensions_ncfile, var_info_fxn
):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    dims = ('/baddim1', '/baddim2')
    expected_dims = []
    actual_dims = _get_horizontal_dims(dims, var_info)
    assert expected_dims == actual_dims


def test__get_horizontal_dims_x_dims_with_bad_variable(
    test_1D_dimensions_ncfile, var_info_fxn
):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    dims = ('/baddim1', '/lon')
    expected_dim = ['/lon']

    actual_dim = _get_horizontal_dims(dims, var_info)
    assert expected_dim == actual_dim


def test___get_vertical_dims_y_dims_multiple_values(
    test_1D_dimensions_ncfile, var_info_fxn
):
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    dims = ('/lat', '/lon', '/lat', '/ba')
    expected_dim = ['/lat', '/lat']

    actual = _get_vertical_dims(dims, var_info)
    assert expected_dim == actual


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
    assert _get_rows_per_scan(input_value) == expected, f'Failed for {description}'


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


def test__crs_from_source_data_expected_case(smap_projected_netcdf_file):
    dt = xr.open_datatree(smap_projected_netcdf_file)
    expected_crs = CRS('epsg:6933')
    crs = _crs_from_source_data(dt, set({'/Forecast_Data/sm_profile_forecast'}))
    assert crs.to_epsg() == expected_crs


def test__crs_from_source_data_missing(smap_projected_netcdf_file):
    dt = xr.open_datatree(smap_projected_netcdf_file)
    dt['/Forecast_Data/sm_profile_forecast'].attrs.pop('grid_mapping')
    with pytest.raises(InvalidSourceCRS, match='No grid_mapping metadata found'):
        _crs_from_source_data(dt, set({'/Forecast_Data/sm_profile_forecast'}))


def test__crs_from_source_data_bad(smap_projected_netcdf_file):
    dt = xr.open_datatree(smap_projected_netcdf_file)
    dt['EASE2_global_projection'].attrs['grid_mapping_name'] = 'nonsense projection'
    dt['/Forecast_Data/sm_profile_forecast'].attrs['grid_mapping']
    with pytest.raises(
        InvalidSourceCRS, match='Could not create a CRS from grid_mapping metadata'
    ):
        _crs_from_source_data(dt, set({'/Forecast_Data/sm_profile_forecast'}))


@pytest.mark.parametrize(
    'file_fixture_name, dimensions, expected_result',
    [
        ('test_2D_dimensions_ncfile', ('/lon', '/lat'), False),
        ('smap_projected_netcdf_file', ('/y', '/x'), True),
    ],
)
def test__dims_are_projected_x_y(
    var_info_fxn, request, file_fixture_name, dimensions, expected_result
):
    """Test if dimensions are projected x/y coordinates."""
    file_fixture = request.getfixturevalue(file_fixture_name)
    var_info = var_info_fxn(file_fixture)
    assert _dims_are_projected_x_y(dimensions, var_info) is expected_result


@pytest.mark.parametrize(
    'file_fixture_name, dimensions, expected_result',
    [
        ('test_2D_dimensions_ncfile', ('/lon', '/lat'), True),
        ('smap_projected_netcdf_file', ('/y', '/x'), False),
    ],
)
def test__dims_are_lon_lat(
    var_info_fxn, request, file_fixture_name, dimensions, expected_result
):
    """Test if dimensions are lon/lat coordinates."""
    file_fixture = request.getfixturevalue(file_fixture_name)
    var_info = var_info_fxn(file_fixture)
    assert _dims_are_lon_lat(dimensions, var_info) is expected_result


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
        result_filename = regrid(message, input_filename, source, logger_mock)

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
