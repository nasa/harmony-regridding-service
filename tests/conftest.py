"""Set up common pytest fixtures."""

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

# from xarray import DataArray, Dataset, DataTree
import xarray as xr
from harmony_service_lib.message import Message as HarmonyMessage
from netCDF4 import Dataset
from pyresample.geometry import AreaDefinition
from varinfo import VarInfoFromNetCDF4

HRS_VARINFO_CONFIG_FILENAME = str(
    Path(
        Path(__file__).parent.parent,
        'harmony_regridding_service',
        'config',
        'HRS_varinfo_config.json',
    )
)


## pytest fixtures
@pytest.fixture(scope='session')
def test_fixtures_dir():
    """Return path to the test fixtures directory."""
    return Path(Path(__file__).parent, 'unit/fixtures')


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

    def var_info(nc_file: str | Path, short_name: str | None = None):
        return VarInfoFromNetCDF4(
            str(nc_file),
            config_file=HRS_VARINFO_CONFIG_FILENAME,
            short_name=short_name,
        )

    return var_info


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

    def test_area(width=360, height=180, area_extent=(-180, -90, 180, 90)):
        return AreaDefinition(
            'test_id',
            'test area definition',
            None,
            '+proj=longlat +datum=WGS84 +no_defs +type=crs',
            width,
            height,
            area_extent,
        )

    return test_area


@pytest.fixture
def message_params():
    """Fixture for creating Harmony Messages."""
    params = {
        'mime': 'application/x-netcdf',
        'crs': 'EPSG:4326',
        'srs': {
            'epsg': 'EPSG:4326',
            'proj4': '+proj=longlat +datum=WGS84 +no_defs',
            'wkt': 'GEOGCS["WGS 84",DATUM...',
        },
        'scale_extent': {
            'x': {'min': -180, 'max': 180},
            'y': {'min': -90, 'max': 90},
        },
        'scale_size': {'x': 10, 'y': 9},
        'height': 100,
        'width': 99,
    }
    return params


@pytest.fixture(scope='session')
def smap_projected_datatree():
    """Provide the sample projected datatree object."""
    return create_projected_datatree()


@pytest.fixture()
def smap_projected_netcdf_file(tmp_path, smap_projected_datatree):
    """Fixture that creates a temporary NetCDF sample file.

    Args:
        tmp_path: pytest's built-in fixture for temporary directories.
        smap_projected_datatree: The fixture that provides the datatree object.

    Yields:
        pathlib.Path: The path to the temporary NetCDF file.
    """
    file_path = tmp_path / 'smap_projected_data.nc'
    smap_projected_datatree.to_netcdf(file_path)

    yield file_path


def create_projected_datatree():
    """Creates a sample xarray datatree in code for testing purposes.

    This data was created from a slice of SMAP_L4_SM_aup_20220103T000000_Vv8010_001.h5
    {"y": slice(114, 120, None), "x": slice(200, 205, None)}, this small area
    has both valid and missing data to use in testing. Only a few of the
    SPL4SMAU variables are included, one from each data group.

    Returns:
         The created datatree object.

    """
    x_coords = xr.DataArray(
        np.array(
            [
                -15561416.159668,
                -15552408.104004,
                -15543400.04834,
                -15534391.992676,
                -15525383.937012,
            ]
        ),
        dims='x',
        attrs={
            'long_name': 'X coordinate of cell center in EASE 2.0 global projection',
            'standard_name': 'projection_x_coordinate',
            'axis': 'X',
            'units': 'm',
        },
    )

    y_coords = xr.DataArray(
        np.array(
            [
                6283118.82568359,
                6274110.77001953,
                6265102.71435547,
                6256094.65869141,
                6247086.60302734,
                6238078.54736328,
            ]
        ),
        dims='y',
        attrs={
            'long_name': 'Y coordinate of cell center in EASE 2.0 global projection',
            'standard_name': 'projection_y_coordinate',
            'axis': 'Y',
            'units': 'm',
        },
    )

    sm_profile_values = np.array(
        [
            [0.926805, 0.366136, 0.403497, 0.378182, 0.38039],
            [0.926797, 0.389841, 0.319339, 0.321144, 0.33077],
            [0.418859, np.nan, 0.322053, 0.338612, 0.341305],
            [np.nan, np.nan, 0.309239, 0.317699, 0.307299],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    # reuse the data
    sm_profile_analysis_values = sm_profile_values
    sm_profile_analysis_attrs = {
        'valid_max': np.float32(0.9),
        '_FillValue': np.float32(-9999.0),
        'fmissing_value': np.float32(-9999.0),
        'missing_value': np.float32(-9999.0),
        'DIMENSION_LABELS': ['y', 'x'],
        'units': 'm3 m-3',
        'grid_mapping': 'EASE2_global_projection',
        'valid_min': np.float32(0.0),
        'long_name': 'Analysis total profile soil moisture (0 cm to model ...)',
    }
    sm_profile_attributes = {
        'valid_max': np.float32(0.9),
        'fmissing_value': np.float32(-9999.0),
        'DIMENSION_LABELS': ['y', 'x'],
        'units': 'm3 m-3',
        'grid_mapping': 'EASE2_global_projection',
        'valid_min': np.float32(0.0),
        'long_name': 'Catchment model forecast total profile soil moisture (0 cm ...)',
    }

    tb_v_obs_values = np.array(
        [
            [249.2, 246.2, 246.2, 246.2, 246.2],
            [249.2, 246.2, 246.2, 246.2, 246.2],
            [249.2, 246.2, 246.2, 246.2, 246.2],
            [249.2, 246.2, 246.2, 246.2, 246.2],
            [249.4, np.nan, np.nan, np.nan, np.nan],
            [249.4, np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    tb_v_obs_attributes = {
        'valid_max': np.float32(350.0),
        '_FillValue': np.float32(-9999.0),
        'fmissing_value': np.float32(-9999.0),
        'missing_value': np.float32(-9999.0),
        'DIMENSION_LABELS': ['y', 'x'],
        'units': 'K',
        'grid_mapping': 'EASE2_global_projection',
        'valid_min': np.float32(100.0),
        'long_name': 'Composite resolution observed (L2_SM_AP or L1C_TB) V-pol ...',
    }

    cell_row_data = np.array(
        [
            [114, 114, 114, 114, 114],
            [115, 115, 115, 115, 115],
            [116, 116, 116, 116, 116],
            [117, 117, 117, 117, 117],
            [118, 118, 118, 118, 118],
            [119, 119, 119, 119, 119],
        ],
        dtype=np.uint32,
    )
    cell_row_attrs = {
        'valid_max': np.uint32(1623),
        '_FillValue': np.uint32(4294967294),
        'fmissing_value': np.uint32(4294967294),
        'missing_value': np.uint32(4294967294),
        'DIMENSION_LABELS': ['y', 'x'],
        'units': 'dimensionless',
        'grid_mapping': 'EASE2_global_projection',
        'valid_min': np.uint32(0),
        'long_name': 'The row index of each cell<...>-Grid 2.0. Type is Unsigned32.',
    }

    dataset_identification_metadata = {
        'SMAPShortName': 'L4_SM_aup',
        'characterSet': 'utf8',
        'VersionID': 'Vv8010',
        'fileName': 'SMAP_L4_SM_aup_20220103T000000_Vv8010_001.h5',
        'longName': 'SMAP L4 Global 3-hourly 9 km Surface and Rootzone Soil Moistur...',
        'shortName': 'SPL4SMAU',
    }

    ease2_attributes = {
        'grid_mapping_name': 'lambert_cylindrical_equal_area',
        'standard_parallel': 30.0,
        'false_easting': 0.0,
        'false_northing': 0.0,
        'longitude_of_central_meridian': 0.0,
    }

    sample_datatree = xr.DataTree(
        xr.Dataset(
            coords={'y': y_coords, 'x': x_coords},
            data_vars={
                'EASE2_global_projection': xr.DataArray('', attrs=ease2_attributes)
            },
        )
    )

    sample_datatree['Metadata/DatasetIdentification'] = xr.Dataset(
        attrs=dataset_identification_metadata
    )

    sample_datatree['cell_row'] = xr.DataArray(
        cell_row_data,
        dims=['y', 'x'],
        attrs=cell_row_attrs,
    )

    sample_datatree['Forecast_Data'] = xr.Dataset(
        {
            'sm_profile_forecast': xr.DataArray(
                sm_profile_values,
                dims=['y', 'x'],
                attrs=sm_profile_attributes,
            )
        }
    )
    sample_datatree['Analysis_Data'] = xr.Dataset(
        {
            'sm_profile_analysis': xr.DataArray(
                sm_profile_analysis_values,
                dims=['y', 'x'],
                attrs=sm_profile_analysis_attrs,
            )
        }
    )

    sample_datatree['Observations_Data'] = xr.Dataset(
        {
            'tb_v_obs': xr.DataArray(
                tb_v_obs_values,
                dims=['y', 'x'],
                attrs=tb_v_obs_attributes,
            )
        }
    )
    sample_datatree['Analysis_Data'] = xr.Dataset(
        {
            'sm_profile_analysis': xr.DataArray(
                sm_profile_values,
                dims=['y', 'x'],
                attrs=sm_profile_attributes,
            )
        }
    )

    return sample_datatree
