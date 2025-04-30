"""Set up common pytest fixtures."""

import numpy as np
import pytest
from xarray import DataArray, Dataset, DataTree


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
    # Create a path for the temporary file within the pytest temporary directory
    file_path = tmp_path / 'smap_projected_data.nc'

    # Save the datatree to a NetCDF file
    smap_projected_datatree.to_netcdf(file_path)

    yield file_path


def create_projected_datatree():
    """Creates a sample xarray datatree in code for testing purposes.

    This data was created from a slice of SMAP_L4_SM_aup_20220103T000000_Vv8010_001.h5
    {"y": slice(114, 120, None), "x": slice(200, 205, None)}, this small area
    has both valid and missing data to use in testing. Only a few of the
    SPL4SMAU variables are included.

    Returns:
         The created datatree object.

    """
    x_coords = np.array(
        [
            -15561416.159668,
            -15552408.104004,
            -15543400.04834,
            -15534391.992676,
            -15525383.937012,
        ]
    )
    y_coords = np.array(
        [
            6283118.82568359,
            6274110.77001953,
            6265102.71435547,
            6256094.65869141,
            6247086.60302734,
            6238078.54736328,
        ]
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
        'QAAbstract': 'An ASCII product that contains statistical information on da...',
        'ECSVersionDescription': 'Three-character string that identifies the type o...',
        'UUID': '48ED4B4C-D0D4-11EF-B181-E58983D16534',
        'language': 'eng',
        'QAFileName': 'SMAP_L4_SM_aup_20220103T000000_Vv8010_001.qa',
        'topicCategoryCode': 'geoscientificInformation',
        'ECSVersionID': '008',
        'abstract': 'The SMAP L4_SM data product provides global, 3-hourly surface ...',
        'creationDate': '2025-01-12T10:59:24.000Z',
        'VersionID': 'Vv8010',
        'status': 'onGoing',
        'QACreationDate': '2025-01-12T10:59:24.000Z',
        'credit': 'The software that generates the L4_SM data product and the data ...',
        'purpose': 'The SMAP L4_SM data product provides spatially and temporally...',
        'fileName': 'SMAP_L4_SM_aup_20220103T000000_Vv8010_001.h5',
        'longName': 'SMAP L4 Global 3-hourly 9 km Surface and Rootzone Soil Moistur...',
        'CompositeReleaseID': 'Vv80100012',
        'originatorOrganizationName': 'NASA Global Modeling and Assimilation Office...',
        'shortName': 'SPL4SMAU',
    }

    ease2_attributes = {
        'grid_mapping_name': 'lambert_cylindrical_equal_area',
        'standard_parallel': 30.0,
        'false_easting': 0.0,
        'false_northing': 0.0,
        'longitude_of_central_meridian': 0.0,
    }

    sample_datatree = DataTree(
        Dataset(
            coords={'y': y_coords, 'x': x_coords},
            data_vars={
                'EASE2_global_projection': DataArray(
                    '', name='EASE2_global_projection', attrs=ease2_attributes
                )
            },
        )
    )

    sample_datatree['Metadata/DatasetIdentification'] = Dataset(
        attrs=dataset_identification_metadata
    )

    sample_datatree['cell_row'] = DataArray(
        cell_row_data,
        dims=['y', 'x'],
        attrs=cell_row_attrs,
    )

    sample_datatree['Forecast_Data'] = Dataset(
        {
            'sm_profile_forecast': DataArray(
                sm_profile_values,
                dims=['y', 'x'],
                attrs=sm_profile_attributes,
            )
        }
    )
    sample_datatree['Analysis_Data'] = Dataset(
        {
            'sm_profile_analysis': DataArray(
                sm_profile_analysis_values,
                dims=['y', 'x'],
                attrs=sm_profile_analysis_attrs,
            )
        }
    )

    sample_datatree['Observations_Data'] = Dataset(
        {
            'tb_v_obs': DataArray(
                tb_v_obs_values,
                dims=['y', 'x'],
                attrs=tb_v_obs_attributes,
            )
        }
    )
    sample_datatree['Analysis_Data'] = Dataset(
        {
            'sm_profile_analysis': DataArray(
                sm_profile_values,
                dims=['y', 'x'],
                attrs=sm_profile_attributes,
            )
        }
    )

    return sample_datatree
