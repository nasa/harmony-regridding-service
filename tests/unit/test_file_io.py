"""Test the File IO module."""

from harmony_service_lib.message import Message
from netCDF4 import Dataset
from numpy.testing import assert_array_equal

from harmony_regridding_service.file_io import (
    clone_variables,
    copy_var_with_attrs,
    copy_var_without_metadata,
    filter_grid_mappings_to_variables,
    get_file_mime_type,
    get_or_create_variable_in_dataset,
    input_grid_mappings,
    transfer_metadata,
    walk_groups,
)
from harmony_regridding_service.message_utilities import (
    has_valid_interpolation,
)
from harmony_regridding_service.resample import (
    transfer_resampled_dimensions,
)


class TestGetFileMimeType:
    """Test get_file_mime_type function."""

    def test_file_with_known_mime_type(self):
        """Ensure a MIME type can be retrieved from an input file path."""
        assert get_file_mime_type('file.nc') == 'application/x-netcdf'

    def test_file_with_mime_type_from_dictionary(self):
        """File with MIME type retrieved from dictionary."""
        assert get_file_mime_type('file.nc4') == 'application/x-netcdf4'

    def test_file_with_unknown_mime_type(self):
        """File with entirely unknown MIME type."""
        assert get_file_mime_type('file.xyzzyx') is None

    def test_upper_case_letters_handled(self):
        """Upper case letters handled."""
        assert get_file_mime_type('file.HDF5') == 'application/x-hdf5'


class TestHasValidInterpolation:
    """Test has_valid_interpolation function.

    Ensure that the function correctly determines if the supplied
    Harmony message either omits the `format.interpolation` attribute,
    or specifies EWA via a fully spelled-out string. The TRT-210 MVP
    only allows for interpolation using EWA.
    """

    def test_format_none_returns_true(self):
        """Format = None returns True."""
        test_message = Message({})
        assert has_valid_interpolation(test_message) is True

    def test_format_interpolation_none_returns_true(self):
        """format.interpolation = None returns True."""
        test_message = Message({'format': {}})
        assert has_valid_interpolation(test_message) is True

    def test_ewa_spelled_fully_returns_true(self):
        """EWA (spelled fully) returns True."""
        test_message = Message(
            {'format': {'interpolation': 'Elliptical Weighted Averaging'}}
        )
        assert has_valid_interpolation(test_message) is True

    def test_unexpected_interpolation_returns_false(self):
        """Unexpected interpolation returns False."""
        test_message = Message({'format': {'interpolation': 'Bilinear'}})
        assert has_valid_interpolation(test_message) is False


def test_walk_groups(test_file):
    """Demonstrate traversing all groups."""
    target_path = test_file
    groups = ['/a/nested/group', '/b/another/deeper/group2']
    expected_visited = {'a', 'nested', 'group', 'b', 'another', 'deeper', 'group2'}

    with Dataset(target_path, mode='w') as target_ds:
        for group in groups:
            target_ds.createGroup(group)

    actual_visited = set()
    with Dataset(target_path, mode='r') as validate:
        for wgroups in walk_groups(validate):
            for group in wgroups:
                actual_visited.update([group.name])

    assert expected_visited == actual_visited


def test_transfer_metadata(test_file, test_1D_dimensions_ncfile):
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
        transfer_metadata(source_ds, target_ds)

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


def test_copy_var_with_attrs(
    test_file, test_area_fxn, test_1D_dimensions_ncfile, var_info_fxn
):
    target_file = test_file
    target_areas = {('/lon', '/lat'): test_area_fxn()}
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    expected_metadata = {'units': 'widgets per month'}
    with (
        Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds,
        Dataset(target_file, mode='w') as target_ds,
    ):
        transfer_resampled_dimensions(source_ds, target_ds, target_areas, var_info)
        copy_var_with_attrs(source_ds, target_ds, '/data')

    with Dataset(target_file, mode='r') as validate:
        actual_metadata = {
            attr: validate['/data'].getncattr(attr)
            for attr in validate['/data'].ncattrs()
        }
        assert actual_metadata == expected_metadata


def test_copy_vars_without_metadata(
    test_file, test_area_fxn, test_1D_dimensions_ncfile, var_info_fxn
):
    target_file = test_file
    target_area = {('/lon', '/lat'): test_area_fxn()}
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    with (
        Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds,
        Dataset(target_file, mode='w') as target_ds,
    ):
        transfer_resampled_dimensions(source_ds, target_ds, target_area, var_info)
        copy_var_without_metadata(source_ds, target_ds, '/data')

    with Dataset(target_file, mode='r') as validate:
        actual_metadata = {
            attr: validate['/data'].getncattr(attr)
            for attr in validate['/data'].ncattrs()
        }
        assert {} == actual_metadata


def test_clone_variables(
    test_file, var_info_fxn, test_area_fxn, test_1D_dimensions_ncfile
):
    target_file = test_file
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    width = 36
    height = 18

    test_areas = {('/lon', '/lat'): test_area_fxn(width=width, height=height)}

    copy_vars = {'/time', '/time_bnds'}
    with (
        Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds,
        Dataset(target_file, mode='w') as target_ds,
    ):
        transfer_resampled_dimensions(source_ds, target_ds, test_areas, var_info)

        copied = clone_variables(source_ds, target_ds, copy_vars)

        assert copy_vars == copied

        with Dataset(target_file, mode='r') as validate:
            assert_array_equal(validate['time_bnds'], source_ds['time_bnds'])
            assert_array_equal(validate['time'], source_ds['time'])


def test_get_or_create_variable_in_dataset(test_ATL14_ncfile):
    with Dataset(test_ATL14_ncfile, mode='r') as source_ds:
        var_grouped = get_or_create_variable_in_dataset(
            source_ds, '/tile_stats/RMS_data'
        )
        expected_grouped = source_ds['tile_stats'].variables['RMS_data']
        assert expected_grouped == var_grouped

        var_flat = get_or_create_variable_in_dataset(source_ds, '/ice_area')
        expected_flat = source_ds.variables['ice_area']
        assert expected_flat == var_flat


def test_collect_grid_mappings_expected(test_spl3ftp_ncfile):
    expected_grid_mappings = {
        '/EASE2_global_projection_36km',
        '/EASE2_north_polar_projection_36km',
    }

    test_vars = {
        'Freeze_Thaw_Retrieval_Data_Global/longitude',
        'Freeze_Thaw_Retrieval_Data_Global/latitude',
        'Freeze_Thaw_Retrieval_Data_Global/transition_direction'
        'Freeze_Thaw_Retrieval_Data_Polar/longitude',
        'Freeze_Thaw_Retrieval_Data_Polar/latitude',
        'Freeze_Thaw_Retrieval_Data_Polar/transition_direction',
    }

    with Dataset(test_spl3ftp_ncfile, mode='r') as source_ds:
        actual_grid_mappings = input_grid_mappings(source_ds, test_vars)
        assert actual_grid_mappings == expected_grid_mappings


def test_collect_grid_mappings_limited_vars(test_spl3ftp_ncfile):
    expected_grid_mappings = {
        '/EASE2_global_projection_36km',
    }

    test_vars = {
        'Freeze_Thaw_Retrieval_Data_Global/longitude',
        'Freeze_Thaw_Retrieval_Data_Global/latitude',
        'Freeze_Thaw_Retrieval_Data_Global/transition_direction',
    }

    with Dataset(test_spl3ftp_ncfile, mode='r') as source_ds:
        actual_grid_mappings = input_grid_mappings(source_ds, test_vars)
        assert actual_grid_mappings == expected_grid_mappings


def test_collect_grid_mappings_vars_has_no_mapping(test_spl3ftp_ncfile):
    expected_grid_mappings = set()

    test_vars = {
        'Freeze_Thaw_Retrieval_Data_Global/am_pm',
        'Freeze_Thaw_Retrieval_Data_Polar/am_pm',
    }

    with Dataset(test_spl3ftp_ncfile, mode='r') as source_ds:
        actual_grid_mappings = input_grid_mappings(source_ds, test_vars)
        assert actual_grid_mappings == expected_grid_mappings


def test_collect_grid_mappings_var_does_not_exist(test_spl3ftp_ncfile):
    expected_grid_mappings = set()

    test_vars = {
        'Freeze_Thaw_Retrieval_Data_Polar/ThisVariableIsFake',
        'FakeFreeze_Thaw_Retrieval_Data_Global/am_pm',
        'Freeze_Thaw_Retrieval_Data_Polar/am_pm',
    }

    with Dataset(test_spl3ftp_ncfile, mode='r') as source_ds:
        actual_grid_mappings = input_grid_mappings(source_ds, test_vars)
        assert actual_grid_mappings == expected_grid_mappings


def test_filter_grid_mappings_to_variables():
    test_mapping_values = {
        '/crsName',
        'anotherCrsName',
        'crs1: coord1 coord2 crs2: coord3 coord4',
        'crs3: coord1 coord2',
        'crs4: coord5 coord6 crs4: coord5 coord6',
        '/crs5: coord6 /crs6: coord6',
    }

    expected_filtered_variables = {
        '/crsName',
        '/anotherCrsName',
        '/crs1',
        '/crs2',
        '/crs3',
        '/crs4',
        '/crs5',
        '/crs6',
    }
    actual_grid_mapping_variables = filter_grid_mappings_to_variables(
        test_mapping_values
    )

    assert actual_grid_mapping_variables == expected_filtered_variables
