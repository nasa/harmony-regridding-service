"""Test the utilities module."""

from unittest import TestCase

from harmony_service_lib.message import Message
from netCDF4 import Dataset
from numpy.testing import assert_array_equal

from harmony_regridding_service.message_utilities import (
    has_valid_interpolation,
)
from harmony_regridding_service.resample import (
    transfer_resampled_dimensions,
)
from harmony_regridding_service.utilities import (
    clone_variables,
    copy_var_with_attrs,
    copy_var_without_metadata,
    get_bounds_var,
    get_file_mime_type,
    get_variable_from_dataset,
    transfer_metadata,
    walk_groups,
)


class TestUtilities(TestCase):
    """A class testing the harmony_regridding_service.utilities module.

    TODO: Update this to pytest.
    """

    def test_get_file_mime_type(self):
        """Ensure a MIME type can be retrieved from an input file path."""
        with self.subTest('File with MIME type known by Python.'):
            self.assertEqual(get_file_mime_type('file.nc'), 'application/x-netcdf')

        with self.subTest('File with MIME type retrieved from dictionary.'):
            self.assertEqual(get_file_mime_type('file.nc4'), 'application/x-netcdf4')

        with self.subTest('File with entirely unknown MIME type.'):
            self.assertIsNone(get_file_mime_type('file.xyzzyx'))

        with self.subTest('Upper case letters handled.'):
            self.assertEqual(get_file_mime_type('file.HDF5'), 'application/x-hdf5')

    def test_has_valid_interpolation(self):
        """Test has_valid_interpolation.

        Ensure that the function correctly determines if the supplied
        Harmony message either omits the `format.interpolation` attribute,
        or specifies EWA via a fully spelled-out string. The TRT-210 MVP
        only allows for interpolation using EWA.

        """
        with self.subTest('format = None returns True'):
            test_message = Message({})
            self.assertTrue(has_valid_interpolation(test_message))

        with self.subTest('format.interpolation = None returns True'):
            test_message = Message({'format': {}})
            self.assertTrue(has_valid_interpolation(test_message))

        with self.subTest('EWA (spelled fully) returns True'):
            test_message = Message(
                {'format': {'interpolation': 'Elliptical Weighted Averaging'}}
            )
            self.assertTrue(has_valid_interpolation(test_message))

        with self.subTest('Unexpected interpolation returns False'):
            test_message = Message({'format': {'interpolation': 'Bilinear'}})
            self.assertFalse(has_valid_interpolation(test_message))


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
        for groups in walk_groups(validate):
            for group in groups:
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
    target_area = test_area_fxn()
    var_info = var_info_fxn(test_1D_dimensions_ncfile)
    expected_metadata = {'units': 'widgets per month'}
    with (
        Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds,
        Dataset(target_file, mode='w') as target_ds,
    ):
        transfer_resampled_dimensions(source_ds, target_ds, target_area, var_info)
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
    target_area = test_area_fxn()
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

    _generate_test_area = test_area_fxn(width=width, height=height)

    copy_vars = {'/time', '/time_bnds'}
    with (
        Dataset(test_1D_dimensions_ncfile, mode='r') as source_ds,
        Dataset(target_file, mode='w') as target_ds,
    ):
        transfer_resampled_dimensions(
            source_ds, target_ds, _generate_test_area, var_info
        )

        copied = clone_variables(source_ds, target_ds, copy_vars)

        assert copy_vars == copied

        with Dataset(target_file, mode='r') as validate:
            assert_array_equal(validate['time_bnds'], source_ds['time_bnds'])
            assert_array_equal(validate['time'], source_ds['time'])


def test_get_variable_from_dataset(test_ATL14_ncfile):
    with Dataset(test_ATL14_ncfile, mode='r') as source_ds:
        var_grouped = get_variable_from_dataset(source_ds, '/tile_stats/RMS_data')
        expected_grouped = source_ds['tile_stats'].variables['RMS_data']
        assert expected_grouped == var_grouped

        var_flat = get_variable_from_dataset(source_ds, '/ice_area')
        expected_flat = source_ds.variables['ice_area']
        assert expected_flat == var_flat


def test_get_bounds_var(var_info_fxn, test_IMERG_ncfile):
    var_info = var_info_fxn(test_IMERG_ncfile)
    expected_bounds = 'lon_bnds'

    actual_bounds = get_bounds_var(var_info, '/Grid/lon')
    assert expected_bounds == actual_bounds
