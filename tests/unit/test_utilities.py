"""Test the Utilities module."""

from unittest import TestCase

from harmony_service_lib.message import Message

from harmony_regridding_service.message_utilities import (
    has_valid_interpolation,
)
from harmony_regridding_service.utilities import get_file_mime_type


class TestUtilities(TestCase):
    """A class testing the harmony_regridding_service.utilities module."""

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
