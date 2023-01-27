from unittest import TestCase

from harmony_service.utilities import get_file_mime_type


class TestUtilities(TestCase):
    """ A class for testing the harmony_service.utilities module. """

    def test_get_file_mime_type(self):
        """ Ensure a MIME type can be retrieved from an input file path. """
        with self.subTest('File with MIME type known by Python.'):
            self.assertEqual(get_file_mime_type('file.nc'),
                             'application/x-netcdf')

        with self.subTest('File with MIME type retrieved from dictionary.'):
            self.assertEqual(get_file_mime_type('file.nc4'),
                             'application/x-netcdf4')

        with self.subTest('File with entirely unknown MIME type.'):
            self.assertIsNone(get_file_mime_type('file.xyz'))
