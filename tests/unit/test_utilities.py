from unittest import TestCase

from harmony.message import Message

from harmony_regridding_service.exceptions import InvalidTargetCRS
from harmony_regridding_service.utilities import (
    _is_geographic_crs,
    get_file_mime_type,
    has_valid_crs,
    has_valid_interpolation,
)


class TestUtilities(TestCase):
    """ A class testing the harmony_regridding_service.utilities module. """

    def test_get_file_mime_type(self):
        """ Ensure a MIME type can be retrieved from an input file path. """
        with self.subTest('File with MIME type known by Python.'):
            self.assertEqual(get_file_mime_type('file.nc'),
                             'application/x-netcdf')

        with self.subTest('File with MIME type retrieved from dictionary.'):
            self.assertEqual(get_file_mime_type('file.nc4'),
                             'application/x-netcdf4')

        with self.subTest('File with entirely unknown MIME type.'):
            self.assertIsNone(get_file_mime_type('file.xyzzyx'))

        with self.subTest('Upper case letters handled.'):
            self.assertEqual(get_file_mime_type('file.HDF5'),
                             'application/x-hdf5')

    def test_has_valid_crs(self):
        """ Ensure the function correctly determines if the input Harmony
            message has a target Coordinate Reference System (CRS) that is
            compatible with the service. Currently this is either to not
            define the target CRS (assuming it to be geographic), or explicitly
            requesting geographic CRS via EPSG code or proj4 string.

        """
        with self.subTest('format = None returns True'):
            test_message = Message({})
            self.assertTrue(has_valid_crs(test_message))

        with self.subTest('format.crs = None returns True'):
            test_message = Message({'format': {}})
            self.assertTrue(has_valid_crs(test_message))

        with self.subTest('format.crs = "EPSG:4326" returns True'):
            test_message = Message({'format': {'crs': 'EPSG:4326'}})
            self.assertTrue(has_valid_crs(test_message))

        with self.subTest('format.crs = "+proj=longlat" returns True'):
            test_message = Message({'format': {'crs': '+proj=longlat'}})
            self.assertTrue(has_valid_crs(test_message))

        with self.subTest('format.crs = "4326" returns True'):
            test_message = Message({'format': {'crs': '4326'}})
            self.assertTrue(has_valid_crs(test_message))

        with self.subTest('Non-geographic EPSG code returns False'):
            test_message = Message({'format': {'crs': 'EPSG:6933'}})
            self.assertFalse(has_valid_crs(test_message))

        with self.subTest('Non-geographic proj4 string returns False'):
            test_message = Message({'format': {'crs': '+proj=cea'}})
            self.assertFalse(has_valid_crs(test_message))

        with self.subTest('String that cannot be parsed raises exception'):
            test_message = Message({'format': {'crs': 'invalid CRS'}})

            with self.assertRaises(InvalidTargetCRS) as context:
                has_valid_crs(test_message)

            self.assertEqual(context.exception.message,
                             'Target CRS not supported: "invalid CRS"')

    def test_is_geographic_crs(self):
        """ Ensure function correctly determines if a supplied string resolves
            to a `pyproj.CRS` object with a geographic Coordinate Reference
            System (CRS). Exceptions arising from invalid CRS strings should
            also be handled.

        """
        with self.subTest('"EPSG:4326" returns True'):
            self.assertTrue(_is_geographic_crs('EPSG:4326'))

        with self.subTest('"+proj=longlat" returns True'):
            self.assertTrue(_is_geographic_crs('+proj=longlat'))

        with self.subTest('"4326" returns True'):
            self.assertTrue(_is_geographic_crs('4326'))

        with self.subTest('Non-geographic EPSG code returns False'):
            self.assertFalse(_is_geographic_crs('EPSG:6933'))

        with self.subTest('Non-geographic proj4 string returns False'):
            self.assertFalse(_is_geographic_crs('+proj=cea'))

        with self.subTest('String that cannot be parsed raises exception'):
            with self.assertRaises(InvalidTargetCRS) as context:
                _is_geographic_crs('invalid CRS')

            self.assertEqual(context.exception.message,
                             'Target CRS not supported: "invalid CRS"')

    def test_has_valid_interpolation(self):
        """ Ensure that the function correctly determines if the supplied
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
            test_message = Message({
                'format': {'interpolation': 'Elliptical Weighted Averaging'}
            })
            self.assertTrue(has_valid_interpolation(test_message))

        with self.subTest('Unexpected interpolation returns False'):
            test_message = Message({'format': {'interpolation': 'Bilinear'}})
            self.assertFalse(has_valid_interpolation(test_message))
