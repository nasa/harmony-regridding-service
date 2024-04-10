from unittest import TestCase

from harmony.message import Message
from harmony.util import config, HarmonyException

from harmony_regridding_service.adapter import RegriddingServiceAdapter
from harmony_regridding_service.exceptions import (InvalidInterpolationMethod,
                                                   InvalidTargetCRS,
                                                   InvalidTargetGrid)
from tests.utilities import create_stac, Granule


class TestAdapter(TestCase):
    """ A class testing the harmony_regridding_service.utilities module. """
    @classmethod
    def setUpClass(cls):
        """ Define test fixtures that can be shared between tests. """
        cls.config = config(validate=False)
        cls.input_stac = create_stac(Granule('www.example.com/file.nc4',
                                             'application/x-netcdf4',
                                             ['data']))

    def test_validate_message(self):
        """ Ensure only messages with expected content will be processed. """
        valid_scale_extents = {'x': {'min': -180, 'max': 180},
                               'y': {'min': -90, 'max': 90}}
        valid_scale_sizes = {'x': 0.5, 'y': 1.0}
        valid_height = 181
        valid_width = 721

        with self.subTest('Valid grid, no CRS or interpolation is valid'):
            test_message = Message({
                'format': {'scaleExtent': valid_scale_extents,
                           'scaleSize': valid_scale_sizes}
            })
            harmony_adapter = RegriddingServiceAdapter(test_message, config=self.config,
                                                       catalog=self.input_stac)

            try:
                harmony_adapter.validate_message()
            except HarmonyException as exception:
                self.fail(f'Unexpected exception: {exception.message}')

        with self.subTest('Valid grid (scaleExtent and height/width) is valid'):
            test_message = Message({
                'format': {'crs': 'EPSG:4326',
                           'height': valid_height,
                           'interpolation': 'Elliptical Weighted Averaging',
                           'scaleExtent': valid_scale_extents,
                           'width': valid_width}
            })
            harmony_adapter = RegriddingServiceAdapter(test_message, config=self.config,
                                                       catalog=self.input_stac)

            try:
                harmony_adapter.validate_message()
            except HarmonyException as exception:
                self.fail(f'Unexpected exception: {exception.message}')

        with self.subTest('Valid grid and CRS, no interpolation is valid'):
            test_message = Message({
                'format': {'crs': 'EPSG:4326',
                           'scaleExtent': valid_scale_extents,
                           'scaleSize': valid_scale_sizes}
            })
            harmony_adapter = RegriddingServiceAdapter(test_message, config=self.config,
                                                       catalog=self.input_stac)

            try:
                harmony_adapter.validate_message()
            except HarmonyException as exception:
                self.fail(f'Unexpected exception: {exception.message}')

        with self.subTest('Valid grid and interpolation, no CRS is valid'):
            test_message = Message({
                'format': {'interpolation': 'Elliptical Weighted Averaging',
                           'scaleExtent': valid_scale_extents,
                           'scaleSize': valid_scale_sizes}
            })
            harmony_adapter = RegriddingServiceAdapter(test_message, config=self.config,
                                                       catalog=self.input_stac)

            try:
                harmony_adapter.validate_message()
            except HarmonyException as exception:
                self.fail(f'Unexpected exception: {exception.message}')

        with self.subTest('Valid grid, CRS and interpolation is valid'):
            test_message = Message({
                'format': {'crs': 'EPSG:4326',
                           'interpolation': 'Elliptical Weighted Averaging',
                           'scaleExtent': valid_scale_extents,
                           'scaleSize': valid_scale_sizes}
            })
            harmony_adapter = RegriddingServiceAdapter(test_message, config=self.config,
                                                       catalog=self.input_stac)

            try:
                harmony_adapter.validate_message()
            except HarmonyException as exception:
                self.fail(f'Unexpected exception: {exception.message}')

        with self.subTest('Inconsistent grid is not valid'):
            test_message = Message({
                'format': {'height': valid_height + 100,
                           'scaleExtent': valid_scale_extents,
                           'scaleSize': valid_scale_sizes,
                           'width': valid_width - 150}
            })
            harmony_adapter = RegriddingServiceAdapter(test_message, config=self.config,
                                                       catalog=self.input_stac)

            with self.assertRaises(InvalidTargetGrid) as context:
                harmony_adapter.validate_message()

            self.assertEqual(
                context.exception.message,
                'Insufficient or invalid target grid parameters.'
            )

        with self.subTest('Non-geographic CRS is not valid'):
            test_message = Message({
                'format': {'crs': 'EPSG:6933',
                           'scaleExtent': valid_scale_extents,
                           'scaleSize': valid_scale_sizes}
            })
            harmony_adapter = RegriddingServiceAdapter(test_message, config=self.config,
                                                       catalog=self.input_stac)

            with self.assertRaises(InvalidTargetCRS) as context:
                harmony_adapter.validate_message()

            self.assertEqual(context.exception.message,
                             'Target CRS not supported: "EPSG:6933"')

        with self.subTest('Non-EWA interpolation method is not valid'):
            test_message = Message({
                'format': {'interpolation': 'Bilinear',
                           'scaleExtent': valid_scale_extents,
                           'scaleSize': valid_scale_sizes}
            })
            harmony_adapter = RegriddingServiceAdapter(test_message, config=self.config,
                                                       catalog=self.input_stac)

            with self.assertRaises(InvalidInterpolationMethod) as context:
                harmony_adapter.validate_message()

            self.assertEqual(context.exception.message,
                             'Interpolation method not supported: "Bilinear"')
