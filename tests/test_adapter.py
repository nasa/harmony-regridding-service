""" End-to-end tests of the Harmony Regridding service. """
from os.path import exists, join as path_join
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import patch

from harmony.message import Message
from harmony.util import config
from pystac import Catalog

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
        cls.access_token = 'fake-token'
        cls.granule_url = 'https://www.example.com/input.nc4'
        cls.input_stac = create_stac(Granule(cls.granule_url,
                                             'application/x-netcdf4',
                                             ['data']))
        cls.staging_location = 's3://example-bucket'
        cls.user = 'blightyear'

    def setUp(self):
        """ Define test fixtures that are not shared between tests. """
        self.temp_dir = mkdtemp()
        self.config = config(validate=False)

    def tearDown(self):
        if exists(self.temp_dir):
            rmtree(self.temp_dir)

    def assert_expected_output_catalog(self, catalog: Catalog,
                                       expected_href: str,
                                       expected_title: str,
                                       expected_media_type: str):
        """ Check the contents of the Harmony output STAC. It should have a
            single data item. The URL, title and media type for this asset will
            be compared to supplied values.

        """
        items = list(catalog.get_items())
        self.assertEqual(len(items), 1)
        self.assertListEqual(list(items[0].assets.keys()), ['data'])
        self.assertDictEqual(
            items[0].assets['data'].to_dict(),
            {'href': expected_href,
             'title': expected_title,
             'type': expected_media_type,
             'roles': ['data']}
        )

    @patch('harmony_regridding_service.adapter.rmtree')
    @patch('harmony_regridding_service.adapter.mkdtemp')
    @patch('harmony_regridding_service.adapter.download')
    @patch('harmony_regridding_service.adapter.stage')
    def test_valid_request(self, mock_stage, mock_download, mock_mkdtemp,
                           mock_rmtree):
        """ Ensure a request with a correctly formatted message is fully
            processed.

            This test will need updating when the service functions fully.

        """
        expected_downloaded_file = f'{self.temp_dir}/input.nc4'
        expected_output_basename = 'input_regridded.nc4'
        expected_staged_url = path_join(self.staging_location,
                                        expected_output_basename)
        mock_mkdtemp.return_value = self.temp_dir
        mock_download.return_value = expected_downloaded_file
        mock_stage.return_value = expected_staged_url

        message = Message({
            'accessToken': self.access_token,
            'callback': 'https://example.com/',
            'format': {
                'height': 181,
                'scaleExtent': {'x': {'min': -180, 'max': 180},
                                'y': {'min': -90, 'max': 90}},
                'width': 361
            },
            'sources': [{'collection': 'C1234-EEDTEST', 'shortName': 'test'}],
            'stagingLocation': self.staging_location,
            'user': self.user,
        })

        regridder = RegriddingServiceAdapter(message, config=self.config,
                                             catalog=self.input_stac)

        _, output_catalog = regridder.invoke()

        # Ensure the output catalog contains the single, expected item:
        self.assert_expected_output_catalog(output_catalog,
                                            expected_staged_url,
                                            expected_output_basename,
                                            'application/x-netcdf4')

        # Ensure a download was requested via harmony-service-lib:
        mock_download.assert_called_once_with(self.granule_url, self.temp_dir,
                                              logger=regridder.logger,
                                              cfg=regridder.config,
                                              access_token=self.access_token)

        # Ensure the file was staged as expected:
        mock_stage.assert_called_once_with(expected_downloaded_file,
                                           expected_output_basename,
                                           'application/x-netcdf4',
                                           logger=regridder.logger,
                                           location=self.staging_location,
                                           cfg=self.config)

        # Ensure container clean-up was requested:
        mock_rmtree.assert_called_once_with(self.temp_dir)

    @patch('harmony_regridding_service.adapter.rmtree')
    @patch('harmony_regridding_service.adapter.mkdtemp')
    @patch('harmony_regridding_service.adapter.download')
    @patch('harmony_regridding_service.adapter.stage')
    def test_missing_grid(self, mock_stage, mock_download, mock_mkdtemp,
                          mock_rmtree):
        """ Ensure a request that fails message validation correctly raises an
            exception that is reported at the top level of invocation. Message
            validation occurs prior to the `RegriddingServiceAdapter.process_item`
            method, so none of the functions or methods within that method
            should be called. In this test there are no target grid parameters,
            so the validation should raise an `InvalidTargetGrid` exception.

        """
        error_message = 'Insufficient or invalid target grid parameters.'

        harmony_message = Message({
            'accessToken': self.access_token,
            'callback': 'https://example.com/',
            'sources': [{'collection': 'C1234-EEDTEST', 'shortName': 'test'}],
            'stagingLocation': self.staging_location,
            'user': self.user,
        })

        regridder = RegriddingServiceAdapter(harmony_message, config=self.config,
                                             catalog=self.input_stac)

        with self.assertRaises(InvalidTargetGrid) as context_manager:
            regridder.invoke()

        # Ensure exception message was propagated back to the end-user:
        self.assertEqual(context_manager.exception.message, error_message)

        # Ensure no additional functions were called after the exception:
        mock_mkdtemp.assert_not_called()
        mock_download.assert_not_called()
        mock_stage.assert_not_called()
        mock_rmtree.assert_not_called()

    @patch('harmony_regridding_service.adapter.rmtree')
    @patch('harmony_regridding_service.adapter.mkdtemp')
    @patch('harmony_regridding_service.adapter.download')
    @patch('harmony_regridding_service.adapter.stage')
    def test_invalid_grid(self, mock_stage, mock_download, mock_mkdtemp,
                          mock_rmtree):
        """ Ensure a request that fails message validation correctly raises an
            exception that is reported at the top level of invocation. Message
            validation occurs prior to the `RegriddingServiceAdapter.process_item`
            method, so none of the functions or methods within that method
            should be called. In this test there ae target grid parameters that
            are inconsistent with one another, so the validation should raise
            an `InvalidTargetGrid` exception.

        """
        error_message = 'Insufficient or invalid target grid parameters.'

        harmony_message = Message({
            'accessToken': self.access_token,
            'callback': 'https://example.com/',
            'format': {
                'height': 234,
                'scaleExtent': {'x': {'min': -180, 'max': 180},
                                'y': {'min': -90, 'max': 90}},
                'scaleSize': {'x': 0.5, 'y': 0.5},
                'width': 123
            },
            'sources': [{'collection': 'C1234-EEDTEST', 'shortName': 'test'}],
            'stagingLocation': self.staging_location,
            'user': self.user,
        })

        regridder = RegriddingServiceAdapter(harmony_message, config=self.config,
                                             catalog=self.input_stac)

        with self.assertRaises(InvalidTargetGrid) as context_manager:
            regridder.invoke()

        # Ensure exception message was propagated back to the end-user:
        self.assertEqual(context_manager.exception.message, error_message)

        # Ensure no additional functions were called after the exception:
        mock_mkdtemp.assert_not_called()
        mock_download.assert_not_called()
        mock_stage.assert_not_called()
        mock_rmtree.assert_not_called()

    @patch('harmony_regridding_service.adapter.rmtree')
    @patch('harmony_regridding_service.adapter.mkdtemp')
    @patch('harmony_regridding_service.adapter.download')
    @patch('harmony_regridding_service.adapter.stage')
    def test_invalid_interpolation(self, mock_stage, mock_download,
                                   mock_mkdtemp, mock_rmtree):
        """ Ensure a request that fails message validation correctly raises an
            exception that is reported at the top level of invocation. Message
            validation occurs prior to the `RegriddingServiceAdapter.process_item`
            method, so none of the functions or methods within that method
            should be called. In this test there is an invalid interpolation in
            the Harmony message, so the validation should raise an
            `InvalidInterpolationMethod` exception.

        """
        error_message = 'Interpolation method not supported: "Bilinear"'

        harmony_message = Message({
            'accessToken': self.access_token,
            'callback': 'https://example.com/',
            'format': {
                'interpolation': 'Bilinear',
                'scaleExtent': {'x': {'min': -180, 'max': 180},
                                'y': {'min': -90, 'max': 90}},
                'scaleSize': {'x': 0.5, 'y': 0.5},
            },
            'sources': [{'collection': 'C1234-EEDTEST', 'shortName': 'test'}],
            'stagingLocation': self.staging_location,
            'user': self.user,
        })

        regridder = RegriddingServiceAdapter(harmony_message, config=self.config,
                                             catalog=self.input_stac)

        with self.assertRaises(InvalidInterpolationMethod) as context_manager:
            regridder.invoke()

        # Ensure exception message was propagated back to the end-user:
        self.assertEqual(context_manager.exception.message, error_message)

        # Ensure no additional functions were called after the exception:
        mock_mkdtemp.assert_not_called()
        mock_download.assert_not_called()
        mock_stage.assert_not_called()
        mock_rmtree.assert_not_called()

    @patch('harmony_regridding_service.adapter.rmtree')
    @patch('harmony_regridding_service.adapter.mkdtemp')
    @patch('harmony_regridding_service.adapter.download')
    @patch('harmony_regridding_service.adapter.stage')
    def test_invalid_crs(self, mock_stage, mock_download, mock_mkdtemp,
                         mock_rmtree):
        """ Ensure a request that fails message validation correctly raises an
            exception that is reported at the top level of invocation. Message
            validation occurs prior to the `RegriddingServiceAdapter.process_item`
            method, so none of the functions or methods within that method
            should be called. In this test there is an invalid target CRS
            specified in the Harmony message, so the validation should raise an
            `InvalidTargetCRS` exception.

        """
        error_message = 'Target CRS not supported: "invalid CRS"'

        harmony_message = Message({
            'accessToken': self.access_token,
            'callback': 'https://example.com/',
            'format': {
                'crs': 'invalid CRS',
                'scaleExtent': {'x': {'min': -180, 'max': 180},
                                'y': {'min': -90, 'max': 90}},
                'scaleSize': {'x': 0.5, 'y': 0.5},
            },
            'sources': [{'collection': 'C1234-EEDTEST', 'shortName': 'test'}],
            'stagingLocation': self.staging_location,
            'user': self.user,
        })

        regridder = RegriddingServiceAdapter(harmony_message, config=self.config,
                                             catalog=self.input_stac)

        with self.assertRaises(InvalidTargetCRS) as context_manager:
            regridder.invoke()

        # Ensure exception message was propagated back to the end-user:
        self.assertEqual(context_manager.exception.message, error_message)

        # Ensure no additional functions were called after the exception:
        mock_mkdtemp.assert_not_called()
        mock_download.assert_not_called()
        mock_stage.assert_not_called()
        mock_rmtree.assert_not_called()
