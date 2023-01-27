""" End-to-end tests of the Harmony Regridding service. """
from os.path import exists, join as path_join
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import patch

from harmony.message import Message
from harmony.util import config
from pystac import Catalog

from harmony_service.adapter import HarmonyAdapter

from tests.utilities import create_stac, Granule


class TestAdapter(TestCase):
    """ A class for testing the harmony_service.utilities module. """
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

    @patch('harmony_service.adapter.rmtree')
    @patch('harmony_service.adapter.mkdtemp')
    @patch('harmony_service.adapter.download')
    @patch('harmony_service.adapter.stage')
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
            'sources': [{'collection': 'C1234-EEDTEST', 'shortName': 'test'}],
            'stagingLocation': self.staging_location,
            'user': self.user,
        })

        regridder = HarmonyAdapter(message, config=self.config,
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

    @patch('harmony_service.adapter.rmtree')
    @patch('harmony_service.adapter.mkdtemp')
    @patch('harmony_service.adapter.download')
    @patch('harmony_service.adapter.stage')
    def test_invalid_request(self, mock_stage, mock_download, mock_mkdtemp,
                             mock_rmtree):
        """ Ensure a request that raises an exception correctly captures that
            exception, re-raises it, before attempting clean-up of the working
            directory.

            This test will need updating when the service functions fully.

        """
        error_message = 'Test exception'
        mock_mkdtemp.return_value = self.temp_dir
        mock_download.side_effect = Exception(error_message)

        message = Message({
            'accessToken': self.access_token,
            'callback': 'https://example.com/',
            'sources': [{'collection': 'C1234-EEDTEST', 'shortName': 'test'}],
            'stagingLocation': self.staging_location,
            'user': self.user,
        })

        regridder = HarmonyAdapter(message, config=self.config,
                                   catalog=self.input_stac)

        with self.assertRaises(Exception) as context_manager:
            regridder.invoke()

        # Ensure exception message was propagated back to the end-user:
        self.assertEqual(str(context_manager.exception), error_message)

        # Ensure a download was requested via harmony-service-lib:
        mock_download.assert_called_once_with(self.granule_url, self.temp_dir,
                                              logger=regridder.logger,
                                              cfg=regridder.config,
                                              access_token=self.access_token)

        # Ensure no additional functions were called after the exception:
        mock_stage.assert_not_called()

        # Ensure container clean-up was still requested in finally block:
        mock_rmtree.assert_called_once_with(self.temp_dir)
