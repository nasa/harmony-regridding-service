from unittest import skip, TestCase

from harmony.util import config

from tests.utilities import create_stac, Granule


class TestAdapter(TestCase):
    """ A class for testing the harmony_service.utilities module. """
    @classmethod
    def setUpClass(cls):
        """ Define test fixtures that can be shared between tests. """
        cls.config = config(validate=False)
        cls.input_stac = create_stac(Granule('www.example.com/file.nc4',
                                             'application/x-netcdf4',
                                             ['data']))

    @skip('Method not yet implemented')
    def test_validate_message(self):
        """ Ensure only messages with expected content will be processed. """
