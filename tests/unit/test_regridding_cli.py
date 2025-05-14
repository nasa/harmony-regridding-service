"""Unit tests for regridding_cli.py."""

from unittest.mock import MagicMock, patch

from harmony_service_lib.message import Message as HarmonyMessage
from harmony_service_lib.message import Source as HarmonySource

from harmony_regridding_service.regridding_cli import regrid_cli_entry


@patch('harmony_regridding_service.regridding_cli.regrid')
def test_regrid_cli_entry(mock_regrid, message_params):
    """Test the regrid_cli_entry function."""
    source_filename = 'source_filename.nc'
    params = message_params
    source = {'collection': 'collection shortname'}
    call_logger = MagicMock()

    regrid_cli_entry(source_filename, params, source, call_logger)

    # Assert that regrid was called once
    mock_regrid.assert_called_once()

    # Get the arguments that regrid was called with
    args, kwargs = mock_regrid.call_args

    # Assert the arguments
    harmony_message_arg, input_filename_arg, source_arg, logger_arg = args

    assert isinstance(harmony_message_arg, HarmonyMessage)
    assert isinstance(source_arg, HarmonySource)
    assert input_filename_arg == source_filename
    assert logger_arg == call_logger
