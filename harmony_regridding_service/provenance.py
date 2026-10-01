"""Functions for writing provenance metadata to regridded output."""

import json
from datetime import UTC, datetime
from pathlib import Path

from harmony_service_lib.message import Message as HarmonyMessage
from harmony_service_lib.message_utility import rgetattr
from netCDF4 import Dataset

# Values needed for the history_json attribute.
HISTORY_JSON_SCHEMA = (
    'https://harmony.earthdata.nasa.gov/schemas/history/0.1.0/history-v0.1.0.json'
)
PROGRAM = 'Harmony Regridding Service'
PROGRAM_REF = 'https://github.com/nasa/harmony-regridding-service'
VERSION_FILE = Path(__file__).resolve().parent.parent / 'docker' / 'service_version.txt'

# Harmony message `format` properties that determine the output grid.
REGRIDDING_PARAMETERS = (
    'crs',
    'interpolation',
    'scaleExtent',
    'scaleSize',
    'height',
    'width',
)


def update_history_metadata(
    source_ds: Dataset,
    target_ds: Dataset,
    message: HarmonyMessage,
    source_url: str,
) -> None:
    """Add provenance metadata describing this regridding operation.

    Reads any existing `history`/`history_json` provenance from the input
    file, appends a record for the current regridding operation, and writes
    the combined provenance to the global attributes of the output file.

    Two forms of history metadata are written to the root of `target_ds`:

    * `history_json` - a JSON string containing the structured provenance
      records. Existing records from the input file are preserved and the
      new regridding record is appended.

    * `history` (or `History`, matching whichever the input used) - a
      newline-delimited, human-readable summary. The new regridding entry is
      appended to any pre-existing history text.

    Args:
        source_ds: The input Dataset, read for existing provenance.
        target_ds: The output Dataset onto which provenance is written.
        message: The Harmony message, read for the regridding parameters.
        source_url: The location of the input granule, used as the
            `derived_from` value when no upstream request URL is available.

    """
    history_attribute_name, existing_history = read_history_attrs(source_ds)

    derived_from = get_request_url(source_ds) or strip_query_string(source_url)
    regridding_parameters = get_regridding_parameters(message)

    new_history_json_record = create_history_json_record(
        derived_from, regridding_parameters
    )

    output_history_json = read_history_json_attrs(source_ds)
    output_history_json.append(new_history_json_record)
    target_ds.setncattr('history_json', json.dumps(output_history_json))

    new_history_items = [
        new_history_json_record['date_time'],
        new_history_json_record['program'],
        new_history_json_record['version'],
    ]
    if regridding_parameters:
        new_history_items.append(json.dumps(regridding_parameters))

    target_ds.setncattr(
        history_attribute_name,
        '\n'.join(filter(None, [existing_history, ' '.join(new_history_items)])),
    )


def read_history_attrs(dataset: Dataset) -> tuple[str, str | None]:
    """Return the history attribute name in use and its existing value.

    Both `History` and `history` are possible. The casing of the input
    is preserved so the output does not end up with two competing attributes.

    Args:
        dataset: Dataset whose global attributes are inspected.

    Returns:
        A tuple of (attribute_name, existing_value). When no history attribute
        is present the name defaults to `history` and the value to None.

    """
    global_attributes = dataset.ncattrs()

    if 'History' in global_attributes:
        return 'History', dataset.getncattr('History')
    if 'history' in global_attributes:
        return 'history', dataset.getncattr('history')
    return 'history', None


def read_history_json_attrs(dataset: Dataset) -> list:
    """Return the existing `history_json` records as a list.

    The `history_json` attribute may be absent, a single record object, or a
    list of records. The return value is always normalized to a list so the
    new record can simply be appended.

    Args:
        dataset: Dataset whose global attributes are inspected.

    Returns:
        A list of existing `history_json` records, or an empty list when the
        attribute is absent.

    """
    if 'history_json' not in dataset.ncattrs():
        return []

    existing_history_json = json.loads(dataset.getncattr('history_json'))
    if isinstance(existing_history_json, list):
        return existing_history_json
    return [existing_history_json]


def get_request_url(dataset: Dataset) -> str | None:
    """Extract the source granule URL from the input's `history_json`.

    Returns the `request_url` recorded by an upstream service (for example,
    the OPeNDAP request that produced the input) in the first `history_json`
    record, with any query string removed.

    Args:
        dataset: The input Dataset.

    Returns:
        The upstream request URL without query parameters, or None when no
        `history_json` attribute or `request_url` is available.

    """
    history_json = read_history_json_attrs(dataset)
    if not history_json or not isinstance(history_json[0], dict):
        return None

    parameters = history_json[0].get('parameters')

    if isinstance(parameters, dict) and 'request_url' in parameters:
        return strip_query_string(parameters['request_url'])

    if isinstance(parameters, list):
        for item in parameters:
            if isinstance(item, dict) and 'request_url' in item:
                return strip_query_string(item['request_url'])

    return None


def strip_query_string(url: str) -> str:
    """Return the URL without any query string."""
    return url.split('?', 1)[0]


def get_regridding_parameters(message: HarmonyMessage) -> dict:
    """Return the Harmony message parameters that determine the output grid.

    Only parameters that were specified in the message are returned, using
    the property names of the Harmony message `format` object, so that
    the record reflects the request that produced the output.

    Args:
        message: The Harmony message for the current request.

    Returns:
        A dictionary of the specified regridding parameters. It will be empty
        when the message did not specify any of them.

    """
    format_data = rgetattr(message, 'format.data') or {}

    return {
        parameter: format_data[parameter]
        for parameter in REGRIDDING_PARAMETERS
        if format_data.get(parameter) is not None
    }


def create_history_json_record(derived_from: str, parameters: dict) -> dict:
    """Build a single `history_json` record for this regridding operation.

    Args:
        derived_from: The source granule the output is derived from.
        parameters: The regridding parameters from the Harmony message, if they exist.

    Returns:
        A dictionary describing the operation, ready to be serialized into the
        `history_json` attribute.

    """
    history_json_record = {
        '$schema': HISTORY_JSON_SCHEMA,
        'date_time': datetime.now(UTC).isoformat(),
        'program': PROGRAM,
        'version': get_semantic_version(),
        'derived_from': derived_from,
        'program_ref': PROGRAM_REF,
    }

    if parameters:
        history_json_record['parameters'] = parameters

    return history_json_record


def get_semantic_version() -> str:
    """Return the service semantic version from `docker/service_version.txt`."""
    return VERSION_FILE.read_text(encoding='utf-8').strip()
