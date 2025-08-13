"""Module for invoking the Harmony Regridding Service from command line."""

from logging import Logger

from harmony_service_lib.message import Source as HarmonySource

from harmony_regridding_service.message_utilities import get_harmony_message_from_params
from harmony_regridding_service.regridding_service import regrid


def regrid_cli_entry(
    source_filename: str, params: dict, source: dict, call_logger: Logger
):
    """Call regrid without the adapter class.

    TODO: This is where a library entry will exist if it is ever made.  In the
    meantime, this is an entrypoint to call the regridder without instantiating
    an adapter for testing.

    Args:
    source_filename: [str], a string to an input file to be regridded.

    params: [dict | None], A dictionary with the following keys:
        crs: [dict | None], Target image's Coordinate Reference System.
             A dictionary with 'epsg', 'proj4' or 'wkt' key.

        scale_extent: [dict | None], Scale Extents for the image. This dictionary
            contains "x" and "y" keys each whose value which is a dictionary
            of "min", "max" values in the same units as the crs.
            e.g.: { "x": { "min": 0.5, "max": 125 },
                    "y": { "min": 52, "max": 75.22 } }

        scale_size: [dict | None], Scale sizes for the image.  The dictionary
            contains "x" and "y" keys with the horizontal and veritcal
            resolution in the same units as the crs.
            e.g.: { "x": 10, "y": 10 }

        height: [int | None], height of the output image in gridcells.

        width: [int | none], width of the output image in gridcells.

    source: [dict | None], a Dictionary suitable for initializing a Harmony
            Source JsonObject.

    call_logger: [Logger], a configured logging object.

    """
    harmony_message = get_harmony_message_from_params(params)
    return regrid(harmony_message, source_filename, HarmonySource(source), call_logger)
