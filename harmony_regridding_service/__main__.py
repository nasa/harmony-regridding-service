"""Run the Harmony Regridding Service Adapter via the Harmony CLI."""

from argparse import ArgumentParser
from sys import argv

from harmony_service_lib import is_harmony_cli, run_cli, setup_cli

from harmony_regridding_service.adapter import RegriddingServiceAdapter


def main(arguments: list[str]):
    """Parse command line arguments and invoke the appropriate method."""
    parser = ArgumentParser(
        prog='harmony-regridding-service', description='Run Harmony regridding service.'
    )

    setup_cli(parser)
    harmony_arguments, _ = parser.parse_known_args(arguments[1:])

    if is_harmony_cli(harmony_arguments):
        run_cli(parser, harmony_arguments, RegriddingServiceAdapter)
    else:
        parser.error('Only --harmony CLIs are supported')


if __name__ == '__main__':
    main(argv)
