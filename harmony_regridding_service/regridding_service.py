"""Regridding service code."""

from harmony_regridding_service.adapter import RegriddingServiceAdapter


def regrid(message: Message, input_data: str, logger: Logger) -> str:
    """Regrid."""
    return 'regrid'


def cache_resamplers(science_variables: List[str]) -> None:
    pass
