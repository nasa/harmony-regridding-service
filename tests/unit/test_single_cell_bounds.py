"""Regression tests for one-row and one-column regridding targets."""

from shutil import copyfile

import numpy as np
import pytest
from harmony_service_lib.message import Message, Source
from netCDF4 import Dataset
from numpy.testing import assert_allclose, assert_array_equal

from harmony_regridding_service.regridding_service import regrid
from harmony_regridding_service.resample import (
    copy_resampled_bounds_variable,
    copy_resampled_dimension_variables,
    transfer_resampled_dimensions,
)


@pytest.mark.parametrize('width,height', [(1, 1), (1, 4), (5, 1), (5, 4)])
@pytest.mark.parametrize('axis', ['lon', 'lat'])
def test_bounds_span_target_cells(
    tmp_path, test_IMERG_ncfile, test_area_fxn, var_info_fxn, width, height, axis
):
    """Both singleton and multi-cell bounds derive from the target extent."""
    area = test_area_fxn(width=width, height=height, area_extent=(-10, -8, 10, 8))
    areas = {('/Grid/lon', '/Grid/lat'): area}
    var_info = var_info_fxn(test_IMERG_ncfile)
    output = tmp_path / 'bounds.nc'
    original = test_IMERG_ncfile.read_bytes()
    name = f'/Grid/{axis}_bnds'
    with Dataset(test_IMERG_ncfile) as source, Dataset(output, 'w') as target:
        transfer_resampled_dimensions(source, target, areas, var_info)
        assert copy_resampled_bounds_variable(
            source, target, name, areas, var_info
        ) == {name}
    edges = (
        np.linspace(-10, 10, width + 1)
        if axis == 'lon'
        else np.linspace(8, -8, height + 1)
    )
    expected = np.column_stack((edges[:-1], edges[1:]))
    with Dataset(output) as result:
        actual = result[name][:]
        assert_allclose(actual, expected, rtol=0, atol=1e-6)
        assert actual.shape == (width if axis == 'lon' else height, 2)
        assert_array_equal(actual[:-1, 1], actual[1:, 0])
        assert result[name].filters()['zlib']
        assert result[name].filters()['complevel'] == 6
    assert test_IMERG_ncfile.read_bytes() == original


@pytest.mark.parametrize('width,height', [(1, 1), (1, 4), (5, 1), (5, 4)])
def test_dimension_transfer_includes_centres_and_bounds(
    tmp_path, test_IMERG_ncfile, test_area_fxn, var_info_fxn, width, height
):
    """The public dimension-copy path writes both coordinates and their bounds."""
    area = test_area_fxn(width=width, height=height, area_extent=(-10, -8, 10, 8))
    areas = {('/Grid/lon', '/Grid/lat'): area}
    info = var_info_fxn(test_IMERG_ncfile)
    output = tmp_path / 'coordinates.nc'
    with Dataset(test_IMERG_ncfile) as source, Dataset(output, 'w') as target:
        transfer_resampled_dimensions(source, target, areas, info)
        copied = copy_resampled_dimension_variables(source, target, areas, info)
        assert copied == {'/Grid/lon', '/Grid/lat', '/Grid/lon_bnds', '/Grid/lat_bnds'}
    with Dataset(output) as result:
        for axis, coords in [
            ('lon', area.projection_x_coords),
            ('lat', area.projection_y_coords),
        ]:
            assert_allclose(result[f'/Grid/{axis}'][:], coords)
            bounds = result[f'/Grid/{axis}_bnds'][:]
            assert_allclose(bounds.mean(axis=1), coords)


@pytest.mark.parametrize('width,height', [(1, 1), (1, 4), (5, 1)])
def test_regrid_writes_single_cell_axes(
    tmp_path, monkeypatch, test_IMERG_ncfile, width, height
):
    """Run the actual regridder on a fixture, without replacing its resampler."""
    monkeypatch.chdir(tmp_path)
    source_path = tmp_path / 'input.nc'
    copyfile(test_IMERG_ncfile, source_path)
    original = source_path.read_bytes()
    message = Message(
        {
            'format': {
                'crs': 'EPSG:4326',
                'width': width,
                'height': height,
                'scaleExtent': {
                    'x': {'min': -10, 'max': 10},
                    'y': {'min': -8, 'max': 8},
                },
            }
        }
    )
    result_path = regrid(
        message,
        str(source_path),
        Source({'shortName': 'GPM_3IMERGHH', 'variables': []}),
    )
    with Dataset(result_path) as result:
        assert result['/Grid/lon'].shape == (width,)
        assert result['/Grid/lat'].shape == (height,)
        assert_allclose(
            result['/Grid/lon_bnds'][[0, -1], [0, 1]],
            [[-10, -10 + 20 / width], [10 - 20 / width, 10]],
        )
        assert_allclose(
            result['/Grid/lat_bnds'][[0, -1], [0, 1]],
            [[8, 8 - 16 / height], [-8 + 16 / height, -8]],
        )
    assert source_path.read_bytes() == original
