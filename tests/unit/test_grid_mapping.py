"""Tests for determining CRS from source data."""

from unittest.mock import Mock, patch

import pytest
from pyproj import CRS

from harmony_regridding_service.grid import (
    InvalidSourceCRS,
    crs_from_source_data,
    get_grid_mapping_attributes,
)


class TestGetGridMappingAttributes:
    """Tests for get_grid_mapping_attributes."""

    def test_single_grid_mapping_variable(self):
        """Test retrieving attributes from a single grid mapping variable."""
        mock_test_variable = Mock()
        mock_test_variable.references = {
            'grid_mapping': {'/EASE2_global_projection'},
            'coordinates': set(),
        }

        mock_grid_mapping_variable = Mock()
        expected_attrs = {
            'grid_mapping_name': 'lambert_cylindrical_equal_area',
            'standard_parallel': 30.0,
            'false_easting': 0.0,
            'false_northing': 0.0,
            'longitude_of_central_meridian': 0.0,
        }

        mock_grid_mapping_variable.attributes = expected_attrs

        # Configure mock_var_info to return appropriate variables
        mock_var_info = Mock()
        mock_var_info.get_variable.side_effect = lambda name: {
            'test_var_name': mock_test_variable,
            '/EASE2_global_projection': mock_grid_mapping_variable,
        }[name]

        actual_attrs = get_grid_mapping_attributes('test_var_name', mock_var_info)

        assert actual_attrs == expected_attrs
        mock_var_info.get_variable.assert_any_call('test_var_name')
        mock_var_info.get_variable.assert_any_call('/EASE2_global_projection')

    def test_grid_mapping_with_coordinates_filtered_out(self):
        """Test that coordinates are excluded from grid mapping variables."""
        mock_variable = Mock()
        mock_variable.references = {
            'grid_mapping': {'/EASE2_global_projection', '/cell_lat', '/cell_lon'},
            'coordinates': {'/cell_lat', '/cell_lon'},
        }

        mock_gm_variable = Mock()
        expected_attrs = {'grid_mapping_name': 'latitude_longitude'}
        mock_gm_variable.attributes = expected_attrs

        mock_var_info = Mock()
        mock_var_info.get_variable.side_effect = lambda name: {
            'test_var': mock_variable,
            '/EASE2_global_projection': mock_gm_variable,
        }[name]

        actual_attrs = get_grid_mapping_attributes('test_var', mock_var_info)

        assert actual_attrs == expected_attrs
        mock_var_info.get_variable.assert_any_call('test_var')
        mock_var_info.get_variable.assert_any_call('/EASE2_global_projection')

    def test_no_grid_mapping_reference(self):
        """Test when variable has no grid_mapping reference."""
        mock_variable = Mock()
        mock_variable.references = {
            'coordinates': {'/lat', '/lon'}
            # No 'grid_mapping' key
        }

        mock_var_info = Mock()
        mock_var_info.get_variable.return_value = mock_variable

        result = get_grid_mapping_attributes('test_var', mock_var_info)

        assert result == {}

    def test_empty_grid_mapping_set(self):
        """Test when grid_mapping reference is empty."""
        mock_variable = Mock()
        mock_variable.references = {'grid_mapping': set(), 'coordinates': set()}

        mock_var_info = Mock()
        mock_var_info.get_variable.return_value = mock_variable

        result = get_grid_mapping_attributes('test_var', mock_var_info)

        assert result == {}

    def test_multiple_grid_mapping_variables_after_filtering(self):
        """Multiple grid mapping variables remain after coordinate filtering."""
        mock_variable = Mock()
        mock_variable.references = {
            'grid_mapping': {'/some_wrong_variable_is_here', '/grid_mapping_var'},
            'coordinates': set(),
        }

        mock_var_info = Mock()
        mock_var_info.get_variable.return_value = mock_variable

        result = get_grid_mapping_attributes('test_var', mock_var_info)

        assert result == {}  # Should return empty dict when not exactly 1 variable


class TestCrsFromSourceData:
    """Exhaustive tests for crs_from_source_data."""

    def ease_attritbutes(cls):
        return {
            'grid_mapping_name': 'lambert_cylindrical_equal_area',
            'standard_parallel': 30.0,
            'false_easting': 0.0,
            'false_northing': 0.0,
            'longitude_of_central_meridian': 0.0,
        }

    @pytest.fixture
    def mock_var_info(self):
        """Create a mock VarInfoFromNetCDF4 object."""
        return Mock()

    @patch('harmony_regridding_service.grid.get_grid_mapping_attributes')
    def test_successful_crs_from_ease_grid_mapping(self, mock_get_attrs, mock_var_info):
        """Test successful CRS creation from EASE2 grid_mapping metadata."""
        mock_get_attrs.return_value = self.ease_attritbutes()

        actual_crs = crs_from_source_data(['test_var'], mock_var_info)

        assert actual_crs.to_epsg() == 6933

    @patch('harmony_regridding_service.grid.get_grid_mapping_attributes')
    def test_multiple_variables_first_valid_wins(self, mock_get_attrs, mock_var_info):
        """Test that the first variable with valid grid_mapping is used."""
        # First variable has no grid mapping, second has valid mapping
        mock_get_attrs.side_effect = [{}, {}, self.ease_attritbutes()]

        actual_crs = crs_from_source_data(['var1', 'var2', 'var3'], mock_var_info)

        assert actual_crs.to_epsg() == 6933
        assert mock_get_attrs.call_count == 3

    @patch('harmony_regridding_service.grid.get_grid_mapping_attributes')
    def test_crs_error_from_cf_metadata(self, mock_get_attrs, mock_var_info):
        """Test handling of CRSError when creating CRS from CF metadata."""
        mock_get_attrs.return_value = {'invalid': 'metadata'}

        with pytest.raises(
            InvalidSourceCRS, match='Could not create a CRS from grid_mapping metadata'
        ):
            crs_from_source_data(['test_var'], mock_var_info)

    @patch('harmony_regridding_service.grid.has_geographic_grid_dimensions')
    @patch('harmony_regridding_service.grid.get_grid_mapping_attributes')
    def test_fallback_to_geographic_epsg_4326(
        self, mock_get_attrs, mock_has_geo_dims, mock_var_info
    ):
        """Fallback to EPSG:4326 with no grid_mapping but geographic dimensions."""
        mock_get_attrs.return_value = {}  # No grid mapping attributes
        mock_has_geo_dims.return_value = True  # is geographic

        expected_crs = CRS.from_epsg(4326)

        actual_crs = crs_from_source_data(['test_var', 'test_var2'], mock_var_info)

        assert actual_crs == expected_crs

    @patch('harmony_regridding_service.grid.has_geographic_grid_dimensions')
    @patch('harmony_regridding_service.grid.get_grid_mapping_attributes')
    def test_no_discernable_crs_raises(
        self, mock_get_attrs, mock_has_geo_dims, mock_var_info
    ):
        """Fallback to EPSG:4326 with no grid_mapping but geographic dimensions."""
        mock_get_attrs.return_value = {}  # No grid mapping attributes
        mock_has_geo_dims.return_value = False  #  is projected

        with pytest.raises(InvalidSourceCRS, match='No grid_mapping metadata found.'):
            crs_from_source_data(['test_var'], mock_var_info)

    def test_empty_variables_list(self, mock_var_info):
        """Empty variables list."""
        with pytest.raises(InvalidSourceCRS, match='No grid_mapping metadata found'):
            crs_from_source_data([], mock_var_info)


def test_crs_from_source_data_expected_case(smap_projected_netcdf_file, var_info_fxn):
    var_info = var_info_fxn(smap_projected_netcdf_file)
    expected_crs = CRS('epsg:6933')
    actual_crs = crs_from_source_data({'/Forecast_Data/sm_profile_forecast'}, var_info)
    assert actual_crs.equals(expected_crs, ignore_axis_order=True)
