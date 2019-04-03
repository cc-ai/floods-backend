"""Common fixtures for the unittests"""
import pytest
import yaml

from googlegeocoder import GeocoderResult

@pytest.fixture
def mock_location():
    """Create a dictionary of a normal Geocoder request for mocked location."""
    with open('mock_location.yaml', 'r') as f:
        location = yaml.load(f)

    return f


@pytest.fixture
def geocoder_result(mock_location):
    """Create a GeocoderResult for mocking."""
    return GeocoderResult(mock_location)
