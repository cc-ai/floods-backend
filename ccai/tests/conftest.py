#!/usr/bin/env python
"""Common fixtures for the unittests"""
import copy
import os
import pytest
import yaml

from ccai.config import Config

from googlegeocoder import GeocoderResult

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MOCK_LOCATION_FILE = os.path.join(TEST_DIR, "mock_location.yaml")

@pytest.fixture
def mock_location():
    """Create a dictionary of a normal Geocoder request for mocked location."""
    with open(MOCK_LOCATION_FILE, 'r') as f:
        location = yaml.load(f)

    return location


@pytest.fixture
def geocoder_result(mock_location):
    """Create a GeocoderResult for mocking."""
    return GeocoderResult(copy.deepcopy(mock_location))
