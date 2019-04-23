#!/usr/bin/env python
"""Common fixtures for the unittests."""
import os

import pytest
import yaml

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MOCK_LOCATION_FILE = os.path.join(TEST_DIR, "mock_location.yaml")


@pytest.fixture
def mock_location():
    """Create a dictionary of a normal Geocoder request for mocked location."""
    with open(MOCK_LOCATION_FILE, 'r') as f:
        location = yaml.load(f)

    return location


@pytest.fixture
def mock_location_no_gc(mock_location):
    """Create same as above but without a `global_code`."""
    mock_location.pop('plus_code', None)
    return mock_location


@pytest.fixture
def mock_location_hash():
    """Return a deterministic hash for the mocked location."""
    return "45_5304828__73_61387789999999"
