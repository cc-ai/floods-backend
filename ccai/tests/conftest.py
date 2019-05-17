#!/usr/bin/env python
"""Common fixtures for the unittests."""
import os

from gridfs import GridFS
from pymongo import MongoClient
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
def mock_location_hash():
    """Return a deterministic hash for the mocked location."""
    return "45_5304828__73_61387789999999"


@pytest.fixture
def database():
    """Return a new empty database"""
    return MongoClient()['database']


@pytest.fixture
def gridfs(database):
    """Return a GridFS instance"""
    return GridFS(database, collection="test")


@pytest.fixture
def image_name():
    return ""
