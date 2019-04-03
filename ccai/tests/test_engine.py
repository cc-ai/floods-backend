"""Collection of tests for the `app.engine` module."""
from unittest.mock import patch
import yaml

from ccai.app.engine import find_location

def test_find_location(mock_location, geocoder_result):
    with patch("googlegeocoder.GoogleGeocoder") as mock:
        geocoder = mock.return_value
        geocoder.get.return_value = [geocoder_result]

        full_location = find_location("mila")

        assert full_location['address'] == "mila"
        assert full_location['latitude'] == mock_location['geometry']['location']['lat']
        assert full_location['longitude'] == mock_location['geometry']['location']['lng']
        assert full_location['global_code'] == mock_location['plus_code']['global_code']
