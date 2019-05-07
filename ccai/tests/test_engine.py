"""Collection of tests for the `app.engine` module."""
import copy
from unittest.mock import patch

from ccai.app.engine import find_location
from googlegeocoder import GeocoderResult


def test_find_location_without_group_code(mock_location_no_gc, mock_location_hash):
    """Test the `find_location` function where the geocoder result has no group_code."""
    with patch("ccai.app.engine.GoogleGeocoder", key="") as mock:
        geocoder_result = GeocoderResult(copy.deepcopy(mock_location_no_gc))
        geocoder = mock.return_value
        geocoder.get.return_value = [geocoder_result]

        full_location = find_location("mila")

        assert full_location['address'] == "mila"
        assert full_location['latitude'] == mock_location_no_gc['geometry']['location']['lat']
        assert full_location['longitude'] == mock_location_no_gc['geometry']['location']['lng']
        assert full_location['_id'] == mock_location_hash
