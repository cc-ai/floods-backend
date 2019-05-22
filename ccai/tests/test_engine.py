"""Collection of tests for the `app.engine` module."""
import copy
from unittest.mock import Mock, patch

from ccai.app.engine import find_location, get_unique_id
from googlegeocoder import GeocoderResult


def test_get_unique_id(mock_location, mock_location_hash):
    """Test `get_unique_id` function."""
    location = ['geometry']['location']
    lat = location['lat']
    lng = location['lng']
    location_dict = {'latitude': lat, 'longitude': lng}
    _id = get_unique_id(location_dict)

    assert _id == mock_location_hash


def test_find_location(mock_location, mock_location_hash):
    """Test the `find_location` function."""
    with patch("ccai.app.engine.GoogleGeocoder", key="") as mock:
        geocoder_result = GeocoderResult(copy.deepcopy(mock_location))
        geocoder = mock.return_value
        geocoder.get.return_value = [geocoder_result]

        full_location = find_location("mila")

        assert full_location['address'] == "mila"
        assert full_location['latitude'] == mock_location['geometry']['location']['lat']
        assert full_location['longitude'] == mock_location['geometry']['location']['lng']
        assert full_location['_id'] == mock_location_hash

'''
@patch("tempfile.TemporaryDirectory")
def test_save_to_database(gridfs, mock_location_hash, mock_tmp):
    """Test if a file is correctly saved inside `GridFS`."""
    with tempfile.TemporaryDirectory as name:
        mock_tmp.return_value.__enter__.return_value = name
        location = {'_id': mock_location_hash}

        results = Mock()

        with patch.object(app.engine, "_get_download_name",
                          return_value=name + '/' + \
                                Config.SV_PREFIX.format(0)):
            '''
