"""
This module hosts the different functions to retrieve a location of
an address throught the GoodleGeocoder API and images of that location
through the GoogleStreetView API.
"""

from typing import Dict

import google_streetview.api as gsv_api
import google_streetview.helpers as gsv_helpers
from googlegeocoder import GoogleGeocoder

from ccai.config import Config


def fetch_street_view_image(address: str) -> gsv_api.results:
    """Retrieve StreetView images for the address."""
    location = find_location(address)
    params = create_params(location)
    api_list = gsv_helpers.api_list(params)
    results = gsv_api.results(api_list)
    return results


def find_location(address: str) -> Dict[str, str]:
    """Find the coordinates of a location."""
    geocoder = GoogleGeocoder(Config.GEO_CODER_API_KEY)
    location = geocoder.get(address)[0]
    lat = str(location.geometry.location.lat)
    lng = str(location.geometry.location.lng)
    full_location = {
        "_id": get_unique_id(lat, lng),
        "address": address,
        "latitude": lat,
        "longitude": lng,
    }
    return full_location


def get_unique_id(lat: str, lng: str) -> str:
    """Return a unique id for that location."""
    string = ",".join([lat, lng])
    return string.replace("-", "_").replace(".", "_").replace(",", "_")


def create_params(location: Dict[str, str]) -> Dict[str, str]:
    """Create the parameters for the StreetView API call."""
    lat_and_long = [location["latitude"], location["longitude"]]
    stringified_location = ",".join(lat_and_long)

    params = {
        "size": "512x512",
        "location": stringified_location,
        "pitch": "0",
        "key": Config.STREET_VIEW_API_KEY,
    }

    return params
