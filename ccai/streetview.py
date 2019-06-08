"""
:mod:`ccai.app.engine` -- API functionalities
=============================================

This module hosts the different functions to retrieve a location of
an address throught the `GoodleGeocoder` API and images of that location
through the `GoogleStreetView` API.

"""
import google_streetview.api as sw_api
import google_streetview.helpers as sw_helpers
from googlegeocoder import GoogleGeocoder

from ccai.config import Config

def fetch_street_view_image(address):
    """Retrieve StreetView images for the address."""
    location = find_location(address)
    params = create_params(location)
    api_list = sw_helpers.api_list(params)
    results = sw_api.results(api_list)
    return results


def find_location(address):
    """Find the coordinates of a location."""
    print("API KEY:", Config.GEO_CODER_API_KEY)
    geocoder = GoogleGeocoder(Config.GEO_CODER_API_KEY)

    location = geocoder.get(address)[0]
    latitude = str(location.geometry.location.lat)
    longitude = str(location.geometry.location.lng)

    full_location = {"address": address, "latitude": latitude, "longitude": longitude}
    full_location["_id"] = str(get_unique_id(full_location))
    return full_location


def get_unique_id(full_location):
    """Return a unique id for that location."""
    string = ",".join([full_location["latitude"], full_location["longitude"]])
    return string.replace("-", "_").replace(".", "_").replace(",", "_")


def create_params(location):
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
