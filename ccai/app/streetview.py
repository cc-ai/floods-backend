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

from ccai.app.config import Config


def find_location(address):
    """Find the coordinates of a location.

    Using `GoogleGeocode`, find the latitude and longitude of an address and extract important
    information for the StreetView API.

    Parameters
    ----------
    address: str
        The address of a location as sent to the API.

    Returns
    -------
    dict:
        Extracted information from the `GeocodeResult` object.

    """
    geocoder_api_key = Config.GEO_CODER_API_KEY
    geocoder = GoogleGeocoder(geocoder_api_key)

    location = geocoder.get(address)[0]
    latitude = str(location.geometry.location.lat)
    longitude = str(location.geometry.location.lng)

    full_location = {"address": address, "latitude": latitude, "longitude": longitude}

    full_location["_id"] = str(get_unique_id(full_location))

    return full_location


def get_unique_id(full_location):
    """Return a unique id for that location.

    A unique id is required for the directory containing the StreetView images.
    Create a string from the longitude and latitude of the address.

    Parameters
    ----------
    full_location: dict
        Dictionary containing extracted information from the `location` object.

    Returns
    -------
    str:
        String of the latitude and longitude

    """
    string = ",".join([full_location["latitude"], full_location["longitude"]])
    return string.replace("-", "_").replace(".", "_").replace(",", "_")


def fetch_street_view_images(location):
    """Retrieve StreetView images for the location.

    Create the parameters for the request and call the StreetView API to retrieve mutliple
    images of the location.

    Parameters
    ----------
    location: dict
        Dictionary returned from `find_location`.

    Returns
    -------
    `google_streetview.api.results`:
        The results of the StreetView call.

    """
    params = create_params(location)

    api_list = sw_helpers.api_list(params)

    results = sw_api.results(api_list)

    return results


def create_params(location):
    """Create the parameters for the StreetView API call.

    Parameters
    ----------
    location: dict
        The dictionary returned by `find_location`.

    Returns
    -------
    dict:
        A dictionary containing the necessary parameters for the StreetView API call.

    """
    lat_and_long = [location["latitude"], location["longitude"]]
    stringified_location = ",".join(lat_and_long)

    params = {
        "size": "512x512",
        "location": stringified_location,
        "pitch": "0",
        "key": Config.STREET_VIEW_API_KEY,
    }

    return params
