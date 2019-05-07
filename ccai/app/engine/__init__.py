"""
:mod:`ccai.app.engine` -- API functionalities
=============================================

This module hosts the different functions to retrieve a location of
an address throught the `GoodleGeocoder` API and images of that location
through the `GoogleStreetView` API.

"""
import os

from ccai.config import Config
import google_streetview.api as sw_api
import google_streetview.helpers as sw_helpers
from googlegeocoder import GoogleGeocoder


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

    full_location = {'address': address,
                     'latitude': latitude,
                     'longitude': longitude
                     }

    full_location['_id'] = str(get_unique_id(location, full_location))

    return full_location


def get_unique_id(location, full_location):
    """Return a unique id for that location.

    A unique id is required for the directory containing the StreetView images.
    Create a string from the longitude and latitude of the address.

    Parameters
    ----------
    location: GeocoderResult
        The `GeocoderResult` returned by the geocoder object.
    full_location: dict
        Dictionary containing extracted information from the `location` object.

    Returns
    -------
    str:
        String of the latitude and longitude

    """
    string = ','.join([full_location['latitude'], full_location['longitude']])
    return string.replace("-", "_").replace(".", "_").replace(",", "_")


def fetch_street_view_images(location):
    """Retrieve StreetView images for the location.

    Create the parameters for the request and call the StreetView API to retrieve an image
    of the location and store it under a certain directory.

    Parameters
    ----------
    location: dict
        Dictionary returned from `find_location`.

    Returns
    -------
    str:
        The path to the directory containing the images.
    `google_streetview.api.results`:
        The results of the StreetView call.

    """
    params = create_params(location)
    image_dir = ensure_image_directory(location)

    api_list = sw_helpers.api_list(params)

    results = sw_api.results(api_list)
    results.download_links(image_dir)

    return image_dir, results


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
    lat_and_long = [location['latitude'], location['longitude']]
    stringified_location = ','.join(lat_and_long)

    params = {'size': '512x512',
              'location': stringified_location,
              'pitch': '0',
              'key': Config.STREET_VIEW_API_KEY
              }

    return params


def ensure_image_directory(location):
    """Make sure the image directory exists.

    Parameters
    ----------
    location: dict
        The dictionary returned by `find_location`.

    Returns
    -------
    str:
        Path to the image directory for this location.

    """
    image_dir = os.path.join(Config.BASE_DIR, Config.DOWNLOAD_DIR, location['global_code'])

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    return image_dir
