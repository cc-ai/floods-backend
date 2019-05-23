"""
:mod:`ccai.app.engine` -- API functionalities
=============================================

This module hosts the different functions to retrieve a location of
an address throught the `GoodleGeocoder` API and images of that location
through the `GoogleStreetView` API.

"""
import tempfile

from ccai.app.database import save_image
from ccai.config import Config
import google_streetview.api as sw_api
import google_streetview.helpers as sw_helpers
from googlegeocoder import GoogleGeocoder
from googleplaces import GooglePlaces


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


def auto_complete(input):
    google_places = GooglePlaces(Config.GOOGLE_PLACE_API_KEY)
    radius = 5000

    query_result = google_places.autocomplete(input=input, radius=radius)
    return [place.description for place in query_result.predictions]


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
    lat_and_long = [location['latitude'], location['longitude']]
    stringified_location = ','.join(lat_and_long)

    params = {'size': '512x512',
              'location': stringified_location,
              'pitch': '0',
              'key': Config.STREET_VIEW_API_KEY
              }

    return params


def save_to_database(location, results):
    """Save the results from Google StreetView to the database.

    Create a temporary directory to download the images then put them inside
    the database using `GridFS`.

    Parameters
    ----------
    location: dict
        Dictionary returned from `find_location`.
    results : `google_streetview.api.results`
        Results from `fetch_street_view_images`.

    Returns
    -------
    str
        Filename of the image inserted in the database.

    """
    # Create a temporary directory to download images
    with tempfile.TemporaryDirectory() as tmpdirname:
        results.download_links(tmpdirname)
        # The name of the downloaded StreetView images
        download_name = _get_download_name(tmpdirname)

        # The name of the image file inside the database
        # Since we only fetch one image, no need to ensure uniquess
        filename = location['_id']

        save_image(open(download_name, 'rb'), filename=filename)
        return filename


def _get_download_name(dirname):
    """Return the download name of an image inside dir.

    Parameters
    ----------
    dirname: str
        The name of the directory in which the images resides.

    Returns
    -------
    str
        The full path of the image relative to `dirname`.

    """
    return dirname + '/' + Config.SV_PREFIX.format('0')
