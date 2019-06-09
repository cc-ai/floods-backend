"""
Utilities for fetching photos from the Google Street View API
"""

import google_streetview.api as gsv_api
import google_streetview.helpers as gsv_helpers
from googlegeocoder import GoogleGeocoder


def fetch_street_view_image(
    address: str, geocoder_api_key: str, streetview_api_key: str
) -> gsv_api.results:
    """Retrieve StreetView images for the address."""
    geocoder = GoogleGeocoder(geocoder_api_key)
    result = geocoder.get(address)[0]
    latitude = str(result.geometry.location.lat)
    longitude = str(result.geometry.location.lng)
    params = {
        "size": "512x512",
        "location": ",".join([latitude, longitude]),
        "pitch": "0",
        "key": streetview_api_key,
    }
    api_list = gsv_helpers.api_list(params)
    results = gsv_api.results(api_list)
    return results
