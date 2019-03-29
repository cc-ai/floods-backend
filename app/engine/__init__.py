from config import Config

from googlegeocoder import GoogleGeocoder
import google_streetview.api as sw_api
import google_streetview.helpers as sw_helpers

import os
import sys

def find_location(address):
    geocoder_api_key = Config.GEO_CODER_API_KEY
    geocoder = GoogleGeocoder(geocoder_api_key)

    location = geocoder.get(address)[0]
    latitude = str(location.geometry.location.lat)
    longitude = str(location.geometry.location.lng)

    full_location = {'address': address,
                     'latitude': latitude,
                     'longitude': longitude
                    }

    full_location['global_code'] = str(get_unique_id(location, full_location))

    return full_location


def get_unique_id(location, full_location):
    if hasattr(location, 'plus_code'):
        return location.plus_code.global_code

    h = hash(','.join([full_location['latitude'], full_location['longitude']]))
    h += sys.maxsize + 1
    return h


def fetch_street_view_images(location):
    params = create_params(location)
    image_dir = ensure_image_directory(location)

    api_list = sw_helpers.api_list(params)

    results = sw_api.results(api_list)
    results.download_links(image_dir)

    return image_dir, results

def create_params(location):
    lat_and_long = [location['latitude'], location['longitude']]
    stringified_location = ','.join(lat_and_long)

    params = {'size': '512x512',
              'location': stringified_location,
              'pitch': '0',
              'key': Config.STREET_VIEW_API_KEY
             }

    return params


def ensure_image_directory(location):
    image_dir = os.path.join(Config.BASE_DIR, Config.DOWNLOAD_DIR, location['global_code'])

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    return image_dir
