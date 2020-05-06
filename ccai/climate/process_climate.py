import xarray as xr
import pandas as pd
import os, geopy.distance, time, json, textwrap

from googleplaces import GooglePlaces, types, lang
from ccai.config import FLOOD_LEVEL,FLOOD_MODE, RP
from ccai.climate.extractor import Extractor
from ccai.climate.frequency import shift_frequency
from ccai.climate.coastal import fetch_coastal
from ccai.config import CONFIG

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ds = xr.open_rasterio(os.path.join(BASE_DIR, CONFIG.CLIMATE_DATA))
flood_risk = 100/RP

def fetch_climate_data(address):
    extractor = Extractor()
    coordinates = extractor.coordinates_from_address(address)
    water_level, address = fetch_water_level(coordinates, address)
    shift = shift_frequency(coordinates)
    coastal = fetch_coastal(coordinates)

    return water_level, shift, RP, int(flood_risk), coastal, address

def fetch_water_level(coordinates, address, band=1):
    water_level = ds.sel(band=band, x=coordinates.lon, y=coordinates.lat, method='nearest').values
    water_level = water_level * 100
    water_level = int(water_level)


    if water_level < 0:
        water_level = "0"
        address = fetch_places(coordinates)

    elif FLOOD_LEVEL > water_level > 0:
        address = fetch_places(coordinates)

    else:
        pass

    return water_level, address

def fetch_places(coordinates):

    google_places = GooglePlaces(CONFIG.STREET_VIEW_API_KEY)

    query_result = google_places.nearby_search(
        lat_lng={"lat": coordinates.lat, "lng": coordinates.lon},
        keyword='landmark', radius=20000, types=[types.TYPE_TOURIST_ATTRACTION])

    place_count = 0
    for place in query_result.places:
        if place_count is 0:
            place.get_details()
            place_count = place_count + 1

        return place.formatted_address




