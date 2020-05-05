import xarray as xr
import pandas as pd
import os, geopy.distance, time, json, textwrap

from googleplaces import GooglePlaces, types, lang
from ccai.config import FLOOD_LEVEL,FLOOD_MODE, RP
from ccai.climate.extractor import Extractor
from ccai.climate.frequency import shift_frequency
from ccai.config import CONFIG

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ds = xr.open_rasterio(os.path.join(BASE_DIR, CONFIG.CLIMATE_DATA))
flood_risk = 100/RP

def fetch_climate_data(address):
    extractor = Extractor()
    coordinates = extractor.coordinates_from_address(address)
    water_level, shift, address = fetch_water_level(coordinates, address)

    return water_level, shift, RP, int(flood_risk), address

def fetch_water_level(coordinates, address, band=1):
    water_level = ds.sel(band=band, x=coordinates.lon, y=coordinates.lat, method='nearest').values
    water_level = water_level * 100
    water_level = int(water_level)
    shift = shift_frequency(coordinates)

    if water_level < 0:
        water_level = "0"
        address = fetch_places(coordinates)
        # noflood_text(water_level, shift)

    elif FLOOD_LEVEL > water_level > 0:
        address = fetch_places(coordinates)
        # noflood_text(water_level, shift)

    else:
        # flood_text(water_level, shift)
        pass
    return water_level, shift, address

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

def flood_text(water_level,shift):
    print("################################################################")
    print("According to the climate models and flood simulations, your address is prone to")
    print("flooding with the return period of", RP, "years. The generated image shows how a flood with such")
    print("severity may look like. For this specific severity of flood, your home had a", int(flood_risk),"%")
    print("flood chance of experiencing", water_level, "cm of flooding during the time period of 1980-2010.")
    print("")
    print("However, due to global warming, this probability may be different in the future according")
    print("to the 'business as usual' global warming scenario, which may result in a 2 degree temperature rise by 2050.")
    print("Based on the climate models, the new chance of this flood is", round(shift, 1),"%.")
    print("################################################################")


def noflood_text(water_level, shift):
    print("################################################################")
    print("According to the climate models and flood simulations, your address is not prone to")
    print("flooding with the return period of", RP,"years. The generated image shows the closest local or")
    print("international landmark which is liable to be affected by severe flooding.")
    print("For this specific severity of flood, your home had a", int(flood_risk),"% chance of experiencing",water_level,"cm of")
    print("flooding during the time period of 1980-2010.")
    print("")
    print("However, due to global warming, this probability may be different in the future according to")
    print("the 'business as usual' global warming scenario, Based on the climate models, the new chance of")
    print("this flood is", round(shift, 1),"%.")
    print("################################################################")




