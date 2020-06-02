import xarray as xr
import pandas as pd
import os, geopy.distance, time, json

from ccai.climate import extractor

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TIF_PATH = os.path.join(BASE_DIR, "data/floodMapGL_rp50y.tif")

ds = xr.open_rasterio(TIF_PATH)
water_min = 40
mode = "asclosest"  # simple, closest, asclosest
revolution = "landmark"  # landmark, political


def waterize(coords, band=1):

    water_level = ds.sel(band=band, x=coords.lon, y=coords.lat, method="nearest").values

    if water_level < 0 or water_level is None:
        water_level = 0
        return water_level

    else:
        water_level_expected = water_level.tolist()
        water_level_expected = int(water_level_expected * 100)
        return water_level_expected


def spiralize(coords):

    init_water_level = 0
    water_level = init_water_level
    init_lon = coords.lon
    init_lat = coords.lat
    geo_factor = 0.001
    geo_cursor = 0
    start = time.time()
    time_out = 2

    while water_level < water_min:

        geo_cursor = geo_cursor + 1

        for i in range(0, geo_cursor):
            if water_level < water_min:
                coords.lon = coords.lon + geo_factor
                water_level = waterize(coords)

        for i in range(0, geo_cursor):
            if water_level < water_min:
                coords.lat = coords.lat - geo_factor
                water_level = waterize(coords)

        geo_cursor = geo_cursor + 1

        for i in range(0, geo_cursor):
            if water_level < water_min:
                coords.lon = coords.lon - geo_factor
                water_level = waterize(coords)

        for i in range(0, geo_cursor):
            if water_level < water_min:
                coords.lat = coords.lat + geo_factor
                water_level = waterize(coords)

        if time.time() > start + time_out:

            water_level = init_water_level
            coords.lon = init_lon
            coords.lat = init_lat
            revolutionize(coords)

            return water_level, coords

    return water_level, coords


def distansize(init_lat, init_lon, coords):

    coords_1 = (init_lat, init_lon)
    coords_2 = (coords.lat, coords.lon)

    distance = geopy.distance.distance(coords_1, coords_2).km
    distance = round(distance, 2)

    return distance


def revolutionize(coords):

    if revolution == "landmark":
        print("landmark")

    elif revolution == "political":
        print("political")

    else:
        pass


def climatize(coords):

    init_lon = coords.lon
    init_lat = coords.lat

    water_level = waterize(coords)

    if mode is "asclosest":
        if water_level < water_min:
            water_level, coords = spiralize(coords)
            distance = distansize(init_lat, init_lon, coords)
        else:
            distance = "0"

    else:
        distance = "0"

    return water_level, distance
