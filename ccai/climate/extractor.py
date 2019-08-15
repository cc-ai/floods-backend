"""Helpers for extracting information from climate data"""

import os
from typing import Tuple

from dataclasses import dataclass
from googlegeocoder import GoogleGeocoder
import numpy as np
import pandas

from ccai.config import CONFIG
from ccai.singleton import Singleton


@dataclass
class Coordinates:
    """Data class for passing around lat and lng"""

    lat: float
    lon: float


@dataclass
class ClimateMetadata:
    """Data class for passing around climate metadata"""

    relative_change_precip: float
    monthly_average_precip: float


class Extractor(Singleton):
    """Class for extracting relevant climate data given an address or lat/long"""

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    LAT_PATH = os.path.join(BASE_DIR, "data/lat.csv")
    LON_PATH = os.path.join(BASE_DIR, "data/lon.csv")
    RELATIVE_COMPARED_TO_2012_PATH = os.path.join(BASE_DIR, "data/relativecomparedto2012.csv")
    YEAR_AVE_PATH = os.path.join(BASE_DIR, "data/yearave.csv")

    def __init__(self) -> None:
        Singleton.__init__(self)
        self.lat = pandas.read_csv(self.LAT_PATH, header=None, names=["lat"])
        self.lon = pandas.read_csv(self.LON_PATH, header=None, names=["lon"])
        self.relative_compared_to_2012 = pandas.read_csv(
            self.RELATIVE_COMPARED_TO_2012_PATH, header=None
        )
        self.year_ave = pandas.read_csv(self.YEAR_AVE_PATH, header=None)
        self.geocoder = GoogleGeocoder(CONFIG.GEO_CODER_API_KEY)

    def coordinates_from_address(self, address: str) -> Coordinates:
        """Find the lat and lng of a given address"""
        result = self.geocoder.get(address)
        return Coordinates(lat=result[0].geometry.location.lat, lon=result[0].geometry.location.lng)

    def indexes_for_coordinates(self, coordinates: Coordinates) -> Tuple[int, int]:
        """Calculate the indexes of the supplies coordinates in the lat.csv and
        lon.csv files"""
        lat = coordinates.lat
        lon = 360 + coordinates.lon

        lat_idx = 1
        for idx, value in self.lat.iterrows():
            if float(value["lat"]) < lat:
                break
            lat_idx = idx + 1

        lon_idx = 1
        for idx, value in self.lon.iterrows():
            if float(value["lon"]) > lon:
                break
            lon_idx = idx + 1

        return (lat_idx, lon_idx)

    def metadata_for_address(self, address: str) -> ClimateMetadata:
        """Calculate climate metadata for a given address"""
        coords = self.coordinates_from_address(address)
        indexes = self.indexes_for_coordinates(coords)
        try:
            relative_change_precip = self.relative_compared_to_2012.iloc[indexes[0], indexes[1]]
            if relative_change_precip == 0.0:
                relative_change_precip = None
            monthly_average_precip = self.year_ave.iloc[indexes[0], indexes[1]]
            if np.isnan(monthly_average_precip):
                monthly_average_precip = None
        except IndexError:
            relative_change_precip = None
            monthly_average_precip = None
        return ClimateMetadata(
            relative_change_precip=relative_change_precip,
            monthly_average_precip=monthly_average_precip,
        )
