"""Helpers for extracting information from climate data"""

import os
from typing import Tuple

from dataclasses import dataclass
from googlegeocoder import GoogleGeocoder
import numpy as np
import pandas

from ccai.config import CONFIG
from ccai.singleton import Singleton
import ccai.climate.process_climate


@dataclass
class Coordinates:
    """Data class for passing around lat and lng"""

    lat: float
    lon: float

@dataclass
class ClimateMetadata:
    """Data class for passing around climate metadata"""

    water_level : float
    shift : float
    rp : int
    flood_risk : float

class Extractor(Singleton):
    """Class for extracting relevant climate data given an address or lat/long"""

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    def __init__(self) -> None:
        Singleton.__init__(self)

        self.geocoder = GoogleGeocoder(CONFIG.GEO_CODER_API_KEY)

    def coordinates_from_address(self, address: str) -> Coordinates:
        """Find the lat and lng of a given address"""
        result = self.geocoder.get(address)
        return Coordinates(lat=result[0].geometry.location.lat, lon=result[0].geometry.location.lng)


    def metadata_for_address(self, address: str) -> ClimateMetadata:
        """Calculate climate metadata for a given address"""
        coords = self.coordinates_from_address(address)
        try:
            water_level, shift, rp, flood_risk = fetch_climate_data(address)
        except IndexError:
            water_level = "0"
        return ClimateMetadata(

            water_level=water_level,
        )
