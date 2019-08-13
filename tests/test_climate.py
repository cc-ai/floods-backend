"""
Unit tests for the climate data logic
"""

import unittest

from ccai.climate.extractor import Extractor


class TestClimate(unittest.TestCase):
    """Tests for the climate data logic"""

    def setUp(self) -> None:
        self.extractor = Extractor()
        self.mila_address = "6666 Saint Urbain Street, Montreal, QC, Canada"
        self.mila_coordinates = self.extractor.coordinates_from_address(self.mila_address)

    def test_lat_long(self) -> None:
        """Test extraction of lat long given an address"""
        self.assertEqual(int(self.mila_coordinates.lat), 45)
        self.assertEqual(int(self.mila_coordinates.lon), -73)

    def test_relative_change(self) -> None:
        """Test calculation of relative precipitation change"""
        self.assertEqual(self.extractor.relative_change(self.mila_coordinates), 0.13491)

    def test_monthly_average_precip(self) -> None:
        """Test calculation of monthly average precipitation"""
        self.assertEqual(self.extractor.monthly_average_precip(self.mila_coordinates), 9.5253)



if __name__ == "__main__":
    unittest.main()
