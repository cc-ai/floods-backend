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

    def test_metadata(self) -> None:
        """Test calculation of climate metadata"""
        meta = self.extractor.metadata_for_address(self.mila_address)
        self.assertEqual(meta.relative_change_precip, 0.13491)
        self.assertEqual(meta.monthly_average_precip, 9.5253)


if __name__ == "__main__":
    unittest.main()
