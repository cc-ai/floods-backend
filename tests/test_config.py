"""
Unit tests for the application configuration
"""

import unittest

from ccai.config import CONFIG


class TestConfig(unittest.TestCase):
    """Tests for the application configuration"""

    def test_config(self) -> None:
        """Testing config initialization and access"""
        self.assertEqual(CONFIG.SECRET_KEY, "secret-key")


if __name__ == "__main__":
    unittest.main()
