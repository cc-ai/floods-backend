"""
The `Config` class is used by the `Flask` application for accessing various
bits of application configuration
"""

import os

import yaml

from ccai.app.singleton import Singleton

# pylint: disable=R0903
class Config(Singleton):
    """Configuration object for the `Flask` application."""

    SECRET_KEY = os.environ.get("SECRET_KEY") or "secret-key"
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    API_KEYS_FILE = os.path.join(BASE_DIR, "../api_keys.yaml")

    API_KEYS_NAME = ["GEO_CODER_API_KEY", "STREET_VIEW_API_KEY"]

    # These variables are populated in __init__
    GEO_CODER_API_KEY = ""
    STREET_VIEW_API_KEY = ""

    def __init__(self):
        Singleton.__init__(self)
        if os.path.exists(self.API_KEYS_FILE):
            with open(self.API_KEYS_FILE, "r") as f:
                keys = yaml.load(f, Loader=yaml.FullLoader)

                for key, value in keys.items():
                    setattr(self, key, value)
        else:
            for key in self.API_KEYS_NAME:
                value = os.environ.get(key, None)

                if value is None:
                    raise ValueError("No API key found for {}".format(key))

                setattr(self, key, value)


# Redefine the "Config" symbol to be an instantiation of the above singleton.
# This redefinition of the symbol allows API users who import this module to use
# `Config.SECRET_KEY` instead of `Config().SECRET_KEY` to access attributes.
Config = Config()
