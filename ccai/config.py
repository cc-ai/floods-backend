"""
Configuration for the `Flask` application.

This module hosts the `Config` class, which is used by the `Flask` application
to initialized certain modules.
"""
import os

import yaml


class Config(object):
    """Configuration object for the `Flask` application.

    The `Config` class contains several attributes use by the `Flask` application
    to initialize modules. It also contains certain value specific to the application
    use.

    Attributes
    ----------
    SECRET_KEY: str
        Secret key used by the app. Normally defined as an environment variable.
    DOWNLOAD_DIR: str
        The download directory for the StreeView images.
    BASE_DIR: str
        The current base directory for the CCAI module.
    ADMINS: list
        List of emails for the admins.
    API_KEYS_FILE: str
        File containing the different keys for the Google APIs.
    SV_PREFIX: str,
        Prefix of the images returned by the StreetView API.

    """

    SECRET_KEY = os.environ.get('SECRET_KEY') or 'secret-key'
    DOWNLOAD_DIR = '.data/'
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    ADMINS = ['corneauf@mila.quebec']
    API_KEYS_FILE = os.path.join(BASE_DIR, 'api_keys.yaml')
    SV_PREFIX = 'gsv_{}.jpg'


with open(Config.API_KEYS_FILE, 'r') as f:
    keys = yaml.load(f)

    for key, value in keys.items():
        setattr(Config, key, value)
