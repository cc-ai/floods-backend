import os
import yaml


class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'secret-key'
    DOWNLOAD_DIR = '.data/'
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    ADMINS = ['corneauf@mila.quebec']
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(BASE_DIR, 'models.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    API_KEYS_FILE = 'api_keys.yaml'
    SV_PREFIX = 'gsv_{}.jpg'

with open(Config.API_KEYS_FILE, 'r') as f:
    keys = yaml.load(f)

    for key, value in keys.items():
        setattr(Config, key, value)
