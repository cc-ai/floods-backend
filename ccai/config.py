"""
The `Config` class is used by the `Flask` application for accessing various
bits of application configuration
"""

import os, yaml
from ccai.singleton import Singleton
from ccai.nn.model.trainer import MUNIT_Trainer as MODEL
from api import GEO_CODER, STREET_VIEW

################################## MODEL Hyperparameters #####################################
FLOOD_MODEL = MODEL  #
ROUTE_MODEL = "model"  #
FLOOD_MODE = "all"  #
FLOOD_LEVEL = 0.4  #
RP = 50  #
##############################################################################################


class ConfigSingleton(Singleton):
    """Configuration object for the `Flask` application."""

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    ################################### CONFIG Hyparameters ###################################
    MODEL_CONFIG_FILE = os.path.join(BASE_DIR, "nn/configs/config.yaml")  #
    MODEL_CHECKPOINT_FILE = os.path.join(BASE_DIR, "nn/configs/gen_00060000.pt")  #
    MODEL_WEIGHT_FILE = os.path.join(BASE_DIR, "nn/configs/resnet_34_8s_cityscapes_best.pth")  #
    CLIMATE_DATA = "data/floodMapGL_rp50y.tif"  #
    ###########################################################################################

    SECRET_KEY = os.environ.get("SECRET_KEY") or "secret-key"
    API_KEYS_FILE = os.path.join(BASE_DIR, "../api_keys.yaml")
    API_KEYS_NAME = ["GEO_CODER_API_KEY", "STREET_VIEW_API_KEY"]

    # These variables are populated in __init__
    GEO_CODER_API_KEY = ""
    STREET_VIEW_API_KEY = ""

    def __init__(self) -> None:
        Singleton.__init__(self)
        if os.path.exists(self.API_KEYS_FILE):
            with open(self.API_KEYS_FILE, "r") as api_keys_file:
                keys = yaml.load(api_keys_file, Loader=yaml.FullLoader)  # type: ignore

                for key, value in keys.items():
                    setattr(self, key, value)
        else:
            for key in self.API_KEYS_NAME:
                value = os.environ.get(key, None)

                if value is None:
                    value = ""

                setattr(self, key, value)

        with open(self.MODEL_CONFIG_FILE, "r") as model_config_file:
            self.model_config = yaml.load(model_config_file, Loader=yaml.FullLoader)  # type: ignore


CONFIG = ConfigSingleton()
