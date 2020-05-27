#!/usr/bin/env python

"""
A web service for indexing and retrieving documents
"""

import os, tempfile, torch
from flask import Flask, Response, jsonify, send_file
from flask_cors import CORS

from ccai.image_processing.process_image import create_temp_dir, fetch_image, encode_image, decode_image
from ccai.climate.process_climate import fetch_climate_data
from ccai.nn.process_model import cuda_check, model_validation, model_launch
from ccai.config import FLOOD_MODEL, ROUTE_MODEL
from ccai.config import CONFIG
from ccai.nn.model.segmentation import Resnet34_8s

# MODELS Initialisation
VALID_MODELS = ["model"]
MODEL_NEW_SIZE = CONFIG.model_config["new_size"]
MODEL = FLOOD_MODEL(CONFIG.model_config)
MASK_MODEL = Resnet34_8s(num_classes=19)
MASK_MODEL.load_state_dict(torch.load(CONFIG.MODEL_WEIGHT_FILE))
MASK_MODEL.cuda()
MASK_MODEL.eval()


# API server initialization
DEBUG = os.environ.get("DEBUG", False)
app = Flask(__name__)
CORS(app)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True

# Check if CUDA is available
cuda_check(MODEL)

@app.route("/flood/<string:model>/<string:address>", methods=["GET"])
def flood(model: str, address: str) -> Response:
    """Endpoint which converts an address into a photo of the flooded future"""

    model_validation(model, VALID_MODELS)
    water_level, shift, rp, flood_risk, history, address = fetch_climate_data(address)
    images = fetch_image(address)

    with tempfile.TemporaryDirectory() as temp_dir:
        create_temp_dir(images, temp_dir)
        path_to_gsv_image, gsv_image_response = encode_image(temp_dir)
        path_to_flooded_image = model_launch(MODEL, MODEL_NEW_SIZE, MASK_MODEL, temp_dir, path_to_gsv_image)
        flooded_image_response  = decode_image(temp_dir, path_to_flooded_image)

    response = {
        "original": gsv_image_response,
        "flooded": flooded_image_response,
        "metadata": {
            "water_level": {
                 "title": " Expected Water Level (in CM):",
                 "value": water_level,
            },
            "rp": {
                "title": " Return Period (in years):",
                "value": rp,
            },
            "flood_risk": {
                "title": " Flood Risk (in %):",
                "value": flood_risk,
            },
            "shift": {
                "title": " New Frequency (in %):",
                "value": shift,
            },
            "history": {
                "title": "Historical Data:",
                "value": history,
            },
                    },
                }

    return jsonify(response)

if __name__ == "__main__":
    app.run(host= '0.0.0.0', port=5000)

