#!/usr/bin/env python

"""
A web service for indexing and retrieving documents
"""

# pylint: disable=R0914

import logging
import os
import tempfile

from flask import Flask, Response, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import prometheus_client
import torch
import torchvision.utils as vutils
from torchvision import transforms

from ccai.config import CONFIG
from ccai.nn.munit.trainer import MUNIT_Trainer
from ccai.streetview import fetch_street_view_image

# Global environment-based configuration
DEBUG = os.environ.get("DEBUG", False)

# Versions of models that are supported by the API endpoint
VALID_VERSIONS = ["munit"]

# API server initialization
app = Flask(__name__)  # pylint: disable=C0103
CORS(app)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)


# MUNIT Hyperparameters and Model
NEW_SIZE = CONFIG.munit_config["new_size"]
MUNIT_MODEL = MUNIT_Trainer(CONFIG.munit_config)
if torch.cuda.is_available():
    MUNIT_STATE_DICT = torch.load(CONFIG.MUNIT_CHECKPOINT_FILE)
else:
    MUNIT_STATE_DICT = torch.load(CONFIG.MUNIT_CHECKPOINT_FILE, map_location={"cuda:0": "cpu"})
MUNIT_MODEL.gen.load_state_dict(MUNIT_STATE_DICT["2"])
if torch.cuda.is_available():
    MUNIT_MODEL.cuda()  # type: ignore
MUNIT_MODEL.eval()


@app.route("/address/<string:version>/<string:address>", methods=["GET"])
def address2photo(version: str, address: str) -> Response:
    """Endpoint which converts an address into a photo of the future"""
    if version.lower() not in VALID_VERSIONS:
        response = jsonify({"error": "Invalid model version", "valid_versions": VALID_VERSIONS})
        response.status_code = 400
        return response

    try:
        images = fetch_street_view_image(
            address, CONFIG.GEO_CODER_API_KEY, CONFIG.STREET_VIEW_API_KEY
        )
    except Exception as exception:  # pylint: disable=W0703
        response = jsonify(
            {
                "error": "An error occurred fetching the Google Street View image",
                "exception_text": str(exception),
            }
        )
        response.status_code = 500
        return response

    with tempfile.TemporaryDirectory() as temp_dir:
        images.download_links(temp_dir)
        files = [f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))]
        if "gsv_0.jpg" not in files:
            response = jsonify({"error": "Image not found in response from Google Street View"})
            response.status_code = 500
            return response
        with torch.no_grad():
            transform = transforms.Compose(
                [
                    transforms.Resize(NEW_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

            # path to streetview image
            path_xa = os.path.join(temp_dir, "gsv_0.jpg")

            image_transformed = transform(Image.open(path_xa).convert("RGB")).unsqueeze(0)
            if torch.cuda.is_available():
                image_transformed = image_transformed.cuda()
            x_a = torch.Tensor(image_transformed)
            c_xa_b, _ = MUNIT_MODEL.gen.encode(x_a, 1)

            # Initiate parameters
            content = c_xa_b
            style_data = torch.mul(torch.randn(1, 16, 1, 1), 0.5)
            if torch.cuda.is_available():
                style_data = style_data.cuda()
            style = torch.Tensor(style_data)

            outputs = MUNIT_MODEL.gen.decode(content, style, 2)
            outputs = (outputs + 1) / 2.0

            output_path = os.path.join(temp_dir, "output" + "{:03d}.jpg".format(0))
            vutils.save_image(outputs.data, output_path, padding=0, normalize=True)
            return send_file(output_path, as_attachment=True)

    # in the happy path, this should be unreachable
    response = jsonify({"error": "Internal Server Error"})
    response.status_code = 500
    return response


@app.route("/metrics", methods=["GET"])
def metrics() -> Response:
    """Prometheus metrics endpoint"""
    return Response(
        prometheus_client.generate_latest(prometheus_client.REGISTRY), mimetype="text/plain"
    )


if __name__ == "__main__":
    app.run(debug=bool(DEBUG))
