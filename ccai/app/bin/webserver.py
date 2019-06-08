#!/usr/bin/env python

"""
A web service for indexing and retrieving documents
"""

import logging
import os
import tempfile

from flask import Flask, Response, jsonify, send_file
import prometheus_client

from ccai.app.streetview import fetch_street_view_image

DEBUG = os.environ.get("DEBUG", False)

app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)


@app.route("/address/<string:version>/<string:address>", methods=["GET"])
def address2photo(version: str, address: str) -> Response:
    """Endpoint which converts an address into a photo of the future"""
    images = fetch_street_view_image(address)
    with tempfile.TemporaryDirectory() as temp_dir:
        print(temp_dir)
        images.download_links(temp_dir)
        files = [f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))]
        print(files)
        if "gsv_0.jpg" in files:
            return send_file(os.path.join(temp_dir, "gsv_0.jpg"), as_attachment=True)
    return jsonify({"version": version, "address": address})


@app.route("/metrics", methods=["GET"])
def metrics() -> Response:
    """Prometheus metrics endpoint"""
    return Response(
        prometheus_client.generate_latest(prometheus_client.REGISTRY), mimetype="text/plain"
    )


if __name__ == "__main__":
    app.run(debug=bool(DEBUG))
