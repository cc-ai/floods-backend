#!/usr/bin/env python

"""
A web service for indexing and retrieving documents
"""

import logging
import os

from flask import Flask, Response, jsonify
import prometheus_client

DEBUG = os.environ.get("DEBUG", False)

app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)


@app.route("/address/<string:version>/<string:address>", methods=["GET"])
def address2photo(version: str, address: str) -> Response:
    """Endpoint which converts an address into a photo of the future"""
    return jsonify({"version": version, "address": address})


@app.route("/metrics", methods=["GET"])
def metrics() -> Response:
    """Prometheus metrics endpoint"""
    return Response(
        prometheus_client.generate_latest(prometheus_client.REGISTRY), mimetype="text/plain"
    )


if __name__ == "__main__":
    app.run(debug=bool(DEBUG))
