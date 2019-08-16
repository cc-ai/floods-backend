#!/usr/bin/env python

"""
A utility for downloading and extracting non-flooded and flooded images from
the API server
"""

# pylint: disable=C0301

import base64

import requests


def main():
    """Application entrypoint"""
    response = requests.get(
        "http://localhost:5000/flood/munit/406%20Rue%20Principale%2C%20Cowansville%2C%20QC%2C%20Canada%0A"
    )

    with open("original.jpg", "wb") as file_handle:
        file_handle.write(base64.b64decode(bytes(response.json()["original"], "utf-8")))

    with open("flooded.jpg", "wb") as file_handle:
        file_handle.write(base64.b64decode(bytes(response.json()["flooded"], "utf-8")))


if __name__ == "__main__":
    main()
