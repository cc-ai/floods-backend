"""
:mod:`ccai.app.main.routes` Routing module
==========================================
"""
from ccai.app import mongo
from ccai.app.engine import fetch_street_view_images, find_location, save_to_database
from ccai.app.main import bp
from flask import abort
from gridfs import GridFS


@bp.route('/address/<version>/<string:address>', methods=['GET'])
def flood(version, address):
    """Handle requests to `/address/` webpage.

    This function is called when a request of the form found in the route
    decorator is made. It will then take the version and address given
    and find images of the location given and run it through the appropriate
    network.

    Parameters
    ----------
    version: str
        Which version of the network to use.
    address: str
        The actual address to find the images of.

    """
    location = find_location(address)
    results = fetch_street_view_images(location)

    if results.metadata[0]['status'] != 'OK':
        abort(404)

    # Open a GridFS instance to save the image
    fs = GridFS(mongo.db)
    filename = save_to_database(location, results, fs)

    # There is a problem with the implementation of `flask_pymongo`
    # See issue `https://github.com/dcrosta/flask-pymongo/issues/120`
    # We need to manually update the mimetype for the image
    response = mongo.send_file(filename)
    response.mimetype = 'image/base64'
    return response
