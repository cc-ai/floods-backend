"""
:mod:`ccai.app.main.routes` Routing module
==========================================
"""
import os

from ccai.app.engine import fetch_street_view_images, find_location
from ccai.app.main import bp
from ccai.config import Config
from flask import abort, send_file


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
    str(version)
    location = find_location(address)
    image_dir, results = fetch_street_view_images(location)

    if results.metadata[0]['status'] != 'OK':
        abort(404)

    image_name = os.path.join(image_dir, Config.SV_PREFIX.format('0'))

    return send_file(image_name, mimetype='image/gif')
