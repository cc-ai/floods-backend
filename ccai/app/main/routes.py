"""
:mod:`ccai.app.main.routes` Routing module
==========================================
"""

from ccai.app import mongo
from ccai.app.engine import fetch_street_view_images, find_location, save_to_database
from ccai.app.main import bp
import ccai.app.utils as utils
from flask import current_app, request


@bp.route('/address/<version>/<string:address>', methods=['GET'])
def ganify(version, address):
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
        return utils.make_response(200, "Error with StreetView", {})

    # Open a GridFS instance to save the image
    filename = save_to_database(location, results)

    # There is a problem with the implementation of `flask_pymongo`
    # See issue `https://github.com/dcrosta/flask-pymongo/issues/120`
    # We need to manually update the mimetype for the image
    response = mongo.send_file(filename)
    response.mimetype = 'image/base64'
    return response


@bp.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    """Handle requests to `upload_file`.

    This endpoint uploads a picture to the database.

    """
    if request.method == 'POST':
        if 'file' not in request.files:
            current_app.logger.error('Request sent with no `file` key.')
            return utils.make_response(500, "Error: no file", data={})

        file = request.files['file']

        if file.filename == '':
            current_app.logger.error('File sent with no filename')
            return utils.make_response(500, "Error: no filename", data={})

        if file and utils.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            fs = GridFS(mongo.db)
            metadata = utils.get_gridfs_metadata()
            fs.put(file.stream, filename=filename, metadata=metadata)
            current_app.logger.info("Image upload: {}".format(filename))

    return utils.make_response(200, "Success", data={})
