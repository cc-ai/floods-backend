from app.engine import fetch_street_view_images, find_location
from app.main import bp
from config import Config
from datetime import datetime
from flask import abort, send_file

import os

@bp.route('/address/<version>/<string:address>', methods=['GET'])
def flood(version, address):
    location = find_location(address)
    image_dir, results = fetch_street_view_images(location)

    if results.metadata[0]['status'] != 'OK':
        abort(404)

    image_name = os.path.join(image_dir, Config.SV_PREFIX.format('0'))

    return send_file(image_name, mimetype='image/gif')
