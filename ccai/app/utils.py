"""
:mod:`ccai.app.utils` Utility functions
=======================================
"""
import json

from ccai.config import Config
from flask import Response


def allowed_file(filename):
    """Check if filename has an allowed extension.

    Parameters
    ----------
    filename: str
        String representing the filename of the file.

    Returns
    -------
    bool
        If the filename name is allowed or not.

    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def get_gridfs_metadata():
    """Return a the correct metadata for `GridFS`.

    Returns
    -------
    dict
        A dictionary containing metadata, like the mimetype.

    """
    return {'mimetype': 'image/base64'}


def make_response(response_code, message, data, mimetype='application/json'):
    """Return a `Flask.Response` object with given attributes.

    Parameters
    ----------
    response_code : int
        Request response code.
    message : str
        Request message.
    data : dict
        Request dictionary of data to send.
    mimetype : str, optional
        The mimetype of this request.

    Returns
    -------
    `Flask.Response`
        A ready-to-send response.

    """
    if not data:
        data = json.dumps({
            "message": message,
            "data": {}
        })

    return Response(data, response_code, mimetype=mimetype)
