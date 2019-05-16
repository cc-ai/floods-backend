"""
:mod:`ccai.app.utils` Utility functions
=======================================
"""
from ccai.config import Config


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
