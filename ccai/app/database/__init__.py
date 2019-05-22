"""
:mod:`ccai.app.database` -- MongoDB interface calls
===================================================
"""
from ccai.app import mongo
from ccai.app.utils import get_gridfs_metadata
from gridfs import GridFS


def save_image(stream, **kwargs):
    """Take a stream and save its content to GridFS.

    Parameters
    ----------
    stream :
        The input stream for the uploaded file.
    kwargs** : dict
        Further arguments for GridFS.

    """
    if 'metadata' not in kwargs:
        kwargs['metadata'] = get_gridfs_metadata()

    fs = GridFS(mongo.db)
    fs.put(stream, **kwargs)
