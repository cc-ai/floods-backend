"""
:mod:`ccai.app.main.file_upload` Module dedicated to upload files
=================================================================
"""

import tempfile
import zipfile

from ccai.app.database import save_image
from flask import current_app
from werkzeug.utils import secure_filename


def upload_image(file):
    """Save a simple image to database.

    Parameters
    ----------
    file : `werkzeug.datastructures.FileStorage`
        A `FileStorage` object obtained from `request.files``

    """
    filename = secure_filename(file.filename)
    save_image(file.stream, filename=filename)
    current_app.logger.info("Image upload: {}".format(filename))


def upload_zip(file):
    """Save the image inside a zip file to database.

    Parameters
    ----------
    file : `werkzeug.datastructures.FileStorage`
        A `FileStorage` object obtained from `request.files``

    """
    with tempfile.TemporaryDirectory() as tempdir:
        secured = secure_filename(file.filename)
        filename = tempdir + "/" + secured
        file.save(filename)

        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(tempdir)

            for name in zip_ref.namelist():
                save_image(open(tempdir + "/" + name, 'rb'), filename=name)
