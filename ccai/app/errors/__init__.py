"""
:mod:`ccai.app.errors` -- Errors handling blueprint
===================================================
"""

from flask import Blueprint

bp = Blueprint('errors', __name__)

from ccai.app.errors import handlers  # noqa
