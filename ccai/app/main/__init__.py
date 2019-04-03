"""Main module."""
from flask import Blueprint

bp = Blueprint('main', __name__)

from ccai.app.main import routes  # noqa
