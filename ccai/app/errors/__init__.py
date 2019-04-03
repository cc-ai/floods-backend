from flask import Blueprint

bp = Blueprint('errors', __name__)

from ccai.app.errors import handlers
