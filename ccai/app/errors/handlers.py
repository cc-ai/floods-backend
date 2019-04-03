"""Module handling the different errors being raised during runtime."""
from ccai.app.errors import bp
from flask import jsonify, make_response


@bp.app_errorhandler(404)
def not_found_error(error):
    """Handle `not found` error."""
    return make_response(jsonify({'error': 'Not found'}), 404)


@bp.app_errorhandler(500)
def internal_error(error):
    """Handle internal error."""
    return make_response(jsonify({'error': 'Internal error'}), 500)
