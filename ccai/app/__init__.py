"""Base module for the CCAI app.

This module serves as the initialization point for all the submodules
used as well as the different blueprints and loggers.
"""
import logging
from logging.handlers import RotatingFileHandler
import os

from ccai.config import Config
from flask import Flask


def create_app(config_class=Config):
    """Create a `Flask` application.

    Parameters
    ----------
    config_class: obj
        A configuration class containing the useful attributes for the `Flask` constructor.

    Returns
    -------
    app:
        The application instance

    """
    app = Flask(__name__)
    app.config.from_object(config_class)

    from app.errors import bp as errors_bp
    app.register_blueprint(errors_bp)

    from app.main import bp as main_bp
    app.register_blueprint(main_bp)

    if not app.debug and not app.testing:
        if not os.path.exists('logs'):
            os.mkdir('logs')

        file_handler = RotatingFileHandler('logs/floods.log', maxBytes=10240, backupCount=10)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s '
                                                    '[in %(pathname)s:%(lineno)d]'))

        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

        app.logger.setLevel(logging.INFO)
        app.logger.info('Floods started.')

    return app
