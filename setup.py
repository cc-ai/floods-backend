#!/usr/bin/env python
"""Installation script for CCAI backend."""
from distutils.core import setup

setup_args = dict(
        name="CCAI-Backend",
        version="0.dev",
        description="Flask backend for the CCAI project.",
        license="Apache",
        author="F. Corneau-Tremblay",
        author_email="corneauf@mila.quebec",
        packages=["ccai",],
        install_requires=["PyYAML", "python-googlegeocoder", "google-streetview"],
        tests_require=["pytest"]
        )

setup(**setup_args)
