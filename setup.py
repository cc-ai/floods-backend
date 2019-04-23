#!/usr/bin/env python
"""Installation script for CCAI backend."""
from setuptools import find_packages, setup

setup(name="ccai-backend",
      version="0.dev",
      packages=find_packages(),
      install_requires=["flask", "PyYAML", "python-googlegeocoder", "google-streetview"],
      tests_require=["pytest"],
      setup_requires=["setuptools"],
      author="F. Corneau-Tremblay",
      author_email="corneauf@mila.quebec",
      license="Apache",
      include_package_data=True,
      package_data={"ccai": ["api_keys.yaml"],
                    "ccai/tests": ["*.yaml"]}
      )
