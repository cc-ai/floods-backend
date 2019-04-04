.. contents:: Developer's Guide: Contribution Standards

**********
Contribute
**********

Coding and Repositery Standards
===============================

We are using flake8_ (along with some of its plugins) and pylint_.
Their styles are provided in ``tox.ini`` under the ``[flake8]`` section
and in ``.pylintrc`` respectively.

You can run both of these tools using the following commands:

.. code-block:: sh

    tox -e flake8
    tox -e pylint

You can find the list of flake8 plugins under the ``dev-requirements.txt`` file.

.. _flake8: http://flake8.pycqa.org/en/latest
.. _pylint: https://www.pylint.org/

Project Structure
=================

The project follows a simple Flask_ structure. At the top level ``floods.py``
creates the ``app``. Under the ``ccai/`` directory, you will find a ``config.py``
file for the configuration process of ``Flask``. Then, under ``ccai/app`` you will
find directories for the followings:

- Errors handling: handlers are found inside ``ccai/app/errors/``.
- Routing: routing is handle inside ``ccai/app/main/``.
- REST API: the api functionality is inside ``ccai/app/engine/``.

.. _Flask: http://flask.pocoo.org/

Fork and Pull Request
=====================

Fork the CCAI backend project remotely to your Github_ account now and start
by submitting a `Pull Request <https://github.com/DatCorno/floods-backend/pulls>`_ to us
or by discussing an `issue <https://github.com/DatCorno/floods-backend/issues>`_ with us.

.. _Github: https://github.com
