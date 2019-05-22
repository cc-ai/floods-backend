.. contents:: Developer's Guide: Documenting

***********
Documenting
***********

We are using `Read The Docs`_ theme for `Sphinx`_.

Run tox command::

    tox -e docs

to build the *html* and *man* pages of the documentation.

You can also host the documentation on the *localhost* by running command::

    tox -e host-docs

which will serve the pages under ``docs/build/html``.

Use command::

    tox -e doc8

to ensure that the documentation follows the project's standards.

When adding a new module, make sure to update the corresponding skeleton under the
documentation outline for the project found at ``docs/src/code/``.

.. _Read The Docs: https://sphinx-rtd-theme.readthedocs.io/en/latest/
.. _Sphinx: http:/www.sphinx-doc.org/en/master
