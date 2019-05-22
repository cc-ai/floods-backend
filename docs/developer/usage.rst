.. contents:: Developer's Guide: Usage


*****
Usage
*****

In this document, we will outline the steps needed to start the backend on the localhost.

Preparation
===========
In this step, we will clone the repo and install the required dependencies as well as the project itself.

1. Clone the repo::

    git clone https://github.com/cc-ai/floods-backend -o origin

2. Move inside the repo::

    cd floods_backend

3. Install the dev. requirements::

    pip install -r dev-requirements.txt

4. Install the CC-AI Backend project::

    pip install -e .

5. Copy your API keys yaml file inside the project::

    cp [...]/api_keys.yaml ccai/api_keys.yaml

Run
===
Now that the basic dependencies are installed, you can run the backend like this:

1. Export the ``FLASK_APP`` variable::

    export FLASK_APP=ganify

2. Run the ``Flask`` app::

    flask run
