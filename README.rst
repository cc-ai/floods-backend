====================
CC-AI Ganify Backend
====================

Run the server
--------------

To run the server, simply follow these steps:

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

6. Export the ``FLASK_APP`` variable::

    export FLASK_APP=ganify

7. Run the ``Flask`` app::

    flask run
