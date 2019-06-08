CONTAINER_NAME = ccai/floods-backend
CONTAINER_TAG = $(shell git rev-parse --verify HEAD)
PYTHONPATH := $(shell pwd):$(PYTHONPATH)

test:
	python -m unittest discover ccai/tests

develop:
	FLASK_APP=ccai/app/bin/webserver.py DEBUG=1 FLASK_DEBUG=1 python -m flask run

format:
	black .

serve:
	gunicorn -w $(shell sysctl -n hw.ncpu) -b 0.0.0.0:5000 ccai.app.bin.webserver:app

container:
	docker build \
		-f Dockerfile \
		-t $(CONTAINER_NAME):${CONTAINER_TAG} \
		.
