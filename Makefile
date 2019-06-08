export PYTHONPATH := $(shell pwd):$(PYTHONPATH)

test:
	python -m unittest discover ccai/test

develop:
	FLASK_APP=ccai/app/bin/webserver.py FLASK_DEBUG=1 python -m flask run

format:
	black .

serve:
	gunicorn -w $(shell sysctl -n hw.ncpu) -b 0.0.0.0:5000 ccai.app.bin.webserver:app
