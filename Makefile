test:
	python -m unittest discover ccai/test

develop:
	FLASK_APP=ccai/app/bin/webserver.py FLASK_DEBUG=1 python -m flask run