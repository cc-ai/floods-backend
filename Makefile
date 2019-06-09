CONTAINER_NAME = gcr.io/mila-ccai/floods-backend
CONTAINER_TAG = $(shell git rev-parse --verify HEAD)
PYTHONPATH := $(shell pwd):$(PYTHONPATH)

test:
	python -m unittest discover tests
	mypy ccai tests
	pylint ccai/** tests/**

format:
	black .

develop:
	FLASK_APP=ccai/bin/webserver.py DEBUG=1 FLASK_DEBUG=1 python -m flask run

serve:
	gunicorn -w $(shell sysctl -n hw.ncpu) -b 0.0.0.0:5000 ccai.bin.webserver:app

container:
	docker build \
		-f Dockerfile \
		-t $(CONTAINER_NAME):$(CONTAINER_TAG) \
		.

push-container:
	docker push $(CONTAINER_NAME):$(CONTAINER_TAG)

deploy: container push-container
	yq w -i k8s/deployment.yml spec.template.spec.containers[0].image $(CONTAINER_NAME):$(CONTAINER_TAG)
	gcloud container clusters get-credentials floods-backend-cluster --zone us-west1-a --project mila-ccai
	kubectl apply -f ./k8s/deployment.yml
