APP_CONTAINER_NAME = gcr.io/climatechangeai/floods-backend
BASE_CONTAINER_NAME = gcr.io/climatechangeai/floods-backend-base
CONTAINER_TAG = $(shell git rev-parse --verify HEAD)
CONTAINER_TAG = $(shell git rev-parse --verify HEAD)
PYTHONPATH := $(shell pwd):$(PYTHONPATH)

################################################################################
# Developer Tools
################################################################################

test:
	python -m unittest discover tests
	mypy ccai tests
	pylint ccai/** tests/**

format:
	black .

################################################################################
# Launching the server
################################################################################

develop:
	FLASK_APP=ccai/bin/webserver.py DEBUG=1 FLASK_DEBUG=1 python -m flask run

serve:
	gunicorn -w $(shell sysctl -n hw.ncpu) -b 0.0.0.0:5000 ccai.bin.webserver:app

################################################################################
# Utilities for building containers and deploying the API
################################################################################

container:
	docker build \
		-f Dockerfile \
		-t $(APP_CONTAINER_NAME):$(CONTAINER_TAG) \
		.

base-container:
	docker build \
		-f base.Dockerfile \
		-t $(BASE_CONTAINER_NAME):$(CONTAINER_TAG) \
		.

gcloud-auth:
	gcloud auth configure-docker
	gcloud container clusters get-credentials floods-backend --zone us-east1-b --project climatechangeai

push-container: gcloud-auth
	docker push $(APP_CONTAINER_NAME):$(CONTAINER_TAG)

push-base-container: gcloud-auth
	docker push $(BASE_CONTAINER_NAME):$(CONTAINER_TAG)

stage-deploy:
	yq w -i k8s/deployment.yml spec.template.spec.containers[0].image $(APP_CONTAINER_NAME):$(CONTAINER_TAG)

deploy: gcloud-auth stage-deploy
	kubectl apply -f ./k8s/deployment.yml
