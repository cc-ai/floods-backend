# Climate Change AI (CCAI) Backend

The CCAI project is an interdisciplinary project aimed at creating images of accurate, vivid, and personalized outcomes of climate change. Our goal is to use cutting-edge machine learning techniques to produce images of how neighborhoods and houses will look like following the effects of global warming. By creating a more visceral understanding of the effects of climate change, we aim to strengthen public support for necessary actions and motivate people to make impactful decisions. As a prototype, we first focus on modeling flood consequences on homes.


For a more detailed motivation explanation, read through our [2 page executive summary](https://docs.google.com/document/d/1WQtugSBgMVB-i0RhgCg_qaP7WDj7aimWvpZytKTEqY4/edit).

This document has the following sections:

- [How Does This Thing Work?](#how-does-this-thing-work)
- [Getting Started](#getting-started)
- [API Endpoints](#api-endpoints)

There are also the following documents which may be useful depending on your objectives:

- If you're looking to build, run, extend, or test this codebase, you should check out the [Contributor Guide](./CONTRIBUTING.md).

## How Does This Thing Work?

This server is an API server written in [Python](https://python.org/) using the [Flask](http://flask.pocoo.org/) microframework. The server code is in [`ccai/app/`](./ccai/app/) and the tests are in [`ccai/test/`](./ccai/test/). For more information on running the server, see the [Contributor Guide](./CONTRIBUTING.md).

## Getting Started

For a thorough accounting of how to set up this repo and use the various developer tools that are setup, you can read the [Contributor Guide](./CONTRIBUTING.md). Alternatively, the following is a minimal set of commands that you can run to get up and running as quickly as possible:

Clone the repo:

```
mkdir -p ~/git
git clone https://github.com/cc-ai/floods-backend ~/git/floods-backend
cd ~/git/floods-backend
```

Install the Python dependencies into your current Python environment (should be at least Python 3.7):

```
pip install -r ccai/requirements.txt
```

Run a development server:

```
make develop
```

The API server will now be available at http://127.0.0.1:5000. See the [API Endpoints](#api-endpoints) section for available endpoints.

## API Endpoints

Once you're running the webserver locally, there are a few API endpoints that are available.

### Fetching An Image For An Address

TODO

### Prometheus Metrics

There is a Prometheus compatible metrics endpoint at `/metrics`. A simple HTTP GET will return a plain text response with metrics in the standard Prometheus format:

```
curl localhost:5000/metrics
```
