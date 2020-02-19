# Climate Change AI (CCAI) Backend [![CircleCI](https://circleci.com/gh/cc-ai/floods-backend.svg?style=svg)](https://circleci.com/gh/cc-ai/floods-backend)

The CCAI project is an interdisciplinary project aimed at creating images of accurate, vivid, and personalized outcomes of climate change. Our goal is to use cutting-edge machine learning techniques to produce images of how neighborhoods and houses will look like following the effects of global warming. By creating a more visceral understanding of the effects of climate change, we aim to strengthen public support for necessary actions and motivate people to make impactful decisions. As a prototype, we first focus on modeling flood consequences on homes.


For a more detailed motivation explanation, read through our [2 page executive summary](https://docs.google.com/document/d/1WQtugSBgMVB-i0RhgCg_qaP7WDj7aimWvpZytKTEqY4/edit).

This document has the following sections:

- [How Does This Thing Work?](#how-does-this-thing-work)
- [Getting Started](#getting-started)
- [API Endpoints](#api-endpoints)

There are also the following documents which may be useful depending on your objectives:

- If you're looking to build, run, extend, or test this codebase, you should check out the [Contributor Guide](./CONTRIBUTING.md).

## How Does This Thing Work?

This server is an API server written in [Python](https://python.org/) using the [Flask](http://flask.pocoo.org/) microframework. The server code is in [`ccai/`](./ccai/) and the tests are in [`tests/`](./tests/). For more information on running the server, see the [Contributor Guide](./CONTRIBUTING.md).

## Getting Started

For a thorough accounting of how to set up this repo and use the various developer tools that are setup, you can read the [Contributor Guide](./CONTRIBUTING.md). Alternatively, the following is a minimal set of commands that you can run to get up and running as quickly as possible:

Clone the repo:

```
mkdir -p ~/git
git clone https://github.com/cc-ai/floods-backend ~/git/floods-backend
cd ~/git/floods-backend
```

Install the large files (models) using [Git  LFS](https://git-lfs.github.com/):

```
git lfs init
git lfs pull
git lfs checkout
```

Install the Python dependencies into your current Python environment (should be at least Python 3.7):

```
pip install -r requirements.txt
```

Run a development server:

```
make develop
```

The API server will now be available at http://127.0.0.1:5000. See the [API Endpoints](#api-endpoints) section for available endpoints.

## API Endpoints

Once you're running the webserver locally, there are a few API endpoints that are available.

### Fetching An Image For An Address

To download an image of an address, you can use the `/address/{address}` endpoint. To download a picture of Mila, you could run the following `curl` command locally:

```
curl http://127.0.0.1:5000/address/6666%20St%20Urbain%20St%2C%20Montreal%2C%20QC%20H2S%203H1%2C%20Canada > mila.jpg
open mila.jpg
```

### Fetching A Flooded Image and Metadata For An Address

To download an unprocessed image as well as a flooded image and metadata, you can use the `/flood/{model}/{address}` endpoint. The model string must be a valid model that we have configured to flood images. At the time of this writing, the only supported model is `munit`. To download the content for Mila, you could run the following curl command locally:

```
curl localhost:5000/flood/MUNIT/6666%20St%20Urbain%20St%2C%20Montreal%2C%20QC%20H2S%203H1%2C%20Canada > images.json
```

This will return a response like:

```json
{
    "original": "...",
    "flooded": "...",
    "metadata": {
        "monthly_average_precipitation": {
            "title": "Monthly Average Precipitation in 2050",
            "value": 9.5253
        },
        "relative_change_precipitation": {
            "title": "Relative Change in Precipitation by 2050",
            "value": 0.13491
        }
    }
}
```

Note that the `"metadata"` key may return arbitrary climate-related metadata about the image. The `"original"` and `"flooded"` keys will contain base64 encoded images.

### Prometheus Metrics

There is a Prometheus compatible metrics endpoint at `/metrics`. A simple HTTP GET will return a plain text response with metrics in the standard Prometheus format:

```
curl http://127.0.0.1:5000/metrics
```
