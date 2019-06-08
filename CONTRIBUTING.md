# Contributor Guide

This document has the following sections:

- [Setup](#setup)
- [Testing](#testing)
- [Running the Server](#running-the-server)

## Setup

First, clone the repo:

```
git clone git@github.com:cc-ai/floods-backend.git
```

From the root of the repository, run the following to install the application dependencies into your current Python environment:

```
pip install -r ccai/requirements.txt
```

## Testing

From the root of the repository, run the following to run all of the tests:

```
make test
```

## Running The Server

For development, the best way to start the server is probably the following, which has hot reloading (which works almost most of the time):

```
make develop
```

For a more production-ready server, run the following, which will use [`gunicorn`](https://gunicorn.org/) to launch n worker processes running the webserver such that n is the number of local CPU cores:

```
make serve
```
