# Contributor Guide

This document has the following sections:

- [Setup](#setup)
- [Testing](#testing)
- [Running The Server](#running-the-server)
- [Building The Container](#building-the-container)

## Setup

First, clone the repo:

```
git clone git@github.com:cc-ai/floods-backend.git
```

From the root of the repository, run the following to install the application dependencies into your current Python environment:

```
pip install -r requirements.txt
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

# Building The Container

To build a container which can run the application server, use the following Make command:

```
make container
```

Under the hood, this will run a `docker build` command with the appropriate container tag, environment variables, etc. If the command is successful, a container name will be printed out at the end. You might see something like:

```
Step 19/19 : CMD gunicorn -w $WORKERS -b 0.0.0.0:80 ccai.bin.webserver:app
 ---> Using cache
 ---> 5aac2c0f7c37
Successfully built 5aac2c0f7c37
Successfully tagged gcr.io/mila-ccai/floods-backend:5ababa7899ff22e8be85e96c7ec03392b67b4d3a
```

You can then take this container name (the part after "Successfully tagged") and run it as such:

```
docker run \
  -e WORKERS=$(sysctl -n hw.ncpu) \
  -p 5000:80 \
  gcr.io/mila-ccai/floods-backend:5ababa7899ff22e8be85e96c7ec03392b67b4d3a
```

As you can see here, you are also able to map port 80 in the container to port 5000 locally as well as specify the number of worker processes to use via the `WORKERS` environment variable.
