FROM nvidia/cuda:10.1-devel

# General container initialization
RUN apt-get update

# Install pyenv to manage the required version of Python
RUN apt-get install git -y
RUN git clone https://github.com/pyenv/pyenv.git /root/.pyenv
RUN cd /root/.pyenv && git checkout v1.2.11

# Set the environment variables that are required to use pyenv
ENV PYENV_ROOT=/root/.pyenv
ENV PATH=$PYENV_ROOT/bin:$PATH

# Install the required version of Python from source
RUN apt-get install \
    build-essential \
    curl \
    libbz2-dev \
    libffi-dev \
    libssl-dev \
    libsqlite3-dev \
    libreadline-dev \
    liblzma-dev \
    zlib1g-dev \
    -y
RUN pyenv install 3.7.3
RUN git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv

# Install Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs

# Setup the pyenv
RUN pyenv virtualenv 3.7.3 floods-backend-3.7.3
ENV PATH=/root/.pyenv/versions/3.7.3/envs/floods-backend-3.7.3/bin/:$PATH
RUN pip install --upgrade pip

# Copy the application code
COPY . /floods-backend
WORKDIR /floods-backend

# Configure and pull files via Git LFS
RUN git remote set-url origin https://github.com/cc-ai/floods-backend.git
RUN git lfs install
RUN git lfs pull

# Install application dependencies
RUN pip install -r requirements.txt

# Setup the execution environment
ENV WORKERS=4
EXPOSE 443
CMD gunicorn \
  --certfile /api-climatechangeai-org-tls/tls.crt \
  --keyfile /api-climatechangeai-org-tls/tls.key \
  --workers $WORKERS \
  --bind 0.0.0.0:443 \
  ccai.bin.webserver:app
