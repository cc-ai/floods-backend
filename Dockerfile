FROM gcr.io/climatechangeai/floods-backend-base:640d5bf4f165ae0e41174aea145cf530dddf5036

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
