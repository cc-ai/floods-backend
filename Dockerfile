FROM gcr.io/climatechangeai/floods-backend-base:b58d1f23c842b73077061a3b02b8cc7d58ab662

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
