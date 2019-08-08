FROM gcr.io/climatechangeai/floods-backend-base:650480431b748564c073009054be618f92e61052

# Copy the application code
COPY . /floods-backend
WORKDIR /floods-backend
RUN git remote set-url origin https://github.com/cc-ai/floods-backend.git
RUN git lfs install
RUN git lfs pull
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
