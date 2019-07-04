FROM gcr.io/climatechangeai/floods-backend-base:5dd5850231496b1494a23c54894c08f948b7d78a

# Copy the application code
COPY . /floods-backend
WORKDIR /floods-backend
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
