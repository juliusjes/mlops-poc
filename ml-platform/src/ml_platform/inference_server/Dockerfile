FROM docker.io/library/python@sha256:e8a38829209aea82774882e5157e1f7c3e8c8393ff53a697b3580896e57f345c

WORKDIR /inference_server
ENV PYTHONPATH=/inference_server 
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY ./src/ml_platform/inference_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY ./src/ml_platform/inference_server ./ml_platform/inference_server
COPY ./src/ml_platform/shared ./ml_platform/shared

RUN mkdir -p /outputs/logs

EXPOSE 8012

ENTRYPOINT [ "uvicorn", "ml_platform.inference_server.main:app",  "--host", "0.0.0.0", "--port", "8012", "--log-level", "info" ]
