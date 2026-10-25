FROM docker.io/library/python@sha256:e8a38829209aea82774882e5157e1f7c3e8c8393ff53a697b3580896e57f345c

WORKDIR /inference_client
ENV PYTHONPATH=/inference_client

COPY ./src/ml_platform/inference_client/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY ./src/ml_platform/inference_client ./ml_platform/inference_client

RUN mkdir -p /inference_client/data_processing
COPY ./src/ml_platform/shared/data_processing/data_processing.py ./ml_platform/shared/data_processing/data_processing.py
COPY ./src/ml_platform/shared/data_processing/augmentation_funcs.py ./ml_platform/shared/data_processing/augmentation_funcs.py

COPY ./src/ml_platform/shared/paths.py ./ml_platform/shared/paths.py
COPY ./src/ml_platform/shared/exceptions.py ./ml_platform/shared/exceptions.py
COPY ./src/ml_platform/shared/utils.py ./ml_platform/shared/utils.py
COPY ./src/ml_platform/shared/services/lakefs_service.py ./ml_platform/shared/services/lakefs_service.py 
# Optional: create directories expected by the app
RUN mkdir -p ./outputs/logs

EXPOSE 8013

ENTRYPOINT ["uvicorn", "ml_platform.inference_client.main:app", "--host", "0.0.0.0", "--port", "8013", "--log-level", "info"]
