FROM docker.io/library/python@sha256:2055081c860db8c842b663415b795b026e7e5cc63b59c5193fcc9997713ea5b8
RUN apk add --no-cache iproute2 bind-tools netcat-openbsd
RUN apk add --no-cache postgresql-client podman
WORKDIR /orchestrator

COPY ./src/ml_platform/orchestrator/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./src/ml_platform/orchestrator ./ml_platform/orchestrator
# COPY ./src/ml_platform/shared/data_processing/data_processing.py ./ml_platform/shared/data_processing/data_processing.py
# COPY ./src/ml_platform/shared/*.py ./ml_platform/shared/*.py
COPY ./src/ml_platform/shared ./ml_platform/shared
COPY ./src/ml_platform/workers ./ml_platform/workers
#COPY ./src/ml_platform/workers/validator/validation_config.py ./ml_platform/workers/validator/validation_config.py

#WORKDIR /orchestrator/ml_platform/orchestrator
ENTRYPOINT ["uvicorn", "ml_platform.orchestrator.main:app", "--host", "0.0.0.0", "--port", "8014", "--log-level", "info"]
