FROM docker.io/pytorch/pytorch@sha256:a77983eb7a3042ccf41a94da547a7f54a47fb70da15c36d67a8e8a329ebaa098

WORKDIR /validator
ENV PYTHONPATH=/validator

RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-client git \
    && rm -rf /var/lib/apt/lists/*

COPY ml_platform/workers/validator/requirements.txt .

RUN python -m pip install --break-system-packages --no-cache-dir -r requirements.txt

COPY ml_platform/workers/validator ./ml_platform/workers/validator
COPY ml_platform/shared ./ml_platform/shared
COPY ml_platform/workers/trainer/training_config.py ./ml_platform/workers/trainer/training_config.py
COPY ml_platform/workers/trainer/plugin_api ./ml_platform/workers/trainer/plugin_api

RUN python -m pip install --break-system-packages --no-cache-dir ./ml_platform/workers/validator/builtin_val

ENV ML_PLATFORM_RUNTIME=worker
ENTRYPOINT ["python", "-m", "ml_platform.workers.validator.main"]
