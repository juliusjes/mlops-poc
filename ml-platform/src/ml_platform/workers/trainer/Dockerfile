FROM docker.io/pytorch/pytorch@sha256:a77983eb7a3042ccf41a94da547a7f54a47fb70da15c36d67a8e8a329ebaa098

WORKDIR /trainer
ENV PYTHONPATH=/trainer 

RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-client git \
    && rm -rf /var/lib/apt/lists/*

COPY ./ml_platform/workers/trainer/requirements.txt ./requirements.txt

RUN which python && python --version && python -c "import torch; print(torch.__version__)"

RUN python -m pip install --break-system-packages \
    --no-cache-dir -r ./requirements.txt

COPY ./ml_platform/workers/trainer ./ml_platform/workers/trainer
COPY ./ml_platform/shared ./ml_platform/shared

RUN python -m pip install --break-system-packages --no-cache-dir ./ml_platform/workers/trainer/builtin

ENV ML_PLATFORM_RUNTIME=worker
ENTRYPOINT ["python", "-m", "ml_platform.workers.trainer.main"]
