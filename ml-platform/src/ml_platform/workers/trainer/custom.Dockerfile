FROM ml-platform_trainer

RUN apt-get update && apt-get install -y iproute2 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --break-system-packages --no-cache-dir ./ml_platform/workers/trainer/example_user_package
