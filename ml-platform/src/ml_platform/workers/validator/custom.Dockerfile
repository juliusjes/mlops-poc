FROM ml-platform_validator

#RUN apt-get update && apt-get install -y iproute2 && rm -rf /var/lib/apt/lists/*
#RUN pip install --no-cache-dir ./ml_platform/workers/validator/example_user_package
RUN python -m pip install --break-system-packages --no-cache-dir psutil