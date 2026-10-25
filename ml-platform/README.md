# ML Platform demonstration setup

This is an proof of concept MLOps system. Podman Compose runs PostgreSQL,
SeaweedFS/S3, MLflow, LakeFS, the orchestrator, and inference services.

## Setup

Requirements: Python 3.12, Rootless Podman 5 or newer, and a Podman Compose
provider. The current local setup supports `podman-compose`.

From the repository root:

```bash
export ML_PLATFORM_DATA_DIR="$PWD"
cd ml-platform
cp .env.example .env
pyenv exec python -m pip install -e ".[dev]"
```

Edit `.env` with the local service credentials and paths. The example file lists every required setting. If pyenv is not used, replace `pyenv exec python` with the
Python executable from the intended virtual environment.

The editable install is required for data-generation scripts, tests, and other
commands that import `ml_platform`.

Enable the rootless Podman API socket. The orchestrator uses it to create
trainer and validator containers:

```bash
systemctl --user enable --now podman.socket
test -S "$XDG_RUNTIME_DIR/podman/podman.sock" && echo "Podman socket ready"
```

After logging in to LakeFS for the first time, download its credentials file
and update the .env file accordingly.

## Start the stack

Run Compose with the `ml-platform` project name. Dynamic workers use its
default network, `ml-platform_default`:

```bash
cd ml-platform
podman compose -p ml-platform up -d
podman ps -a
```

The normal worker mode is CPU-only. Set `WORKER_GPU_ENABLED=true` in `.env`
only when rootless Podman can expose `nvidia.com/gpu=all`. This requires compatible Nvidia drivers.

Note that some container images can take over 10 minutes to build.

Check that the services are ready:

```bash
curl -f http://localhost:8014/health       # orchestrator
curl -f http://localhost:8012/state        # inference server
curl -f http://localhost:8013/health       # inference client
curl -f http://localhost:5000/health       # MLflow
curl -f http://localhost:8000/healthcheck  # LakeFS
```

and

```bash
podman ps
```

If a service fails to start, inspect its logs before recreating containers:

```bash
podman logs ml-platform_orchestrator_1
```

Changing the application Dockerfiles or requirement manifests requires rebuilding the affected Compose images with `podman compose -p ml-platform build`. Trainer and
validator images are built on demand by the orchestrator.

## Validation and reproducibility

Tests and live workflows write data to the local LakeFS `pytest` repository,
PostgreSQL, and MLflow. Run them only against this local demonstration stack.

Images and direct Python dependencies are pinned to the validated baseline.
Transitive Python and OS packages are not hash-locked, so dependency upgrades
should be treated as intentional changes and validated after rebuilding.
