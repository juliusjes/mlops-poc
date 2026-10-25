import sys
from types import SimpleNamespace
from uuid import uuid4

# Workflow imports currently load PostgresService even though these command
# construction tests do not create a database connection.  Keep the test
# isolated from the undeclared optional connection-pool dependency.
sys.modules.setdefault("psycopg_pool", SimpleNamespace(ConnectionPool=object))

from ml_platform.orchestrator.workflows.training import TrainingWorkflow


class FakeContainer:
    id = "container-id"
    name = "worker"

    def __init__(self):
        self.started = False

    def start(self):
        self.started = True


class FakePodmanClient:
    def __init__(self):
        self.container = FakeContainer()
        self.containers = SimpleNamespace(get=lambda _container_id: self.container)
        network = SimpleNamespace(connect=lambda _container: None)
        self.network_names = []
        self.container.network_names = self.network_names
        self.networks = SimpleNamespace(
            get=lambda network_name: self.network_names.append(network_name) or network
        )


def make_workflow():
    workflow = object.__new__(TrainingWorkflow)
    workflow.workflow_name = "Training-workflow"
    workflow.event_id = uuid4()
    workflow.podman_client = FakePodmanClient()
    return workflow


def start_worker(monkeypatch, gpu_enabled, worker_network="ml-platform_default"):
    workflow = make_workflow()
    command = []
    image = SimpleNamespace(id="image-id")

    for key in (
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING",
        "MLFLOW_S3_ENDPOINT_URL",
        "MLFLOW_TRACKING_URI",
        "SERVER_ENDPOINT_URL",
        "CREDENTIALS_ACCESS_KEY_ID",
        "CREDENTIALS_SECRET_ACCESS_KEY",
    ):
        monkeypatch.setenv(key, "test")

    monkeypatch.setenv("WORKER_GPU_ENABLED", gpu_enabled)
    monkeypatch.setenv("WORKER_NETWORK", worker_network)

    def fake_run(cmd, **_kwargs):
        command.extend(cmd)
        return SimpleNamespace(returncode=0, stdout="container-id\n", stderr="")

    monkeypatch.setattr(
        "ml_platform.orchestrator.workflows.training.subprocess.run", fake_run
    )
    workflow._start_container(image)
    return command, workflow.podman_client.container


def test_worker_is_cpu_first(monkeypatch):
    command, container = start_worker(monkeypatch, "false")

    assert "--device=nvidia.com/gpu=all" not in command
    assert container.started


def test_worker_gpu_is_explicitly_enabled(monkeypatch):
    command, container = start_worker(monkeypatch, "true")

    assert "--device=nvidia.com/gpu=all" in command
    assert container.started


def test_worker_uses_the_configured_compose_network(monkeypatch):
    command, _container = start_worker(monkeypatch, "false", "custom-worker-network")

    assert "--network=custom-worker-network" in command
    assert _container.network_names == ["custom-worker-network"]
