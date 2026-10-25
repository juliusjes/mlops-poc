import asyncio
from contextlib import suppress

import pytest

from ml_platform.orchestrator.container_handler import ContainerHandler


class FakeImage:
    id = "image-id"


class FakeContainer:
    id = "container-id"
    name = "worker"
    image = FakeImage()

    def __init__(self, status="exited", exit_code=1):
        self.attrs = {"State": {"Status": status, "ExitCode": exit_code}}
        self.stopped = False
        self.removed = False

    def reload(self):
        return None

    def logs(self, **_kwargs):
        return b"worker failed"

    def stop(self, **_kwargs):
        self.stopped = True
        self.attrs["State"]["Status"] = "exited"

    def remove(self, **_kwargs):
        self.removed = True


class FakeContainers:
    def __init__(self, container):
        self.container = container

    def get(self, _container_id):
        return self.container


class FakeImages:
    def __init__(self):
        self.removed = []

    def remove(self, image_id, **_kwargs):
        self.removed.append(image_id)


class FakePodmanClient:
    def __init__(self, container):
        self.containers = FakeContainers(container)
        self.images = FakeImages()


def make_handler(container):
    handler = object.__new__(ContainerHandler)
    handler.podman_client = FakePodmanClient(container)
    handler.containers = {container.id}
    handler.failures = {}
    handler._monitor_task = None
    return handler


def test_failed_container_releases_waiter_with_error():
    async def scenario():
        container = FakeContainer()
        handler = make_handler(container)
        monitor = asyncio.create_task(handler.monitor())

        for _ in range(10):
            if container.id not in handler.containers:
                break
            await asyncio.sleep(0.01)

        with pytest.raises(RuntimeError, match="failed with exit code 1"):
            await handler.wait_until_exit(container)

        monitor.cancel()
        with suppress(asyncio.CancelledError):
            await monitor

        assert container.removed
        assert handler.podman_client.images.removed == [container.image.id]

    asyncio.run(scenario())


def test_timed_out_container_is_stopped_and_reported_as_failure():
    async def scenario():
        container = FakeContainer(status="running", exit_code=None)
        handler = make_handler(container)

        with pytest.raises(RuntimeError, match="timed out after"):
            await handler.wait_until_exit(container, timeout_seconds=0.01)

        assert container.stopped
        assert container.removed
        assert container.id not in handler.containers
        assert handler.podman_client.images.removed == [container.image.id]

    asyncio.run(scenario())
