import asyncio
from uuid import uuid4

from ml_platform.inference_server.coordinator import Coordinator
from ml_platform.inference_server.deployment import Deployment


def test_stop_cancels_observation_worker():
    async def worker():
        await asyncio.Event().wait()

    async def scenario():
        deployment = object.__new__(Deployment)
        deployment.worker_task = asyncio.create_task(worker())
        await asyncio.sleep(0)

        await deployment.stop()

        assert deployment.worker_task is None

    asyncio.run(scenario())


def test_remove_deployment_stops_worker_before_unregistering():
    class FakeDeployment:
        def __init__(self):
            self.stopped = False

        async def stop(self):
            self.stopped = True

        def get_model_name(self):
            return "test-model"

    async def scenario():
        coordinator = Coordinator()
        deployment_id = uuid4()
        deployment = FakeDeployment()
        coordinator.deployment_registry[deployment_id] = deployment

        assert await coordinator.remove_deployment(deployment_id)
        assert deployment.stopped
        assert deployment_id not in coordinator.deployment_registry

    asyncio.run(scenario())
