import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import podman

from ml_platform.orchestrator.states import TrainingState as TState
from ml_platform.orchestrator.workflows.workflow import Workflow, WorkflowResult
from ml_platform.shared.paths import PACKAGE_ROOT, SOURCE_ROOT

TRAINER_DOCKERFILE = PACKAGE_ROOT / "workers/trainer/Dockerfile"
CUSTOM_DOCKERFILE = PACKAGE_ROOT / "workers/trainer/custom.Dockerfile"
TRAINER_BUILD_DIR = SOURCE_ROOT

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class TrainingWorkflow(Workflow):
    def __init__(self, event_id, container_handler, state_listener = None):
        super().__init__(event_id, state_listener)

        self.workflow_name = "Training-workflow"
        self.container_handler = container_handler

        self.podman_client = self.create_podman_client()

        self._set_state(TState.CREATED)


    def create_podman_client(self) -> podman.PodmanClient:
        container_host = os.environ.get("CONTAINER_HOST")
        logger.info(f"CONTAINER_HOST={container_host}")
        # First try PodmanClient's normal/default configuration.
        
        try:
            client = podman.PodmanClient()
            if client.ping():
                return client
        except Exception:
            logger.debug("Could not connect using Podman default configuration", exc_info=True)
            client.close()

        # Fall back to the Podman socket specified by CONTAINER_HOST.
        if container_host:
            try:
                client = podman.PodmanClient(base_url=container_host)
                if client.ping():
                    return client
            except Exception:
                logger.debug(
                    "Could not connect using CONTAINER_HOST=%s",
                    container_host,
                    exc_info=True,
                )
                client.close()

        logger.error("Could not connect to Podman")
        raise RuntimeError("Could not connect to Podman")

    def _image_exists(self, image_name: str = "ml-platform_trainer"):
        return self.podman_client.images.exists(image_name)

    def _get_image(self, image_name: str = "ml-platform_trainer") -> tuple:
        try:
            image = self.podman_client.images.get(image_name)
            return image, True
        except Exception: # TODO: add podman exceptions
            return None, False
        

    def _build_container(
            self, dockerfile: Path=TRAINER_DOCKERFILE, image_name: str="ml-platform_trainer"
            ) -> tuple:
        """
        Build the training container
        """
        logger.info(f"Building podman container from {dockerfile} in build context: {TRAINER_BUILD_DIR}")
        try:
            image, logs = self.podman_client.images.build(
                path=TRAINER_BUILD_DIR,
                dockerfile=dockerfile,
                tag=f"{image_name}",
                rm=True,
                forcerm=True,
                )
            success = True
        except Exception as e:
            logger.error(f"Failed to build image from {dockerfile}: {str(e)}")
            raise e

        for line in logs:
            data = json.loads(line)
            logger.debug(data.get("stream", ""))

        return image, success


    def _start_container(self, image):
        """
        Start the training container. Using a subprocess enables the use of the --no-hosts
        argument that disables copying the hostname table from the host machine. 
        """
        container_name = f"{self.workflow_name}_{self.event_id}"
        worker_network = os.environ.get("WORKER_NETWORK", "ml-platform_default")

        
        cmd = [
            "podman",
            "create",
            f"--name={container_name}",
            "--label=managed-by=container-handler",
            "-e", f"POSTGRES_USER={os.environ['POSTGRES_USER']}",
            "-e", f"POSTGRES_PASSWORD={os.environ['POSTGRES_PASSWORD']}",
            "-e", f"POSTGRES_URL={os.environ['POSTGRES_URL']}",
            "-e", f"AWS_ACCESS_KEY_ID={os.environ['AWS_ACCESS_KEY_ID']}",
            "-e", f"AWS_SECRET_ACCESS_KEY={os.environ['AWS_SECRET_ACCESS_KEY']}",
            "-e", f"MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING={os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING']}",
            "-e", f"MLFLOW_S3_ENDPOINT_URL={os.environ['MLFLOW_S3_ENDPOINT_URL']}",
            "-e", f"MLFLOW_TRACKING_URI={os.environ['MLFLOW_TRACKING_URI']}",
            "-e", f"SERVER_ENDPOINT_URL={os.environ['SERVER_ENDPOINT_URL']}",
            "-e", f"CREDENTIALS_ACCESS_KEY_ID={os.environ['CREDENTIALS_ACCESS_KEY_ID']}",
            "-e", f"CREDENTIALS_SECRET_ACCESS_KEY={os.environ['CREDENTIALS_SECRET_ACCESS_KEY']}",
            f"--network={worker_network}",
            "--no-hosts",
            image.id,
            "--jobid",
            str(self.event_id),
        ]
        if os.environ.get("WORKER_GPU_ENABLED", "false").lower() in {
            "1",
            "true",
            "yes",
        }:
            cmd.insert(4, "--device=nvidia.com/gpu=all")


        #logger.debug(f"{" ".join(cmd)}")

        result = subprocess.run(cmd, check=True, capture_output=True, text=True, )

        if result.returncode != 0:
            raise RuntimeError(
                f"podman create failed ({result.returncode}):\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
        container_id = result.stdout.strip()

        # Get the actual podman-py Container object
        container = self.podman_client.containers.get(container_id)

        network = self.podman_client.networks.get(worker_network)

        network.connect(container)
        container.start()

        logger.info(f"Started container: {container.name} with id {container.id}")

        return container

    async def run(self) -> WorkflowResult:
        """
        Run the training workflow
        """
        self._set_state(TState.STARTING)

        user_provided_plugin = False

        if not self._image_exists():
            # build base image
            image, success = self._build_container()
        else:
            image, success = self._get_image()

        if not success:
            logger.error("Failed to build or find base trainer image")
            raise RuntimeError
        
        image, success = self._build_container(
            dockerfile=CUSTOM_DOCKERFILE,
            image_name=f"ml-platform_trainer:{self.event_id}"
            )

        container = self._start_container(image)

        self.podman_client.close()

        self._set_state(TState.TRAINING)

        self.container_handler.append_container(container)
        try:
            await self.container_handler.wait_until_exit(container)
        except RuntimeError as e:
            self._set_state(TState.FAILED)
            raise e

        self._set_state(TState.COMPLETE)



if __name__ == "__main__":
    import asyncio
    import uuid

    event_id = uuid.uuid4()

    t = TrainingWorkflow(event_id)

    result = asyncio.run(t.run())
