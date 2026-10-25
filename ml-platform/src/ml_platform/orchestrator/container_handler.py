import asyncio
import logging
import os

import podman

logger = logging.getLogger(__name__)

DEFAULT_WORKER_TIMEOUT_SECONDS = 60 * 60

class ContainerHandler:
    def __init__(self):
        self.podman_client = self.create_podman_client()
        
        self.containers: set[str] = set()
        self.failures: dict[str, RuntimeError] = {}
        self._monitor_task = None


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


    async def wait_until_exit(
        self,
        container,
        timeout_seconds: int = DEFAULT_WORKER_TIMEOUT_SECONDS,
    ):
        attempts = 0

        try:
            async with asyncio.timeout(timeout_seconds):
                while not self.has_exited(container):
                    failure = self.failures.get(container.id)
                    if failure is not None:
                        raise failure

                    attempts += 1

                    if attempts == 1 or attempts % 3 == 0:
                        logger.info( f"Waiting for {container.id} to exit ({attempts})")

                    await asyncio.sleep(5)
        except TimeoutError:
            failure = RuntimeError(
                f"Container {container.name} timed out after {timeout_seconds} seconds"
            )
            logger.error(str(failure))
            self.containers.discard(container.id)

            try:
                container.reload()
                if container.attrs["State"]["Status"] == "running":
                    container.stop(timeout=10)
            except Exception:
                logger.exception("Failed to stop timed-out container %s", container.id)

            await self.handle_failure(container, -1, str(failure))
            self.failures[container.id] = failure
            raise failure

        failure = self.failures.pop(container.id, None)
        if failure is not None:
            raise failure


    def has_exited(self, container):
        """ Check if a container has exited. """
        return container.id not in self.containers


    def append_container(self, container):
        """Start monitoring a container."""
        self.containers.add(container.id)


    async def monitor(self):
        """Monitor all registered containers."""
        while True:
            for container_id in list(self.containers):
                try:
                    container = self.podman_client.containers.get(container_id)
                    container.reload()

                    state = container.attrs["State"]
                    status = state["Status"]

                    if status == "running":
                        continue

                    if status == "exited":
                        exit_code = state.get("ExitCode", -1)

                        if exit_code != 0:
                            logs = container.logs(
                                stdout=True,
                                stderr=True,
                                stream=False,
                            )

                            if isinstance(logs, bytes):
                                logs = logs.decode("utf-8", errors="replace")
                            else:
                                logs = b"".join(logs).decode("utf-8", errors="replace")

                            self.failures[container_id] = await self.handle_failure(
                                container,
                                exit_code,
                                logs,
                            )
                        else:
                            await self.handle_success(container)

                        self.containers.discard(container_id)

                except podman.errors.NotFound:
                    self.containers.discard(container_id)

                except Exception as exc:
                    logger.error( f"Error monitoring container " f"{container_id}: {exc}" )
                    #raise RuntimeError(f"{container_id}: {exc!s}")

                # except RuntimeError:
                #     await self.handle_failure(container, 1, )

            await asyncio.sleep(2)

    async def handle_failure( self, container, exit_code: int, logs: str, ):
        logger.error( f"Container {container.name} failed with exit code {exit_code}" )

        logger.error("Container logs:")
        logger.error(logs)
        image_id = container.image.id
        try:
            # Remove the container first.
            container.remove(force=False)

            logger.info( f"Removed container {container.name}" )

            # Now that the container is gone, remove its image.
            self.podman_client.images.remove( image_id, force=True, )

            logger.info( f"Removed image {image_id}" )

        except Exception:
            logger.exception( f"Failed to clean up container {container.name}" )

        return RuntimeError(f"Container {container.name} failed with exit code {exit_code}")


    async def handle_success(self, container):
        logger.info( f"Container {container.name} completed successfully" )

        image_id = container.image.id

        try:
            # Remove the container first.
            container.remove(force=True)

            logger.info( f"Removed container {container.name}" )

            # Now that the container is gone, remove its image.
            self.podman_client.images.remove( image_id, force=True, )

            logger.info( f"Removed image {image_id}" )

        except Exception:
            logger.exception( f"Failed to clean up container {container.name}" )
