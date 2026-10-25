import logging
from uuid import UUID, uuid4

from ml_platform.inference_server.config import (
    CollectorConfig,
    CoordinatorState,
    DeploymentState,
    EngineConfig,
    PluginConfig,
    Settings,
)
from ml_platform.inference_server.deployment import Deployment

logger = logging.getLogger(__name__)


class Coordinator:
    def __init__(self):
        self.state = None
        self._set_state(CoordinatorState.CREATED)

        self.deployment_registry: dict[UUID, Deployment] = {} # uuid: Deployment
        #self.deployment_queue = asyncio.Queue()

        self._set_state(CoordinatorState.AVAILABLE)


    def _set_state(self, state:CoordinatorState):
        if self.state is not state:
            logger.info(f"{self.state or 'None'} -> {state}")
            self.state = state


    async def add_deployment(self, engine_config: EngineConfig, colletor_config: CollectorConfig, plugin_configs: list[PluginConfig]):
        self._set_state(CoordinatorState.DEPLOYING)
        uuid = uuid4()

        deployment = Deployment(uuid, engine_config, colletor_config, plugin_configs)
        await deployment.start()

        success = deployment.state == DeploymentState.READY
        if success:
            self.deployment_registry[uuid] = deployment

        self._set_state(CoordinatorState.AVAILABLE)
        return uuid, success


    async def remove_deployment(self, uuid) -> bool:
        dep_to_remove = self.deployment_registry.get(uuid, None)
        if dep_to_remove is None:
            logger.warning(f"{uuid} does not correspond to any deployment")
            return False
        
        await dep_to_remove.stop()
        removed = self.deployment_registry.pop(uuid)
        logger.info(f"Removed {removed.get_model_name()} from deployment registry")
        return True


    def is_deployed(self, uuid):
        return uuid in self.deployment_registry


    def get_model_names(self):
        names = {}
        for uuid in self.deployment_registry:
            names[uuid] = self.deployment_registry[uuid].get_model_name()
        return names


# dir_path = os.path.dirname(os.path.realpath(__file__))
# settings_path = os.path.join(dir_path, "settings.json")

# if os.path.exists(settings_path):
#     logger.info("Using settings.json")
#     with open(settings_path, "r", encoding="utf-8") as f:
#         settings = Settings.model_validate_json(f.read())
# else:
#     settings = Settings()


coordinator = Coordinator()
