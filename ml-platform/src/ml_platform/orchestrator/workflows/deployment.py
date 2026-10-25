import logging

from ml_platform.orchestrator.states import DeploymentState as DState
from ml_platform.orchestrator.workflows.workflow import Workflow
from ml_platform.shared.utils import info

logger = logging.getLogger(__name__)

class DeploymentWorkflow(Workflow):
    def __init__(self, inference_service, **kwargs):
        super().__init__(**kwargs)
        self.workflow_name = "Deployment workflow"
        self.inference_service = inference_service
        self.state = None
        self._set_state(DState.CREATED)

    async def run(self, model_uri, data_update_location=None, mv=None) -> dict:
        self._set_state(DState.DEPLOYING)

        if mv is not None:
            model_name = mv.name if mv else ""
            model_version = mv.version if mv else None
            model_uri = f"models:/{model_name}/{model_version}"
        else:
            model_version = model_uri.split("/")[-1]
            model_name = model_uri.split("/")[-2]

        logger.info(f"Deploying model {model_name}, version {model_version}")

        success, uuid = await self.inference_service.deploy_model(model_uri)

        if success:
            self._set_state(DState.COMPLETE)
            logger.info("Model deployed successfully")
        else:
            logger.info("Model deployment failed")
            self._set_state(DState.FAILED)
        
        return {
            "deployed": self.state == DState.COMPLETE,
            "model_uri": model_uri,
            "uuid": uuid
        }