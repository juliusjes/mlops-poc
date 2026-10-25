import logging
import os

import dotenv
import httpx

from ml_platform.shared.paths import ENV_FILE
from ml_platform.shared.utils import get_healthy_endpoint

logger = logging.getLogger(__name__)

class InferenceService:
    instance = None
    _initialized = False

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self):
        if self._initialized:
            return

        dotenv.load_dotenv(ENV_FILE)

        urls = [
            os.environ["INFERENCE_SERVER_URL"],
            os.environ["INFERENCE_SERVER_URL"].replace("inference-server", "0.0.0.0")
        ]

        self.inference_server_url = get_healthy_endpoint(
            urls, health_path="/state", timeout=10
            )

        logger.info(f"Using endpoint: {self.inference_server_url}")
        self._initialized = True

    async def deploy_model(self, model_uri: str) -> tuple[bool, str|None]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post( # TODO: change this to be dynamic
                    f"{self.inference_server_url}/deploy_model",
                    json={
                        "engine_config": {
                            "model_uri": model_uri,
                            "device": "cpu",
                            "preprocessing": ["numpy_to_tensor", "add_batch_dim"],
                            "postprocessing": []
                            },
                        "collector_config": {},
                        "plugin_configs": []
                    }
                )
            
            return response.status_code == httpx.codes.OK, response.json().get("uuid")

        except httpx.HTTPError as e:
            logger.warning(f"Failed to deploy model: {e}")
            return False, None

if __name__ == "__main__":
    import asyncio

    iser = InferenceService()

    asyncio.run(iser.deploy_model("models:/simple_resnet_exp2/5"))
    