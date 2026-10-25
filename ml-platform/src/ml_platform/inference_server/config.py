
import os
from enum import Enum
from typing import Literal

import dotenv
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

from ml_platform.shared.paths import ENV_FILE


class EngineConfig(BaseModel):
    model_uri: str
    device: Literal["cpu", "cuda"]
    preprocessing: list[str] = Field(default_factory=list)
    postprocessing: list[str] = Field(default_factory=list)


class EngineState(Enum):
    CREATED = "created"
    LOADING = "loading"
    PREDICTING = "predicting"
    AVAILABLE = "available"

class CoordinatorState(Enum):
    CREATED = "created"
    DEPLOYING = "deploying"
    AVAILABLE = "available"

class DeploymentState(Enum):
    DEPLOYING = "deploying"
    READY = "ready"
    FAILED = "failed"


class CollectorConfig(BaseModel):
    max_observations: int = Field(default=10_000, ge=1)


class PluginConfig(BaseModel):
    plugin_type: Literal["eval", "drift"]
    name: str
    parameters: dict
    trace_fields: list[str]
    notifiers: list[str] = Field(default_factory=list)
    reference_window_update: int = Field(default=10_000)
    reset_on_detection: bool = Field(default=False)



class Settings(BaseSettings):
    dotenv.load_dotenv(ENV_FILE)
        
    lakefs_username: str = os.environ["CREDENTIALS_ACCESS_KEY_ID"]
    lakefs_password: str = os.environ["CREDENTIALS_SECRET_ACCESS_KEY"]
    lakefs_endpoint: str = os.environ["SERVER_ENDPOINT_URL"]

    mlflow_tracking_uri: str = os.environ["MLFLOW_TRACKING_URI"]

    evidently_url: str = os.environ["EVIDENTLY_ENDPOINT_URL"]
    
    reference_data_source: str | None = None