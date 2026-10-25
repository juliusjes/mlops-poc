from enum import Enum
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from ml_platform.inference_server.config import (
    CollectorConfig,
    EngineConfig,
    PluginConfig,
)


class InferenceServiceState(Enum):
    CREATED = "created"
    LOADING = "loading"
    PREDICTING = "predicting"
    AVAILABLE = "available"
    UPDATING = "updating"
    RETRAINING = "retraining"


class StateResponse(BaseModel):
    state: InferenceServiceState | str


class PredictRequest(BaseModel):
    deployment_uuid: UUID
    X: str  # utf-8 encoded blobs
    y: Optional[Any] = None


class PredictResponse(BaseModel):
    y_pred: list


class PredictEmbResponse(BaseModel):
    y_pred: list
    emb: list


class DeployRequest(BaseModel):
    engine_config: EngineConfig
    collector_config: CollectorConfig
    plugin_configs: list[PluginConfig] = Field(default_factory=list)


class DeployResponse(BaseModel):
    uuid: UUID
    success: bool
    

class RemoveDeploymentRequest(BaseModel):
    uuid: UUID

class RemoveDeploymentResponse(BaseModel):
    uuid: UUID
    remaining: dict[UUID, str]


class AvailableModelsResponse(BaseModel):
    available: dict[UUID, str]


class AddPluginRequest(BaseModel):
    uuid: UUID
    configs: list[PluginConfig]


class AddPluginResponse(BaseModel):
    uuid: UUID
    success: list[bool]


class RemovePluginRequest(BaseModel):
    uuid: UUID
    method: Literal["all", "byname", "regex"]
    name: str

class RemovePluginResponse(BaseModel):
    uuid: UUID
    n_removed: int
    remaining: list[str]


class SyncPluginsRequest(BaseModel):
    uuid: UUID

class SyncPluginsResponse(BaseModel):
    uuid: UUID


class ListPluginsRequest(BaseModel):
    uuid: UUID

class ListPluginsResponse(BaseModel):
    uuid: UUID
    plugins: list[str]


class ResetPluginLogRequest(BaseModel):
    uuid: UUID

class ResetPluginLogResponse(BaseModel):
    uuid: UUID
    log_file: str
    new_log_file: str

class ResetPluginsRequest(BaseModel):
    uuid: UUID

class ResetPluginsResponse(BaseModel):
    uuid: UUID


class FlushRequest(BaseModel):
    location: str


class FlushResponse(BaseModel):
    location: str
    n_samples: int


class RemoveDDRequest(BaseModel):
    method: Literal["all", "byname"]
    name: Optional[str] = None  # regex

    @model_validator(mode="after")
    def validate_names(self):
        if self.method == "byname" and self.name is None:
            raise ValueError(
                "Provide 'name' if method is 'byname'"
            )
        return self


class RemoveDDResponse(BaseModel):
    removed: list[str]
    remaining: list[str]


class ResetDDRequest(BaseModel):
    create_new_log: bool = Field(default=True)


class ResetDDResponse(BaseModel):
    success: bool
    log_file: str


class AddDDRequest(BaseModel):
    detector_type: str
    kwargs: dict
    tracing: bool = Field(default=True)
    trace_fields: list[str] = Field(default_factory=list)
    in_dash: bool = Field(default=False)


class AddDDResponse(BaseModel):
    success: bool
    log_file: str
