from datetime import datetime
from typing import Any, Literal
from enum import Enum
from uuid import UUID

from ml_platform.orchestrator.events import (
    DataUpdateParams,
    DeployParams,
    EventStatus,
    EventType,
)
from pydantic import BaseModel, Field

from ml_platform.workers.trainer.training_config import TrainingConfig
from ml_platform.workers.validator.validation_config import ValidationConfig


class ManualStartRequest(BaseModel):
    training_config: TrainingConfig
    validation_config: ValidationConfig | None
    deploy_criteria: Literal["validated", "none", "no_deployment"] = "none"


class DriftStartRequest(BaseModel):
    training_config: TrainingConfig
    validation_config: ValidationConfig

class DataUpdateRequest(BaseModel):
    new_data_endpoint: str
    dataset_location: str

class DeployRequest(BaseModel):
    model_uri: str
    validation_config: ValidationConfig
    deploy_criteria: Literal["validated", "none"] = "none"


def build_deploy_params(request: DeployRequest) -> DeployParams:
    """Bind the requested model URI to the validation configuration."""
    validation_config = request.validation_config.model_copy(deep=True)
    validation_config.model_load_kwargs.model_uri = request.model_uri
    return DeployParams(
        model_uri=request.model_uri,
        validation_config=validation_config,
        deploy_criteria=request.deploy_criteria,
    )

class ValidateRequest(BaseModel):
    validation_config: ValidationConfig

class EventSort(str, Enum):
    OLDEST = "oldest"
    NEWEST = "newest"


class EventListRequest(BaseModel):
    status: EventStatus | None = None
    event_type: EventType | None = None
    workflow: str | None = None

    sort: EventSort = EventSort.NEWEST

    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class EventResponse(BaseModel):
    id: UUID
    event_type: EventType
    params: dict
    status: EventStatus
    created: datetime
    started: datetime | None = None
    completed: datetime | None = None
    error: str | None = None
    result: Any
    


class EventListResponse(BaseModel):
    events: list[EventResponse]
    limit: int
    offset: int
    has_more: bool


class EventHistoryResponse(BaseModel):
    workflow: str
    old_state: str | None
    new_state: str
    changed_at: datetime


class EventHistoryListResponse(BaseModel):
    event_id: UUID
    history: list[EventHistoryResponse]


class EventStatsResponse(BaseModel):
    pending: int
    started: int
    complete: int
    failed: int
    total: int


class HealthResponse(BaseModel):
    status: str
