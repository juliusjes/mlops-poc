from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import UUID


class State(Enum):
    """
    Base class for workflow states
    """
    pass


@dataclass
class WorkflowState:
    event_id: UUID
    workflow: str
    old_state: State | None
    new_state: State
    changed_at: datetime


class TrainingState(State):
    CREATED = "created"
    STARTING = "starting"
    TRAINING = "training"
    COMPLETE = "complete"
    FAILED = "failed"


class DeploymentState(State):
    CREATED = "created"
    DEPLOYING = "deploying"
    COMPLETE = "complete"
    FAILED = "failed"
    

class DataUpdateState(State):
    CREATED = "created"
    REQUESTING = "requesting"
    PUSHING = "pushing"
    COMPLETE = "complete"
    FAILED = "failed"


class ValidationState(State):
    CREATED = "created"
    STARTING = "starting"
    VALIDATING = "validating"
    COMPLETE = "complete"
    FAILED = "failed"

class TestState(State):
    CREATED = "created"
    COMPLETE = "complete"