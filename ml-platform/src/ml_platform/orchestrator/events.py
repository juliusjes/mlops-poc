import json
import logging
from datetime import UTC, datetime
from enum import Enum
from typing import Any,  Literal,  Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from ml_platform.orchestrator.states import State, WorkflowState
from ml_platform.shared.services.postgres_service import PostgresService
from ml_platform.workers.trainer.training_config import TrainingConfig
from ml_platform.workers.validator.validation_config import ValidationConfig

logger = logging.getLogger(__name__)

class EventType(Enum):
    MANUAL_START = "manual_start"
    DATA_UPDATE = "data_update"
    DRIFT_START = "drift_start"
    DEPLOY = "deploy"
    TEST = "test"
    VALIDATE = "validate"


class ManualStartParams(BaseModel):
    training_config: TrainingConfig
    validation_config: ValidationConfig | None
    deploy_criteria: Literal["validated", "none", "no_deployment"] = "none"


class DataUpdateParams(BaseModel):
    new_data_endpoint: str
    dataset_location: str


class DriftStartParams(BaseModel):
    training_config: TrainingConfig
    validation_config: ValidationConfig

class DeployParams(BaseModel):
    model_uri: str
    validation_config: ValidationConfig
    deploy_criteria: Literal["validated", "none"] = "none"


class TestParams(BaseModel):
    log_message: str = "test workflow"


class ValidateParams(BaseModel):
    validation_config: ValidationConfig

EventParams = Union[
    ManualStartParams,
    DataUpdateParams,
    DriftStartParams,
    DeployParams,
    TestParams,
    ValidateParams
]

class EventStatus(Enum):
    PENDING = "pending"
    STARTED = "started"
    COMPLETE = "complete"
    FAILED = "failed"


class Event(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    event_type: EventType
    params: EventParams
    status: EventStatus = EventStatus.PENDING
    created: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started: datetime | None = None
    completed: datetime | None = None
    error: str | None = None
    result: dict | int | float| list | None | TrainingConfig = Field(default_factory=dict)
    

def check_event_success(event: Event, success) -> bool:
    """
    Chechk success and assign EventStatus.FAILED if success is false
    """
    if not success:
        event.status = EventStatus.FAILED
        return False
    return True


class EventHandler:
    def __init__(self, db_service: PostgresService):
        self.ps = db_service
        self._init_table()

    def _init_table(self):
        """
        Create event table if it does not exist
        """
        sql = """
            CREATE TABLE IF NOT EXISTS events (
                id UUID PRIMARY KEY,
                event_type TEXT NOT NULL,
                params JSONB,
                status TEXT NOT NULL,
                created TIMESTAMPTZ NOT NULL DEFAULT now(),
                started TIMESTAMPTZ,
                completed TIMESTAMPTZ,
                error TEXT,
                result JSONB
            );
            """
        self.ps.execute(sql)

    def create_event(self, event: Event):
        """
        insert a new event into the events table and add create date
        """
        result = event.model_dump(mode="json")

        for field in ("params", "result"):
            result[field] = json.dumps(result[field])

        self.ps.insert("events", result)

    def start_event(self, event: Event) -> None:
        """Set the status of an event to started and add the start date."""
        logger.info("%s start", event.id)
        event.status = EventStatus.STARTED
        sql = """
            UPDATE events
            SET status = %s,
                started = %s
            WHERE id = %s
        """
        self.ps.execute(
            sql,
            (event.status.value, datetime.now(UTC), event.id),
        )


    def complete_event(self, event: Event) -> None:
        """
        Set the status of an event to complete and add the completion date.
        """
        #logger.info("%s complete", event.id)

        event.status = EventStatus.COMPLETE

        sql = """
            UPDATE events
            SET status = %s,
                completed = %s
            WHERE id = %s
        """

        self.ps.execute(
            sql,
            (event.status.value, datetime.now(UTC), event.id),
        )

        logger.info("completed event %s", event.id)


    def set_event_error(self, event: Event, error: str) -> None:
        """
        Set the status of an event to failed and store the error.
        """
        logger.info("Event %s failed", event.id)

        event.status = EventStatus.FAILED

        sql = """
            UPDATE events
            SET status = %s,
                completed = %s,
                error = %s
            WHERE id = %s
        """

        self.ps.execute(
            sql,
            (
                event.status.value,
                datetime.now(UTC),
                error,
                event.id,
            ),
        )


    def set_event_result(self, event: Event, result: Any) -> None:
        """
        Set the result field of the event as JSON.
        """
        event.result = json.dumps(result)

        sql = """
            UPDATE events
            SET result = %s
            WHERE id = %s
        """

        self.ps.execute(
            sql,
            (event.result, event.id),
        )


    def current_event(self):
        """
        Get the current event regardless of its status.
        """
        sql = """
            SELECT *
            FROM events
            ORDER BY created DESC
            LIMIT 1
        """

        # TODO: convert row to Event
        return self.ps.fetch_one(sql)


    def n_last_events(self, n: int):
        """
        Get the n most recently created events.
        """
        sql = """
            SELECT *
            FROM events
            ORDER BY created DESC
            LIMIT %s
        """

        # TODO: convert rows to Event
        return self.ps.fetch_all(sql, (n,))


    def next_pending_event(self):
        """
        Get the next pending event.
        """
        sql = """
            SELECT *
            FROM events
            WHERE status = %s
            ORDER BY created ASC
            LIMIT 1
        """

        # TODO: convert row to Event
        return self.ps.fetch_one(
            sql,
            (EventStatus.PENDING.value,),
        )


    def get_events_by_status(self, status, limit, offset):
        """
        Get pending events, oldest first.
        """
        sql = """
            SELECT *
            FROM events
            WHERE status = %s
            ORDER BY created ASC
            LIMIT %s
            OFFSET %s
        """

        events = self.ps.fetch_all(
            sql,
            (status, limit, offset),
        )
        return events


    def get_event_by_id(self, id: UUID):
        sql = """
            SELECT *
            FROM events
            WHERE id = %s
            LIMIT 1
        """
        return self.ps.fetch_one(
            sql,
            (id,)
        )


class WorkflowStateHandler:
    def __init__(self, db_service: PostgresService):
        self.ps = db_service
        self._init_table()

    def _init_table(self):
        sql = """
        CREATE TABLE IF NOT EXISTS workflow_states (
            id BIGSERIAL PRIMARY KEY,
            event_id UUID NOT NULL,
            workflow TEXT NOT NULL,
            old_state TEXT,
            new_state TEXT NOT NULL,
            changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            result JSONB DEFAULT NULL
        );
        """
        self.ps.execute(sql)

    def save_workflow_state(self, state: WorkflowState) -> None:
        sql = """
            INSERT INTO workflow_states (
                event_id,
                workflow,
                old_state,
                new_state,
                changed_at
            )
            VALUES (%s, %s, %s, %s, %s)
        """

        self.ps.execute(
            sql,
            (
                state.event_id,
                state.workflow,
                state.old_state.value if state.old_state else None,
                state.new_state.value,
                state.changed_at,
            ),
        )

    def get_states_by_id(self, id: UUID):
        sql = """
            SELECT event_id, workflow, old_state, new_state, changed_at
            FROM workflow_states
            WHERE event_id = %s
            """
        return self.ps.fetch_all(
                sql,
                (id, )
            )

    def get_result(self, workflow, state) -> dict:
        sql = """
            SELECT result
            FROM workflow_states
            WHERE event_id = %s AND new_state = %s AND workflow = %s;
        """
        id = workflow.event_id
        workflow_ = workflow.workflow_name
        try:
            result = self.ps.fetch_one(sql, (id, state, workflow_))["result"]
        except KeyError:
            raise RuntimeError(f"{workflow} has no result available")
        return result

