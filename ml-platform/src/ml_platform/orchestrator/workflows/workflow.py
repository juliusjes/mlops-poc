from datetime import UTC, datetime
from typing import Callable
from uuid import UUID

from ml_platform.orchestrator.events import WorkflowState
from ml_platform.orchestrator.states import State


class WorkflowResult:
    success: bool
    result: dict
    kwargs_next: dict
    


class Workflow:
    def __init__( self, event_id: UUID, state_listener: Callable[[WorkflowState], None] | None = None, ):
        super().__init__()

        self.workflow_name = "base"
        self.event_id = event_id
        self.state: State | None = None
        self.state_listener = state_listener

    def _set_state(self, new_state: State) -> None:
        """
        Change the workflow state and notify the listener
        """

        if self.state == new_state:
            return

        old_state = self.state
        self.state = new_state

        workflow_state = WorkflowState(
            event_id=self.event_id,
            workflow=self.workflow_name,
            old_state=old_state,
            new_state=new_state,
            changed_at=datetime.now(UTC),
        )

        if self.state_listener:
            self.state_listener(workflow_state)



    async def run(self) -> WorkflowResult:
        """
        Execute defined workflow
        """
        raise NotImplementedError