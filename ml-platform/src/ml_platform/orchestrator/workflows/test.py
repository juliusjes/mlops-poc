from ml_platform.orchestrator.states import TestState
from ml_platform.orchestrator.workflows.workflow import Workflow
from ml_platform.shared.utils import info


class TestWorkFlow(Workflow):
    def __init__(self, event_id, state_listener=None):
        super().__init__(event_id, state_listener)
        self.workflow_name = "Test workflow"
        self._set_state(TestState.CREATED)

    async def run(self, message="hello from Test workflow"):
        info(f"{message}", tf=1)
        self._set_state(TestState.COMPLETE)
        return True