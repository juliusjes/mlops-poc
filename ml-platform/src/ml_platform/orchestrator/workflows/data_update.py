from ml_platform.orchestrator.workflows.workflow import Workflow
from ml_platform.orchestrator.states import DataUpdateState as DUState

# Not used
class DataUpdateWorkflow(Workflow):
    def __init__(self, lakefs_service):
        super().__init__()
        self.workflow_name = "Data update workflow"
        self.lakefs_service = lakefs_service
        self.state = None
        self._set_state(DUState.CREATED)


    async def run(self, dataset_location, new_data_endpoint=None):
        dataset_location = await self.lakefs_service.build_new_training_snapshot(dataset_location, new_data_endpoint=new_data_endpoint)
        self._set_state(DUState.COMPLETE)
        return dataset_location