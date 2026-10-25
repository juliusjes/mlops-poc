import asyncio
import logging
import traceback

#from ml.training_config import ModelLoadKwargs, ModelSourceType, TrainingConfig
# services
#from ml.validation_config import ValidationConfig
from ml_platform.orchestrator.container_handler import ContainerHandler
from ml_platform.orchestrator.deployment_policy import should_deploy

# events and states
from ml_platform.orchestrator.events import (
    DataUpdateParams,
    DeployParams,
    DriftStartParams,
    Event,
    EventHandler,
    #    EventState,
    EventStatus,
    EventType,
    ManualStartParams,
    TestParams,
    WorkflowStateHandler,
)
from ml_platform.orchestrator.states import State
from ml_platform.orchestrator.workflows.data_update import DataUpdateWorkflow

# workflows
from ml_platform.orchestrator.workflows.deployment import DeploymentWorkflow
from ml_platform.orchestrator.workflows.test import TestWorkFlow
from ml_platform.orchestrator.workflows.training import TrainingWorkflow
from ml_platform.orchestrator.workflows.validation import ValidationWorkflow
from ml_platform.shared.exceptions import WorkflowException
from ml_platform.shared.services.inference_service import InferenceService

from ml_platform.shared.services.postgres_service import PostgresService

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self):
        self.events = asyncio.Queue()

        # initialize singleton services

        logger.info(f"Starting services...")
        self.inference_service = InferenceService()
        self.postgres_service = PostgresService(db_name="orchestrator_db")
        self.event_handler = EventHandler(self.postgres_service)
        self.workflow_handler = WorkflowStateHandler(self.postgres_service)

        self.container_handler = ContainerHandler()

        logger.info("Orchestartor started")


    async def add_event(self, event: Event):
        self.event_handler.create_event(event)
        await self.events.put(event)

    async def run(self):
        while True:
            event = await self.events.get()

            try:
                self.event_handler.start_event(event)   
                await self._process_event(event)                

            except Exception as e:
                self.event_handler.set_event_error(event, str(e))
                tb = traceback.format_exc()
                logger.error(f"ERROR WHILE PROCESSING EVENT: {event}\n{tb}")

            else:
                self.event_handler.complete_event(event)

            finally:
                self.events.task_done()

    
    async def _process_event(self, event: Event):
        """
        Execute workflows related to an action

        event contains:
            event.event_type: EventType
            event.params: Any
        """
        id = event.id

        match event.event_type:

            case EventType.MANUAL_START:

                workflow = TrainingWorkflow(
                    id,
                    self.container_handler,
                    state_listener=self.workflow_handler.save_workflow_state
                    )
                await workflow.run()

                result_tr = self.workflow_handler.get_result(workflow, "training")
                if len(result_tr) < 1:
                    raise RuntimeError("Training result had less than 1 element")
                
                workflow = ValidationWorkflow(
                    id,
                    self.container_handler,
                    state_listener=self.workflow_handler.save_workflow_state
                    )
                await workflow.run()
                
                result_vl = self.workflow_handler.get_result(workflow, "validating")
                if len(result_vl) < 1:
                    raise RuntimeError("Validation result had less than 1 element")

                model_uri = result_tr["model_uri"]
                deploy_condition = should_deploy(
                    event.params.deploy_criteria,
                    result_vl,
                )

                if deploy_condition:
                    workflow = DeploymentWorkflow(
                        self.inference_service,
                        event_id=id,
                        state_listener=self.workflow_handler.save_workflow_state
                        )

                    result_dp = await workflow.run(model_uri=model_uri)
                else: result_dp = {"deployed": False, "model_uri": model_uri}

                self.event_handler.set_event_result( event, { **result_tr, **result_vl, **result_dp } )
                

            case EventType.DATA_UPDATE:
                raise RuntimeError("Not implemented")
                # Intentionally unsupported in this demonstration. The route
                # records the request but performs no data migration.
                pass
                # location = event.params.dataset_location
                # endpoint = event.params.new_data_endpoint

                # workflow = DataUpdateWorkflow(self.lakefs_service)
                # success = await workflow.run(location, new_data_endpoint=endpoint)
                # if not success: return


            case EventType.DRIFT_START:
                raise RuntimeError("Not implemented")
                # Intentionally unsupported in this demonstration. Drift
                # events are recorded but do not start retraining.
                # tcfg = event.params.training_config
                # vcfg = event.params.validation_config

                # workflow = TrainingWorkflow(tcfg, self.lakefs_service, self.mlflow_service, state_listener=self.workflow_handler.save_workflow_state)
                # uri = await workflow.run()

                # # TODO: check event success

                # workflow = ValidationWorkflow(vcfg, self.mlflow_service, self.lakefs_service, state_listener=self.workflow_handler.save_workflow_state)
                # success, mv = await workflow.run(model_uri=uri)
                
                # if not check_event_success(event, success):
                #     return # TODO

                # workflow = DeploymentWorkflow(self.inference_service, state_listener=self.workflow_handler.save_workflow_state)
                # success, model_uri = await workflow.run(uri, tcfg.dataset_location, mv=mv)
                
                # if not check_event_success(event, success):
                #     return # TODO
                
                # self.event_handler.set_event_result(event, model_uri)
                pass

            case EventType.DEPLOY:
                workflow = ValidationWorkflow(
                    id,
                    self.container_handler,
                    state_listener=self.workflow_handler.save_workflow_state
                    )
                await workflow.run()
                result_vl = self.workflow_handler.get_result(workflow, "validating")
                if len(result_vl) < 1:
                    raise RuntimeError("Validation result had less than 1 element")

                model_uri = event.params.model_uri
                deploy_condition = should_deploy(
                    event.params.deploy_criteria,
                    result_vl,
                )

                if deploy_condition:
                    workflow = DeploymentWorkflow(
                        self.inference_service,
                        event_id=id,
                        state_listener=self.workflow_handler.save_workflow_state
                        )

                    result_dp = await workflow.run(model_uri=model_uri)
                else: result_dp = {"deployed": False, "model_uri": model_uri}

                self.event_handler.set_event_result( event, {**result_vl, **result_dp})


            case EventType.VALIDATE:
                workflow = ValidationWorkflow(
                    id,
                    self.container_handler,
                    state_listener=self.workflow_handler.save_workflow_state
                    )
                await workflow.run()
                result_vl = self.workflow_handler.get_result(workflow, "validating")

                self.event_handler.set_event_result(event, result_vl)


            case EventType.TEST:
                message = event.params.log_message
                workflow = TestWorkFlow(id, state_listener=self.workflow_handler.save_workflow_state)
                success = await workflow.run(message=message)
                if not success:
                    raise WorkflowException()

            case _:
                raise Exception(f"Invalid EventType: {event.event_type}")



                
orch = Orchestrator()
#monitor = orch.wm
