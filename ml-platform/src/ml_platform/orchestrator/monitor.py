# from ml_platform.orchestrator.events import EventState, WorkflowState, Event

# class WorkflowMonitor:
#     """
#     Simple monitoring service for orchestrator that tracks workflows and events
#     """
#     def __init__(self, db_service):
#         self.db_service = db_service
#         self.queue_size = 0
#         self._history_workflow: list[WorkflowState] = []
#         self._history_event: list[Event] = []

#     def change(self, workflow, old_state, new_state):
#         workflow_state = WorkflowState(
#             workflow=workflow.workflow_name,
#             old_state=old_state,
#             new_state=new_state
#         )
#         self._history_workflow.append(workflow_state)

#     def append_event(self, event: Event):
#         self._history_event.append(event.model_copy())

#     @property
#     def current_workflow(self):
#         return self._history_workflow[-1] if len(self._history_workflow) > 0 else None
    

#     def history_workflow(self, n: int) -> tuple[WorkflowState, ...]:
#         n = min(n, len(self._history_workflow))
#         return tuple(self._history_workflow[-n:])
    
#     @property
#     def current_event(self):
#         return self._history_event[-1] if len(self._history_event) > 0 else None
    

#     def history_event(self, n: int) -> tuple[Event, ...]:
#         n = min(n, len(self._history_event))
#         return tuple(self._history_event[-n:])