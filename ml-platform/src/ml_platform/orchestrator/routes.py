import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ml_platform.orchestrator.events import (
    DataUpdateParams,
    DeployParams,
    DriftStartParams,
    Event,
    EventStatus,
    EventType,
    ManualStartParams,
    TestParams,
    ValidateParams,
)
from ml_platform.orchestrator.manager import orch
from ml_platform.orchestrator.schemas import (
    DataUpdateRequest,
    DeployRequest,
    DriftStartRequest,
    EventHistoryListResponse,
    EventListRequest,
    EventListResponse,
    EventResponse,
    EventStatsResponse,
    ManualStartRequest,
    ValidateRequest,
    build_deploy_params,
)

#from ml_platform.shared.utils import tuple_to_dict
from ml_platform.workers.trainer.training_config import TrainingConfig
from ml_platform.workers.validator.validation_config import ValidationConfig

logger = logging.getLogger(__name__)


router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "healthy"}


@router.get( "/events/stats", response_model=EventStatsResponse)
async def get_event_stats():
    ...

@router.get("/events/pending", response_model=EventListResponse)
async def get_event_pending(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    events_data = orch.event_handler.get_events_by_status(
        status=EventStatus.PENDING.value,
        limit=limit,
        offset=offset,
    )
    events_data = [] if not events_data else events_data
    return {
        "events": events_data,
        "limit": limit,
        "offset": offset,
        "has_more": False,
    }


@router.get("/events/complete", response_model=EventListResponse)
async def get_event_complete(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    events_data = orch.event_handler.get_events_by_status(
        status=EventStatus.COMPLETE.value,
        limit=limit,
        offset=offset,
    )
    events_data = [] if not events_data else events_data
    return {
        "events": events_data,
        "limit": limit,
        "offset": offset,
        "has_more": False,
    }

@router.get("/events/last", response_model=EventListResponse)
async def get_event_last(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    events_data = orch.event_handler.n_last_events(
        n=limit
        )
    events_data = [] if not events_data else events_data

    return {
        "events": events_data,
        "limit": limit,
        "offset": 0,
        "has_more": False,
    }


@router.get("/events/{event_id}", response_model=EventResponse)
async def get_event(event_id: UUID):
    event_data = orch.event_handler.get_event_by_id(event_id)
    if event_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No event with id {event_id} found"
            )
    return event_data


@router.get( "/events/{event_id}/history", response_model=EventHistoryListResponse)
async def get_event_history(event_id: UUID):
    state_data = orch.workflow_handler.get_states_by_id(event_id)
    state_data = [] if not state_data else state_data
    return {
        "event_id": event_id,
        "history": state_data
    }



@router.post( "/events/manual-start", status_code=status.HTTP_202_ACCEPTED)
async def manual_start(request: ManualStartRequest):
    """
    Validate configs and enqueue manual training.
    """
    try:
        #logger.info(request.training_config)
        tcfg = TrainingConfig.model_validate(request.training_config)
        vcfg = ValidationConfig.model_validate(request.validation_config)
        #logger.debug(f"{type(tcfg)}")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    event = Event(
        event_type=EventType.MANUAL_START,
        params=ManualStartParams(
            training_config=tcfg,
            validation_config=vcfg,
            deploy_criteria=request.deploy_criteria
        )
    )
    await orch.add_event(event)

    # db.add(event)

    return {
        "message": "event queued",
        "event_id": str(event.id),
    }



@router.post( "/events/drift-start", status_code=status.HTTP_202_ACCEPTED, )
async def drift_start(request: DriftStartRequest):
    """
    Validate configs and retrain
    """
    try:
        tcfg = TrainingConfig.model_validate(request.training_config)
        vcfg = ValidationConfig.model_validate(request.validation_config)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    
    event = Event(
        event_type=EventType.DRIFT_START,
        params=DriftStartParams(
            training_config=tcfg,
            validation_config=vcfg,
        ),
    )

    await orch.add_event(event)

    return {
        "message": "event queued",
        "event_id": str(event.id),
    }


@router.post("/events/validate", status_code=status.HTTP_202_ACCEPTED)
async def validate(request: ValidateRequest):
    event = Event(
        event_type= EventType.VALIDATE,
        params=ValidateParams(validation_config = request.validation_config)
    )
    await orch.add_event(event)
    return {
        "message": "event queued",
        "event_id": str(event.id),
    }


@router.post( "/events/data-update", status_code=status.HTTP_202_ACCEPTED)
async def data_update( request: DataUpdateRequest):
    """
    Validate endpoint/data location and enqueue update.
    """
    new_endpoint = request.new_data_endpoint
    dataset_location = request.dataset_location

    event = Event(
        event_type=EventType.DATA_UPDATE,
        params=DataUpdateParams(
            dataset_location=dataset_location,
            new_data_endpoint=new_endpoint
        ),
    )

    # db.add(event)

    await orch.add_event(event)

    return {
        "message": "event queued",
        "event_id": str(event.id),
    }



@router.post( "/events/deploy", status_code=status.HTTP_202_ACCEPTED)
async def deploy(request: DeployRequest):
    event = Event(
        event_type=EventType.DEPLOY,
        params=build_deploy_params(request),
    )

    await orch.add_event(event)

    return {
        "message": "event queued",
        "event_id": str(event.id),
    }



@router.post( "/events/test", status_code=status.HTTP_202_ACCEPTED)
async def test_event():
    event = Event(
        event_type=EventType.TEST,
        params=TestParams(),
    )

    await orch.add_event(event)

    return {
        "message": "test event queued",
        "event_id": str(event.id),
    }
