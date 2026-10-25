import logging

from fastapi import APIRouter, HTTPException

from ml_platform.inference_server.config import PluginConfig
from ml_platform.inference_server.coordinator import coordinator
from ml_platform.inference_server.schemas import (
    AddPluginRequest,
    AddPluginResponse,
    AvailableModelsResponse,
    DeployRequest,
    DeployResponse,
    ListPluginsRequest,
    ListPluginsResponse,
    PredictRequest,
    PredictResponse,
    RemoveDeploymentRequest,
    RemoveDeploymentResponse,
    RemovePluginRequest,
    RemovePluginResponse,
    ResetPluginLogRequest,
    ResetPluginLogResponse,
    ResetPluginsRequest,
    ResetPluginsResponse,
    StateResponse,
    SyncPluginsRequest,
    SyncPluginsResponse,
)
from ml_platform.shared.data_processing.data_processing import (
    bytes_to_array,
    decode_from_json,
)

router = APIRouter()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ROUTES

@router.post("/deploy_model", response_model=DeployResponse)
async def update(payload: DeployRequest):
    logger.debug("Recieved model deployment request")

    ecfg = payload.engine_config
    ccfg = payload.collector_config
    pcfgs = payload.plugin_configs
    uuid, success = await coordinator.add_deployment(
            engine_config=ecfg,
            colletor_config=ccfg,
            plugin_configs=pcfgs,
        )

    return {
        "uuid": uuid,
        "success": success
    }


@router.post("/remove_deployment", response_model=RemoveDeploymentResponse)
async def remove_deployment(payload: RemoveDeploymentRequest):
    if not coordinator.is_deployed(payload.uuid):
        raise HTTPException(status_code=409, detail=f"No deployment with uuid={payload.uuid} present")

    success = await coordinator.remove_deployment(payload.uuid)
    assert success

    remaining = coordinator.get_model_names()

    return {
        "uuid": payload.uuid,
        "remaining": remaining,
    }


@router.post( "/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest):
    logger.debug("Received prediction request")

    uuid = payload.deployment_uuid    

    
    if not coordinator.is_deployed(uuid):
        # maybe raise httpexception
        return { "y_pred": [] }

    raw_x = decode_from_json(payload.X) # decode from json

    prediction = await coordinator.deployment_registry[uuid].predict(raw_x, label=payload.y)

    if isinstance(prediction, tuple): # TODO: update
        return {
            "y_pred": prediction[0].tolist(),
            "emb": prediction[1].tolist()
        }
    elif isinstance(prediction, dict):
        return {
            "y_pred": prediction["logits"].tolist(),
            "emb": prediction["emb"].tolist()
        }


@router.get("/available_models", response_model=AvailableModelsResponse)
async def available_models():
    names = coordinator.get_model_names()
    return {
        "available": names
    }


@router.post("/add_plugins", response_model=AddPluginResponse)
async def add_plugins(payload: AddPluginRequest):

    if not coordinator.is_deployed(payload.uuid):
        raise HTTPException(status_code=409, detail=f"No deployment with uuid={payload.uuid} present")

    cfgs = []
    for pcfg in payload.configs:
        cfgs.append(PluginConfig(
            plugin_type=pcfg.plugin_type,
            name=pcfg.name,
            parameters=pcfg.parameters,
            trace_fields=pcfg.trace_fields,
            notifiers=pcfg.notifiers,
            reference_window_update=pcfg.reference_window_update,
            reset_on_detection=pcfg.reset_on_detection,
            ))

    results = coordinator.deployment_registry[payload.uuid].add_plugins(cfgs)
    return {
        "uuid": payload.uuid,
        "success": results,
    }


@router.post("/remove_plugin", response_model=RemovePluginResponse)
async def remove_plugin(payload: RemovePluginRequest):
    if not coordinator.is_deployed(payload.uuid):
        raise HTTPException(status_code=409, detail=f"No deployment with uuid={payload.uuid} present")

    n_removed, remaining = coordinator.deployment_registry[payload.uuid].remove_plugin(
        method=payload.method,
        name=payload.name,
    )
    return {
        "uuid": payload.uuid,
        "n_removed": n_removed,
        "remaining": remaining
    }

@router.post("/list_plugins", response_model=ListPluginsResponse)
async def list_plugins(payload: ListPluginsRequest):
    if not coordinator.is_deployed(payload.uuid):
        raise HTTPException(status_code=409, detail=f"No deployment with uuid={payload.uuid} present")

    plugins = coordinator.deployment_registry[payload.uuid].get_plugins()

    return {
        "uuid": payload.uuid,
        "plugins": plugins,
    }

@router.post("/sync_plugins", response_model=SyncPluginsResponse)
async def sync_plugins(payload: SyncPluginsRequest):
    """
    Synchornize the plugins with the predictions. Blocks any prediction requests while waiting for plugins to finish. 
    """
    if not coordinator.is_deployed(payload.uuid):
        raise HTTPException(status_code=409, detail=f"No deployment with uuid={payload.uuid} present")

    await coordinator.deployment_registry[payload.uuid].sync_observation_queue()

    return {
        "uuid": payload.uuid
    }

@router.post("/reset_plugins", response_model=ResetPluginsResponse)
async def reset_plugins(payload: ResetPluginsRequest):
    if not coordinator.is_deployed(payload.uuid):
        raise HTTPException(status_code=409, detail=f"No deployment with uuid={payload.uuid} present")

    coordinator.deployment_registry[payload.uuid].reset_plugins()
    coordinator.deployment_registry[payload.uuid].reset_collector() # TODO: conseider moving this elsewhere

    return {
        "uuid": payload.uuid
    }
        


@router.post("/reset_plugin_log", response_model=ResetPluginLogResponse)
async def reset_plugin_log(payload: ResetPluginLogRequest):
    if not coordinator.is_deployed(payload.uuid):
        raise HTTPException(status_code=409, detail=f"No deployment with uuid={payload.uuid} present")

    nf, of = coordinator.deployment_registry[payload.uuid].reset_plugin_logger()

    return {
        "uuid": payload.uuid,
        "log_file": of,
        "new_log_file": nf
    }


# @router.post("/flush_data", response_model=FlushResponse)
# async def flush_data(payload: FlushRequest):
#     logger.debug("Recieved buffer flush request")
#     location, n_samples = await service.flush(payload.location)
#     return {
#         "location": location,
#         "n_samples": n_samples
#     }


# @router.post("/reset_dd", response_model=ResetDDResponse)
# async def reset_dd(payload: ResetDDRequest):
#     logger.debug("Recieved reset request")
#     new_log = payload.create_new_log
#     success, log_file = evidently_service._resetdd(new_log)
#     return {
#         "success": success,
#         "log_file": str(log_file),
#         }


# @router.post("/remove_dd", response_model=RemoveDDResponse)
# async def remove_dd(payload: RemoveDDRequest):
#     logger.debug("Recieved remove dd request")
#     method = payload.method
#     name = payload.name
#     removed, remaining = evidently_service.remove_dds(method, name)
#     return {
#         "removed": removed,
#         "remaining": remaining
#     }

# @router.post("/add_dd", response_model=AddDDResponse)
# async def add_dd(payload: AddDDRequest):
#     logger.debug("recieved add dd request")

#     data = {
#         "detector_type": payload.detector_type,
#         "tracing": payload.tracing,
#         #"trace_fields": payload.trace_fields,
#         "parameters": payload.kwargs,
#         "in_dash": payload.in_dash
#     }
#     d = TypeAdapter(DetectorUnion).validate_python(data)
    
#     success, log_file = evidently_service.add_dd(d)

#     return {
#         "success": success,
#         "log_file": str(log_file)
#     }


@router.get("/state", response_model=StateResponse, status_code=200)
def get_state():
    return {"state": str(coordinator.state)}
