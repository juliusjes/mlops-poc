import logging

from fastapi import APIRouter, HTTPException
from ml_platform.inference_client.schemas import (
    GenerateValidationRequest,
    SendRequest,
    SendResponse,
    SetDataSourceRequest,
    SetDataSourceResponse,
    SetDriftRequest,
    SetDriftResponse,
)
from ml_platform.inference_client.services import client

router = APIRouter()
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)


@router.post("/set_drift_func", response_model=SetDriftResponse)
async def set_drift_func(payload: SetDriftRequest):
    found = client.set_drift_func(payload.drift_func_name, payload.drift_func_kwargs)
    return {"found": found}


@router.post("/set_data_source", response_model=SetDataSourceResponse)
async def set_data_source(payload: SetDataSourceRequest):
    found = client.set_data_source(payload.location, payload.dataset_description)
    return {"found": found}


@router.post("/send_n_samples", response_model=SendResponse)
async def send_n_samples(payload: SendRequest):
    try:
        n = int(payload.n)
        assert n > 0
        labels = payload.labels
    except Exception as e:
        raise HTTPException( status_code=400, detail=f"Invalid request parameters: n={n}, labels={labels}" )
    
    preds = await client.send_n_samples(n, payload.uuid ,label=labels)
    if len(preds) == 0:
        raise HTTPException(status_code=400, detail="Out of samples")

    
    return {
        "n": n,
        "preds": preds
    }


@router.get("/is_out_of_samples")
async def is_out_of_samples():
    oos, csi, ts = client.is_out_of_samples()
    return {
        "out_of_samples": oos,
        "current_sample_index": csi,
        "total_samples": ts
    }


@router.post("/generate_drifted_dataset")
async def generate_drifted_dataset(payload: GenerateValidationRequest):
    success = client.generate_drifted_validation_dataset(
        payload.new_location,
        payload.size,
        payload.start
    )
    return {
        "new_location": payload.new_location if success else None
    }


@router.get("/health")
async def health():
    return {"status": "ok"}