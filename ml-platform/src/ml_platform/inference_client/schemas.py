from enum import Enum
from uuid import UUID

from ml_platform.inference_client.config import DatasetDescription
from ml_platform.inference_client.drifts import CP2DFSettings, CPNDFSettings
from pydantic import BaseModel


class SendRequest(BaseModel):
    uuid: UUID
    n : int
    labels: bool | None

class SendResponse(BaseModel):
    n: int
    preds: list

class SetDataSourceRequest(BaseModel):
    location: str
    dataset_description: DatasetDescription

class SetDataSourceResponse(BaseModel):
    found: bool

class DriftFunction(str, Enum):
    CPN = "CPNDriftFunc"
    CP2 = "CP2DriftFunc"
    DF = "DriftFunc"

class SetDriftRequest(BaseModel):
    drift_func_name: DriftFunction
    drift_func_kwargs: CP2DFSettings | CPNDFSettings| dict

class SetDriftResponse(BaseModel):
    found: bool

class GenerateValidationRequest(BaseModel):
    new_location: str
    size: int
    start: int | None = None

class GenerateValidationResponse(BaseModel):
    new_location: str|None