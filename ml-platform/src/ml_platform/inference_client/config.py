import os
from enum import Enum

import dotenv
from ml_platform.shared.paths import ENV_FILE
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class ColumnTypes(str, Enum):
    INT = "int"
    FLOAT = "float"
    STR = "str"
    BYTES = "bytes"
    DATE = "date"
    DATETIME = "datetime"
    BOOL = "bool"


class TargetInterpret(str, Enum): # TODO: depricated
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class DatasetDescription(BaseModel):
    sample_column: list[str, ColumnTypes]
    sample_shape: list[int]
    target_column: list[str, ColumnTypes]
    target_shape: list[int]
    order_column: list[str , ColumnTypes] | None
    target_interp: TargetInterpret

class Settings(BaseSettings):
    dotenv.load_dotenv(ENV_FILE)

    drift_func_name: str | None =  None
    drift_func_kwargs: dict | None = None
    #     "augmentation_func": "f",
    #     "augmentation_kwargs": {"constant": 1},
    #     "parameter_name": "offset",
    #     "param_values": [0, 2],
    #     "param_cps": [0.4, 0.8],
    #     "class_ratio": 0.5,
    #     "replace_all_labels": False,
    # }

    inference_endpoint :str = os.environ["INFERENCE_SERVER_URL"]
    send_path: str = "predict"
    data_source: str | None = None
    dsd: DatasetDescription| None = None
    sample_column: str = "waferMap" # in the source dataset
    target_column: str = "failure_label" # in the source dataset
    sample_shape: list = [1,128,128]
    inference_server_timeout: int = 120 # seconds, long for autoupdates
    max_timeouts: int = 100 # arbitrary
    label_delay: int = 20
    current_sample_index: int = 0
    
    DEBUG_loopback: bool = True # NOT USED reset the data index if it ends

    # S3 storage
    s3_access_key: str = os.environ["CREDENTIALS_ACCESS_KEY_ID"]
    s3_secret_key: str = os.environ["CREDENTIALS_SECRET_ACCESS_KEY"]
    s3_endpoint_override: str = os.environ["SERVER_ENDPOINT_URL"]


settings = Settings(
    drift_func_name=None,
    drift_func_kwargs=None
)