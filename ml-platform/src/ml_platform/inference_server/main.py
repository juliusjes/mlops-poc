import logging
import os
import sys

import mlflow
from fastapi import FastAPI

from ml_platform.shared.utils import init_log_path

filename = init_log_path("inference_server", "log")
if not os.path.exists(filename.parent):
    filename = os.path.join("/outputs/logs/", filename.name)
    

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(filename, mode="a"),
        logging.StreamHandler(sys.stdout), 
    ],
    force=True
)

from ml_platform.inference_server.config import Settings
from ml_platform.inference_server.routes import router

app = FastAPI()
app.include_router(router)

settings = Settings()

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)


"""
TODO:
    1. add drift detection
        - determine how to store reference window and current window
        - needs evidently integration
    2. add a state save when service is shutdown
"""