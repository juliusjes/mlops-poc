import logging
import os
import sys

from fastapi import FastAPI
from ml_platform.shared.utils import init_log_path

filename = init_log_path("inference_client", "log")
if not os.path.exists(filename.parent):
    filename = os.path.join("outputs/logs/", filename.name)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(filename),
        logging.StreamHandler(sys.stdout),  # or omit sys.stdout for stderr
    ],
    force=True
)

from ml_platform.inference_client.routes import router

app = FastAPI()
app.include_router(router)

"""
TODO:
    1. add state saving to recover from restarts
        - save only important attributes to .json file
        - initialize object with state file
"""