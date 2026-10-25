from typing import Any
#from pydantic import BaseModel
from ml_platform.workers.trainer.plugin_api.context import TrainingContext

class TrainingResult:
    def __init__(self, model_info):
        self.model_info: Any = model_info


class TrainingLoop:

    def train(self, ctx: TrainingContext) -> TrainingResult:
        raise NotImplementedError