import logging
from typing import Callable
from uuid import UUID

import mlflow
import torch

from ml_platform.inference_server.config import EngineConfig, EngineState
from ml_platform.shared.data_processing import postprocessing, preprocessing

logger = logging.getLogger(__name__)

class Engine:
    def __init__(self, config: EngineConfig, uuid: UUID):
        self.uuid = uuid
        self.state = None
        self._set_state(EngineState.CREATED)

        self.config = config
        self.model_uri = config.model_uri
        self.device = config.device

        self.model = None
        self.model = self._load_pytorch_model()

        # preprocessing pipeline
        self.preprocessing = config.preprocessing
        self.preprocessing_pipeline = None
        self.preprocessing_pipeline = self._build_pipeline(self.preprocessing)

        # postprocessing pipeline
        self.postprocessing = config.postprocessing
        self.postprocessing_pipeline = None
        self.postprocessing_pipeline = self._build_pipeline(self.postprocessing)

        self._set_state(EngineState.AVAILABLE)


    def _set_state(self, state:EngineState):
        if self.state is not state:
            logger.info(f"Engine-{str(self.uuid)[:5]} : {self.state or 'None'} -> {state}")
            self.state = state


    def get_state(self) -> EngineState:
        return self.state


    def get_model_name(self) -> str:
        if self.model is None:
            return f"NO_MODEL"

        name, version = self.model_uri.split("/")[-2:]
        return f"{name}-{version}"


    def _load_pytorch_model(self):
        model = mlflow.pytorch.load_model(self.model_uri, map_location=torch.device(self.device))
        return model


    def preprocess(self, x):
        if self.preprocess is not None:
            for pp in self.preprocessing_pipeline:
                x = pp(x)
        return x


    def postprocess(self, x):
        if self.postprocess is not None:
            for pp in self.postprocessing_pipeline:
                x = pp(x)
        return x
    

    def _build_pipeline(self, parts) -> list | None:
        pipe = []
        for pp in parts:

            try:
                f = getattr(preprocessing, pp)
                assert isinstance(f, Callable), f"Preprocessing function {pp} is not callable"
                pipe.append(f)

            except Exception as e:
                logger.error(f"Failed to build Preprocessing pipeline: {str(e)}")
                raise ValueError

        return pipe


    def _predict(self, x):
        self._set_state(EngineState.PREDICTING)

        if len(self.preprocessing_pipeline) > 0:
            x = self.preprocess(x)

        x = x.to(device=self.device)
        with torch.no_grad():
            output = self.model(x)

        if len(self.preprocessing_pipeline) > 0:
            output = self.postprocess(output)

        self._set_state(EngineState.AVAILABLE)

        return output
