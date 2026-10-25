import logging

import mlflow

from ml_platform.shared.services.mlflow_service import MLflowService
from ml_platform.workers.trainer.training_config import TrainingConfig

logger = logging.getLogger(__name__)

class TrainingContext:
    def __init__(self, training_config: TrainingConfig, plugin):
        self.plugin = plugin
        self.training_config = training_config
        self.mlf_logger = MLflowService()

        logger.info(f"Context initialized with MLFLow tracking uri: {mlflow.get_tracking_uri()}")
        self.model, self.checkpoint = self._get_model()
        self.source = self._get_data_source()

        if self.checkpoint is not None:
            trcfg = self.checkpoint.get("training_config")
            if trcfg is None:
                raise RuntimeError("Checkpoint did not include a training configuration")

            self.training_config = TrainingConfig.model_validate(
                trcfg
                )

    def _get_data_source(self):
        raise NotImplementedError("Context implementations should implement '_get_data_source'")

    def _get_model(self) -> tuple:
        raise NotImplementedError("Context implementations should implement '_get_model'")