from venv import logger

import torch
from ml_platform.workers.trainer.plugin_api.context import TrainingContext
from ml_platform.workers.trainer.training_config import (
    DataSource,
    ModelSourceType,
)

from builtin.data_sources.lakefs_source import LakeFSSource


class BasicContext(TrainingContext):
    def __init__(self, training_config, plugin):
        super().__init__(training_config, plugin)

        #print(type(self.training_config))

    def _get_data_source(self):

        match self.training_config.dataset_implementation:
            case DataSource.LAKEFSSOURCE:

                source = LakeFSSource(
                    self.training_config.dataset_location,
                    self.training_config.dataset_description
                    )

            case _:
                # user defined source
                pass

        return source

    def _get_model(self) -> tuple:

        checkpoint = None
        device = torch.device(self.training_config.device)
        match self.training_config.model_source_type:

            case ModelSourceType.MLFLOW:
                

                checkpoint = self.mlf_logger.load_checkpoint_from_model_uri(
                    self.training_config.model_load_kwargs.model_uri,
                    device,
                )

                model = self.plugin["model"][
                    self.training_config.model_load_kwargs.model_name
                ]()

                model.load_state_dict(checkpoint["model_state_dict"])

            case ModelSourceType.PYTHON:
                model = self.plugin["model"][self.training_config.model_load_kwargs.model_name]()

            case _:
                raise ValueError

        model.to(device)

        return model, checkpoint