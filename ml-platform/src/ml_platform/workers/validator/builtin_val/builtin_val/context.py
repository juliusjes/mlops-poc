
from ml_platform.workers.trainer.training_config import DataSource, ModelSourceType
from ml_platform.workers.validator.plugin_api.context import ValidationContext
from ml_platform.workers.validator.validation_config import ValidationConfig

from .data_sources.lakefs_source import LakeFSSource


class BasicContext(ValidationContext):
    def __init__(self, validation_config: ValidationConfig, plugin):
        super().__init__(validation_config, plugin)


    def _get_data_source(self, location):

        match self.validation_config.dataset_implementation:
            case DataSource.LAKEFSSOURCE:

                source = LakeFSSource(
                    location,
                    self.validation_config.dataset_description
                    )


            case _:
                # user defined source
                raise NotImplementedError


        return source

    def _get_model(self):

        match self.validation_config.model_source:

            case ModelSourceType.MLFLOW:
                model = self.mlf_logger.get_model(
                    self.validation_config.model_load_kwargs.model_uri,
                    self.validation_config.device
                    )

            case ModelSourceType.PYTHON:
                model = self.plugin["model"][self.validation_config.model_load_kwargs.model_name]()

            case _:
                raise ValueError

        return model
