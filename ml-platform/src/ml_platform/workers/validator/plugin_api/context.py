from ml_platform.shared.services.mlflow_service import MLflowService


class ValidationContext:

    def __init__(self, validation_config, plugin):
        self.plugin = plugin
        self.validation_config = validation_config
        self.mlf_logger = MLflowService()

        self.model = self._get_model()
        self.source = None #self._get_data_source()

    def _get_model(self):
        raise NotImplementedError

    def _get_data_source(self):
        raise NotImplementedError
