import importlib
import json
import logging
from uuid import UUID

import torch

from ml_platform.shared.exceptions import TrainerException
from ml_platform.shared.services.postgres_service import PostgresService
from ml_platform.workers.trainer.plugin_api.context import TrainingContext
from ml_platform.workers.trainer.plugin_api.training_loop import (
    TrainingLoop,
    TrainingResult,
)
from ml_platform.workers.trainer.training_config import (
    ColumnTypes,
    DatasetDescription,
    DataSource,
    ModelSourceType,
    TargetInterpret,
    TrainingConfig,
)

logger = logging.getLogger(__name__)

job_query = """
    SELECT *
    FROM events
    WHERE id = %s
"""
# import mlflow
# import logging
# logging.getLogger("mlflow").setLevel(logging.DEBUG)

def parse_event_params(event: dict) -> tuple[str, TrainingConfig]:

    # parse event id
    # parse params

    id = event.get("id", "ID")
    params = event["params"]

    training_config = params["training_config"]
    
    return id, TrainingConfig.model_validate(training_config)
    


class Trainer:
    def __init__(self, job_id: str|UUID|None = None, event=None, postgres_override_url=None):
        self.ps = PostgresService(
            postgres_url=postgres_override_url,
            db_name="orchestrator_db"
            )

        # load job
        if job_id is not None:
            event = self.ps.fetch_one(job_query, (job_id,))

        self.id, self.event_params = parse_event_params(event)
        if not torch.cuda.is_available():
            logger.warning("Falling back on cpu, CUDA not available")
            self.event_params.device = "cpu"

        self.short_id = str(self.id)[0:5] if self.id != "ID" else "ID"

        placeholder = "example_training"
        self.plugin = self.load_plugin(placeholder)
        self.builtin = self.load_plugin("builtin")

        self.ctx = self.build_training_context()

        self.loop = self.build_loop()

        result = self.train_with_loop()

        self.finalize(result)


    def build_training_context(self) -> TrainingContext:
        return self.builtin["context"]["basic_context"](self.event_params, self.plugin)

    def build_loop(self) -> TrainingLoop:

        loop = None
        # try to find in builtin
        try:
            loop_cls = self.builtin["loop"][self.event_params.loop_name]
        except KeyError:
            pass

        try:
            loop_cls = self.plugin["loop"][self.event_params.loop_name]
        except KeyError:
            pass

        try:
            loop = loop_cls(**self.event_params.loop_kwargs)
        except Exception as e:
            print(str(e))

        return loop

    def load_plugin(self, package_name: str | None):

        module = importlib.import_module(package_name)
        logger.info(f"Using module {module.__name__} from {module.__file__}")
        if not hasattr(module, "PLUGIN"):
            raise ValueError("Plugin must expose PLUGIN")

        plugin = module.PLUGIN
        logger.info(f"Available: {plugin.keys()}")

        return plugin

    def train_with_loop(self):
        self.ctx.mlf_logger.start_run(f"Training-{self.short_id!s}")

        self.ctx.mlf_logger.log_params(dict(self.event_params))

        # TODO
        #self.ctx.mlf_logger.log_training_input()
        #self.ctx.mlf_logger.log_testing_input()
        try:
            result: TrainingResult = self.loop.train(self.ctx)
        except Exception as e:
            logger.error(f"Training loop was intrupted by error: {e!s}")
            raise e

        model_uri = result.model_info.model_uri

        mv = self.ctx.mlf_logger.register_model(model_uri, self.event_params.model_register_name, tags={"validation_ok": False})

        result.registered_name = mv.name
        result.version = mv.version

        self.ctx.mlf_logger.end_run()

        return result


    def finalize(self, training_result: TrainingResult):
        sql = """
            UPDATE workflow_states
            SET result = %s::jsonb
            WHERE event_id = %s AND new_state = 'training' AND workflow = 'Training-workflow'
        """
        if self.id == "ID":
            logger.info("Skipped result insertion for event with id='ID'")
            return

        result_json = json.dumps(
            {
                "run_id": training_result.model_info.run_id,
                "model_uri_uuid": training_result.model_info.model_uri,
                "model_uri": f"models:/{training_result.registered_name}/{training_result.version}",
                "model_name": training_result.registered_name,
                "model_version": training_result.version
            }
                )

        self.ps.execute(sql, (result_json, self.id))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    test_event = {
        "params": {
        "training_config": {
                "dataset_location": "masters-thesis/expedc681c7d6c3d260/expdata/0/offline/",
                "n_training_samples" : 3000,
                "model_source_type": "python",
                "model_load_kwargs": {
                        "model_name": "simple_resnet"
                    },

                "loop_name": "pytorch_classification_loop",
                "loop_kwargs": {},

                "train_loader_kwargs": {"batch_size": 64},
                "test_loader_kwargs": {"batch_size": 64},
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss",
                "epochs": 3,
                "device": "cuda",
                "register_device": "cpu",
                
                "run_name": "Training_run",
                "model_register_name": "simple_resnet_exp2",

                "dataset_implementation": "lakefs_source",
                "dataset_description": {
                    "sample_column": ["waferMap", "bytes"],
                    "sample_shape": [1, 128, 128],
                    "target_column": ["failure_label", "int"],
                    "target_shape": [1],
                    "order_column": ["write_timestamp", "datetime"],
                    "target_interp": "classification"
                }
                
            },
        "validation_config": {
            "model_source": "mlflow",
            "model_load_kwargs": {
                "model_uri": "placemodel"
            },
            "dataset_implementation": "lakefs_source",
            "dataset_description": {
                "sample_column": ["waferMap", "bytes"],
                "target_column": ["failure_label", "int"],
                "order_column": ["write_timestamp", "datetime"],
                "target_interp": "classification",
                "sample_shape": [1, 128, 128],
                "target_shape": [1]
                },
            "validations": [
                    {
                        "validation_type": "standalone",
                        "rules": [
                            {
                                "rule_name": "accuracy_W200",
                                "data_source": "masters-thesis/expedc681c7d6c3d260/expdata/0/offline/",
                                "n_samples" : 200,
                                "metric": "accuracy_score",
                                "max_value": 1.0,
                                "min_value": 0.0
                            },
                            {
                                "rule_name": "recall_W200",
                                "data_source": "masters-thesis/expedc681c7d6c3d260/expdata/0/offline/",
                                "n_samples" : 200,
                                "metric": "recall_score",
                                "max_value": 1.0,
                                "min_value": 0.0
                            },
                            {
                                "rule_name": "f1_W200",
                                "data_source": "masters-thesis/expedc681c7d6c3d260/expdata/0/offline/",
                                "n_samples" : 200,
                                "metric": "f1_score",
                                "max_value": 1.0,
                                "min_value": 0.0
                            }
                        ]
                    }
                ]
            }
        }
    }
    # from ml_platform.shared.services.mlflow_service import MLflowService
    # s =MLflowService()
    # model_info = mlflow.models.get_model_info("models:/test_model0/24")
    # print(model_info.run_id)

    t = Trainer(
        job_id="edcb0e39-98cf-44d9-8d95-3092fd153ff8",
        postgres_override_url=""
    )

    