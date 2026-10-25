import importlib
import json
import logging
from uuid import UUID

from ml_platform.shared.services.postgres_service import PostgresService
from ml_platform.workers.validator.validation_config import ValidationConfig


logger = logging.getLogger(__name__)

def parse_event_params(event: dict, tr_result:dict|None=None) -> tuple[str, ValidationConfig]:

    # parse event id
    # parse params

    id = event.get("id", "ID")
    params = event["params"]

    validation_config = params["validation_config"]

    logger.info(f"matching training result had entries for \
                {tr_result.get("result").keys() if tr_result is not None else []}")
    model_uri = tr_result["result"].get("model_uri") if tr_result is not None else None

    if model_uri:    
        validation_config["model_source"] = "mlflow"
        logger.info(f"Set validation config model_source to \
                    {validation_config["model_source"]}")

        validation_config["model_load_kwargs"]["model_uri"] = model_uri
        logger.info(f"Set validation config model_load_kwargs.model_uri to \
                    {validation_config["model_load_kwargs"]["model_uri"]}")

    return id, ValidationConfig.model_validate(validation_config)


training_result_query = """
        SELECT result
        FROM workflow_states
        WHERE event_id = %s AND workflow = 'Training-workflow' AND new_state = 'training'
"""


job_query = """
    SELECT *
    FROM events
    WHERE id = %s
"""

class Validator:
    def __init__(self, job_id: str|UUID|None=None, event=None):

        self.ps = PostgresService(db_name="orchestrator_db")


        # load job
        matching_training_result = None
        if job_id is not None:
            event = self.ps.fetch_one(job_query, (job_id,))
            logger.info(f"Found {0 if not event else 1} event_params")

            matching_training_result = self.ps.fetch_one(training_result_query, (job_id,))
            logger.info(f"Found {0 if not matching_training_result else 1} matching training results")

        self.id, self.validation_config = parse_event_params(event, matching_training_result)
        self.short_id = str(self.id)[0:5] if self.id != "ID" else "ID"

        logger.info(f"Event id: {self.id!s} ({self.short_id})")

        self.plugin = None #self.load_plugin("")
        self.builtin = self.load_plugin("builtin_val")

        self.ctx = self.build_validation_context()

        self.loop = self.build_validation_loop()

        result = self.validate_with_loop()

        self.finalize(result)

    def load_plugin(self, package_name: str | None):

        module = importlib.import_module(package_name)
        logger.info(f"Using module {module.__name__} from {module.__file__}")
        if not hasattr(module, "PLUGIN"):
            raise ValueError("Plugin must expose PLUGIN")

        plugin = module.PLUGIN
        logger.info(f"Available: {plugin.keys()}")

        return plugin

    def build_validation_context(self):
        # temporary
        return self.builtin["context"]["basic_context"](
            self.validation_config, self.plugin
            )

    def build_validation_loop(self):
        # temporary
        return self.builtin["loop"]["pytorch_classification_validation_loop"]()

    def validate_with_loop(self):
        self.ctx.mlf_logger.start_run(f"Validation-{self.short_id!s}")

        result = self.loop.validate(self.ctx)
        # self.ctx.mlf_logger.log_params(
        #     {
        #         ""
        #     }
        # )

        logger.info(f"Validation pass={result.validation_info["passed"]}")

        self.ctx.mlf_logger.end_run()
        self.ctx.mlf_logger.update_model_tags(
            self.validation_config.model_load_kwargs.model_uri,
            {"validation_ok": result.validation_info["passed"]})

        return result


    def finalize(self, validation_result):  
        sql = """
            UPDATE workflow_states
            SET result = %s::jsonb
            WHERE event_id = %s AND new_state = 'validating' AND workflow = 'Validation-workflow'
        """

        result_json = json.dumps(validation_result.to_dict())

        self.ps.execute(sql, (result_json, self.id))


if __name__ == "__main__":
    from ml_platform.workers.trainer.training_config import (
        ColumnTypes,
        DatasetDescription,
        TargetInterpret,
    )
    event = {
        "params": {"validation_config": {
            "model_source": "mlflow",
            "model_load_kwargs": {
                "model_uri": "models:/simple_resnet_exp2/52"
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
                                "data_source": "masters-thesis/expbe299a15b35a310b/expdata/0/offline/",
                                "n_samples" : 2500,
                                "metric": "accuracy_score",
                                "max_value": 1.0,
                                "min_value": 0.0
                            },
                            {
                                "rule_name": "recall_W200",
                                "data_source": "masters-thesis/expbe299a15b35a310b/expdata/0/offline/",
                                "n_samples" :2500,
                                "metric": "recall_score",
                                "max_value": 1.0,
                                "min_value": 0.0
                            },
                            {
                                "rule_name": "f1_W200",
                                "data_source": "masters-thesis/expbe299a15b35a310b/expdata/0/offline/",
                                "n_samples" : 2500,
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

    Validator(event=event)