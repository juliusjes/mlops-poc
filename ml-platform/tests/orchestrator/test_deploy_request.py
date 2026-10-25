import sys
from types import SimpleNamespace

sys.modules.setdefault("psycopg_pool", SimpleNamespace(ConnectionPool=object))

from ml_platform.orchestrator.schemas import DeployRequest, build_deploy_params


def test_deploy_request_binds_its_model_uri_to_validation():
    request = DeployRequest.model_validate(
        {
            "model_uri": "models:/test-model/7",
            "deploy_criteria": "validated",
            "validation_config": {
                "model_source": "mlflow",
                "model_load_kwargs": {},
                "dataset_description": {
                    "sample_column": ["X", "bytes"],
                    "sample_shape": [8],
                    "target_column": ["y", "int"],
                    "target_shape": [1],
                    "order_column": ["write_timestamp", "datetime"],
                    "target_interp": "classification",
                },
                "validations": [
                    {
                        "validation_type": "standalone",
                        "rules": [
                            {
                                "data_source": "pytest/main/data",
                                "metric": "accuracy_score",
                                "min_value": 0.0,
                                "max_value": 1.0,
                            }
                        ],
                    }
                ],
            },
        }
    )

    params = build_deploy_params(request)

    assert params.model_uri == "models:/test-model/7"
    assert params.validation_config.model_load_kwargs.model_uri == "models:/test-model/7"
    assert params.deploy_criteria == "validated"
