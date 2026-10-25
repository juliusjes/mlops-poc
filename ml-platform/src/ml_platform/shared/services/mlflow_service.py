from __future__ import annotations

import os
import tempfile
from typing import Any

import dotenv
import mlflow
import numpy as np
import torch
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.dataset_source_registry import register_dataset_source
from mlflow.models import ModelSignature, infer_signature
from mlflow.tracking import MlflowClient

from ml_platform.shared.paths import ENV_FILE
from ml_platform.shared.utils import get_healthy_endpoint, info


class LakeFSDatasetSource(DatasetSource):
    def __init__(self, uri: str):
        self.uri = uri

    @staticmethod
    def _get_source_type() -> str:
        return "lakefs"

    def load(self) -> str:
        """
        For now just return the URI.
        In the future this could download or mount the dataset.
        """
        return self.uri

    @staticmethod
    def _can_resolve(raw_source: Any) -> bool:
        return (
            isinstance(raw_source, str)
            and raw_source.startswith("lakefs://")
        )

    @classmethod
    def _resolve(cls, raw_source: Any) -> LakeFSDatasetSource:
        return cls(raw_source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
        }

    @classmethod
    def from_dict(cls, source_dict: dict[str, Any]) -> LakeFSDatasetSource:
        return cls(source_dict["uri"])

    def __repr__(self):
        return f"LakeFSDatasetSource(uri={self.uri!r})"

    def __str__(self):
        return self.uri

register_dataset_source(LakeFSDatasetSource) # TODO:


class MLflowService:
    """
    Singleton service for communicating with mlflow
    """
    instance = None

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance
    

    def __init__(self):
        dotenv.load_dotenv(ENV_FILE)

        uri = get_healthy_endpoint(
            [
                os.environ["MLFLOW_TRACKING_URI"],
                os.environ["MLFLOW_TRACKING_URI"].replace("mlflow", "ml-platfrom-mlflow-1"),
                "http://localhost:5000",
            ],
            "/health",
            timeout=0.5

        )
        print("USING TRACKING URI:", uri)
        mlflow.set_tracking_uri( uri )
        mlflow.set_system_metrics_sampling_interval(1)
        mlflow.set_system_metrics_samples_before_logging(3)

        self.client = MlflowClient()

    def _strip_source(self, source):
        source = source.removeprefix("s3://")
        source = source.removeprefix("lakefs://")
        return source 

    def start_run(self, run_name: str):
        return mlflow.start_run(run_name=run_name, log_system_metrics=True)

    def log_params(self, params: dict):
        for k, v in params.items():
            params[k] = str(v) if not isinstance(v, (float, int, str)) else v

        mlflow.log_params(params)

    def log_param(self, key: str, value: str|float):
        mlflow.log_param(key, value)

    def log_metric(self, key, value, step=None):
        mlflow.log_metric(key, value, step=step)

    def log_metrics( self, metrics: dict[str, float], steps):
        assert len(steps) == len(metrics.items())
        for name, value in zip(metrics.items(), steps):
            mlflow.log_metric(name, value)

    def log_model( self, model, input_example, step=0, name=None, signature: ModelSignature = None, device="cpu"):
        name = name or model.__class__.__name__

        if device is not None:
            model = model.to(device)

            if str(device) == "cpu":
                os.environ["MLFLOW_DEFAULT_PREDICTION_DEVICE"] = "cpu"
            elif str(device).startswith("cuda"):
                os.environ["MLFLOW_DEFAULT_PREDICTION_DEVICE"] = "cuda"

        return mlflow.pytorch.log_model(
            model,
            step=step,
            name=name,
            serialization_format="pt2",
            input_example=input_example,
            signature=signature,
        )



    def register_model(self, model_uri, name, tags={}):
        # TODO: combine somehow with log_model
        info(f"Registering model with uri {model_uri}, tags {tags}")
        mv = mlflow.register_model(model_uri, name, tags=tags)
        return mv

    def end_run(self):
        mlflow.end_run()

    def log_training_input(self, df, source, additional_tags={}):
        self._log_input(df, source, context="training", additional_tags=additional_tags)

    def log_testing_input(self, df, source, additional_tags={}):
        self._log_input(df, source, context="testing", additional_tags=additional_tags)

    def log_validation_input(self, df, source, suffix="", additional_tags={}):
        if suffix != "":
            context = f"validation_{suffix}"
        else:
            context = "validation"

        assert all(isinstance(v, str) for v in additional_tags.values()), additional_tags

        self._log_input(df, source, context=context, additional_tags=additional_tags)

    def _log_input(self, df, source, context, additional_tags={}): # TODO: modify to support arbirary data sources

        # lakefs://my-repo/main/data/train.parquet?ref=<commit_hash>
        # lakefs://test-data1/1f6ae9e4be72033c419f73de27427720676b5cf8208d03e34ea5c1d1a7ddc348
        repo, branch, path= self._strip_source(source).split("/", 2)

        ref = additional_tags.get("lakefs_commit", f"{branch}/{path}")

        lfs_source = LakeFSDatasetSource(f"lakefs://{repo}/{ref}") # TODO: not actually usign this source

        ds = mlflow.data.from_pandas(df, source=lfs_source, targets="y_true")

        lineage_tags = {
            "lakefs_repo": repo,
            "lakefs_branch": branch,
            "data_path": path,
            **additional_tags,
        }

        mlflow.log_input(ds, context=context, tags=lineage_tags)
        mlflow.set_tags(lineage_tags)


    def get_model(self, model_uri, device="cpu"):
        model = mlflow.pytorch.load_model(model_uri, map_location=torch.device(device))
        return model
    
    def attach_model_to_run(self, model_uri):
        raise NotImplementedError
        mlflow.set_active_model()

    def infer_signature(self, input: np.ndarray|torch.Tensor, output: np.ndarray|torch.Tensor):
        input = input.numpy(force=True) if isinstance(input, torch.Tensor) else input
        output = output.numpy(force=True) if isinstance(output, torch.Tensor) else output
        #print(type(input))
        return infer_signature(input, output)


    def log_checkpoint( self, checkpoint, training_config, artifact_path="checkpoints", run_id=None, ):
        """Serialize and log a training checkpoint to MLflow."""

        run_id = ( run_id if run_id is not None else mlflow.active_run().info.run_id )
        # Include training configuration in the checkpoint so it can be
        # reconstructed when resuming training.
        checkpoint = {
            **checkpoint,
            "training_config": training_config.model_dump(mode="json"),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join( tmpdir, "checkpoint.pt", )
            torch.save(checkpoint, checkpoint_path)
            mlflow.log_artifact( checkpoint_path, artifact_path=artifact_path, )

        return f"runs:/{run_id}/{artifact_path}/checkpoint.pt"

    
    def load_artifact(self, artifact_uri, map_location="cpu"):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=artifact_uri,
                dst_path=tmpdir,
            )

            return torch.load(
                local_path,
                map_location=torch.device(map_location),
                weights_only=True,
            )


    def load_checkpoint_from_model_uri(self, model_uri, device):
        model_info = mlflow.models.get_model_info(model_uri)

        checkpoint_uri = f"runs:/{model_info.run_id}/checkpoints/checkpoint.pt"

        return self.load_artifact(checkpoint_uri, device)


    def get_run_metrics(self, run_id) -> dict:
        return self.client.get_run(run_id).data.to_dictionary()


    def update_model_tags(self, model_uri: str, tags: dict) -> None:
        if not model_uri.startswith("models:/"):
            return

        path = model_uri[len("models:/"):]

        try:
            # models:/<name>/<version>
            if "/" in path:
                name, version = path.rsplit("/", 1)
                self.client.get_model_version(name, version)  # verify it exists

            # models:/m-<model_id>
            elif path.startswith("m-"):
                model_info = mlflow.models.get_model_info(model_uri)

                if model_info.registered_model_version is None:
                    return

                version = str(model_info.registered_model_version)

                # Find the registered model containing this version.
                name = None
                for registered_model in self.client.search_registered_models():
                    try:
                        mv = self.client.get_model_version(
                            registered_model.name,
                            version,
                        )
                        if mv.model_id == path:
                            name = registered_model.name
                            break
                    except mlflow.exceptions.MlflowException:
                        pass

                if name is None:
                    return

            else:
                return

            for key, value in tags.items():
                self.client.set_model_version_tag(
                    name=name,
                    version=version,
                    key=key,
                    value=str(value),
                )

        except mlflow.exceptions.MlflowException:
            return


if __name__ == "__main__":
    mls = MLflowService()
    mls.update_model_tags(
        "models:/simple_resnet_exp2/2",
        {"validation_ok": True}
    )