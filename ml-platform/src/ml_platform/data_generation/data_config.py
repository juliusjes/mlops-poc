import importlib
from inspect import Parameter, signature
from typing import Any, Callable, Optional

#from ml_platform.shared import data_processing as datap
from ml_platform.shared.utils import load_object
from pydantic import BaseModel, Field, model_validator


class PhaseConfiguration(BaseModel):
    data_branch: str
    pipeline: list[str]
    initial_arg: Any
    metadata_select: Optional[list[str]] = None
    pipeline_kwargs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    final_arg: Any

    @model_validator(mode="after")
    def validate_pipeline_parts(self):
        for name in self.pipeline:
            try:
                mod, nam = name.split(".")

                module = importlib.import_module(
                    f"ml_platform.shared.data_processing.{mod}"
                )

                func = getattr(module, nam)
            except ImportError as e:
                raise ValueError(f"Pipeline piece {name!r} (mod: {mod}, nam:{nam}) cannot be imported") from e

            if not callable(func):
                raise ValueError(f"Pipeline piece {name!r} is not callable")

            kwargs = self.pipeline_kwargs.get(name)
            if kwargs is None:
                raise ValueError(
                    f"Pipeline piece {name!r} has no entry in 'pipeline_kwargs'"
                )

            sig = signature(func)

            # Required keyword/positional arguments after the first input argument
            required = [
                param.name
                for i, param in enumerate(sig.parameters.values())
                if i > 0
                and param.default is Parameter.empty
                and param.kind
                in (
                    Parameter.POSITIONAL_OR_KEYWORD,
                    Parameter.KEYWORD_ONLY,
                )
            ]

            missing = [arg for arg in required if arg not in kwargs]
            if missing:
                raise ValueError(
                    f"Pipeline piece {name!r} is missing required arguments: {missing}"
                )

            # Optional: reject unexpected kwargs unless **kwargs is accepted
            accepts_kwargs = any(
                p.kind is Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )

            if not accepts_kwargs:
                allowed = {
                    p.name
                    for p in sig.parameters.values()
                    if p.kind
                    in (
                        Parameter.POSITIONAL_OR_KEYWORD,
                        Parameter.KEYWORD_ONLY,
                    )
                }

                unexpected = set(kwargs) - allowed
                if unexpected:
                    raise ValueError(
                        f"Pipeline piece {name!r} received unexpected arguments: "
                        f"{sorted(unexpected)}"
                    )

        return self


class DataConfiguration(BaseModel):
    phases: list[str]
    repository: str

    phase_configs: dict[str, PhaseConfiguration]

    def model_post_init(self, __context):
        missing = set(self.phases) - set(self.phase_configs)
        if missing:
            raise ValueError(f"Missing configuration for phases: {missing}")


if __name__ == "__main__":
    pass
    #print(dir(datap))
    #getattr(datap, "image_processing")
    # import json

    # file = "/home/mfijulje/Documents/masters-thesis-main/code/configs/LWMD_data_config.json"

    # with open(file, "r") as f:
    #     cfg = json.load(f)["data_configuration"]
    # DataConfiguration.model_validate(cfg)
