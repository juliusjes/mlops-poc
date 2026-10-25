from collections import defaultdict
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from ml_platform.workers.trainer.training_config import ColumnTypes, DatasetDescription, DataSource, ModelLoadKwargs, ModelSourceType, Sampling

class ValidationType(str, Enum):
    STANDALONE = "standalone"
    SHADOW = "shadow"
    AB_TEST = "ab_test"



class ValidationRule(BaseModel):
    rule_name: str | None = None
    data_source: str
    n_samples: int = Field(default=-1) # -1 = all
    sampling: Sampling = Field(default=Sampling.UNIFORM)
    sampling_seed: int = Field(default=0)
    metric: str

    # standalone
    max_value: float | None = None
    min_value: float | None = None
    max_p_value: float | None = None

    # ab
    model_b: str | None = None # uri
    min_relative_change: float | None = None
    max_relative_change: float | None = None

    max_absolute_change: float | None = None
    min_absolute_change: float | None = None

    @model_validator(mode="after")
    def set_rule_name(self):
        if self.rule_name is None:
            self.rule_name = self.metric
        return self

    @field_validator("n_samples")
    @classmethod
    def validate_n_samples(cls, v: int) -> int:
        if v == -1 or v > 0:
            return v
        raise ValueError("must be -1 or a positive integer")

class Validation(BaseModel):
    validation_type: ValidationType
    rules: list[ValidationRule]

    @model_validator(mode="after")
    def ensure_unique_rule_name(self):
        counts = defaultdict(int)

        for r in self.rules:
            name = r.rule_name or "rule"

            if counts[name] == 0:
                r.rule_name = name
            else:
                r.rule_name = f"{name}_{counts[name]}"

            counts[name] += 1

        return self
    
    @model_validator(mode="after")
    def ensure_fields(self):
        for rule in self.rules:
            match self.validation_type:
                case ValidationType.STANDALONE:
                    if rule.max_value is None or rule.min_value is None:
                        raise ValueError( "STANDALONE validation requires max_value and min_value" )

                case ValidationType.AB_TEST:
                    if rule.model_b is None:
                        raise ValueError( "AB_TEST validation requires model_b" )

                    has_relative = rule.max_relative_change is not None or rule.min_relative_change is not None

                    has_absolute = rule.max_absolute_change is not None or rule.min_absolute_change is not None

                    if not (has_relative or has_absolute):
                        raise ValueError( "AB_TEST requires relative or absolute change limits" )
                    
                    if has_absolute and has_relative:
                        raise ValueError("AB_TEST takes only absolute or relative change per validation rule")

                case ValidationType.SHADOW:
                    raise NotImplementedError

        return self


class ValidationConfig(BaseModel):
    validations: list[Validation]
    #validator_name: str = Field(default="BasicValidator")
    model_source: ModelSourceType
    model_load_kwargs: ModelLoadKwargs
    device: Literal["cpu", "cuda"] = Field(default="cpu")

    dataset_implementation: DataSource = Field(default=DataSource.LAKEFSSOURCE)
    dataset_description: DatasetDescription = Field(
        default_factory=lambda: DatasetDescription(
            sample_column=["X", ColumnTypes.BYTES],
            target_column=["y", ColumnTypes.INT],
            order_column=["write_timestamp", ColumnTypes.DATETIME]
            )
        )