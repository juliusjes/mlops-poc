from pydantic import BaseModel, model_validator, Field, field_validator
from typing import Any, Literal
from enum import Enum


class ModelSourceType(str, Enum):
    PYTHON = "python"
    MLFLOW = "mlflow"
    PICKLE = "pickle"


class DataSource(str, Enum):
    LAKEFSSOURCE = "lakefs_source"




class ColumnTypes(str, Enum):
    INT = "int"
    FLOAT = "float"
    STR = "str"
    BYTES = "bytes"
    DATE = "date"
    DATETIME = "datetime"
    BOOL = "bool"


class TargetInterpret(str, Enum): # TODO: depricated
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class DatasetDescription(BaseModel):
    sample_column: list[str, ColumnTypes]
    sample_shape: list[int]
    target_column: list[str, ColumnTypes]
    target_shape: list[int]
    order_column: list[str , ColumnTypes] | None
    target_interp: TargetInterpret


class Sampling(str, Enum): # TODO: depricated
    NO_SAMPLING = "no_sampling"
    UNIFORM = "uniform"
    LATEST = "latest"
    EXP_WEIGHT_LATEST = "exp_weight_latest"
    LIN_WEIGHT_LATEST = "lin_weight_latest"


class ModelLoadKwargs(BaseModel):
    model_name: str | None = None
    model_uri: str | None = None
    pickle_file: str | None = None

    # @model_validator(mode="after")
    # def validate_config(self):

    #     if self.model_name is not None:
    #         if self.model_uri or self.pickle_file:
    #             raise ValueError("python_file/class_name cannot be combined")

    #     elif self.model_uri is not None:
    #         if self.pickle_file:
    #             raise ValueError("model_uri and pickle_file are mutually exclusive")

    #     elif self.pickle_file is None:
    #         raise ValueError( "Must provide either " "(python_file + class_name), " "model_uri, or pickle_file" )

    #     return self


class TrainerConfig(BaseModel):
    pass


class TrainingConfig(BaseModel):
    dataset_location: str
    n_training_samples: int = Field(default=-1) # -1 means all

    train_size: float = Field(default=0.8, le=1)
    test_size: float = Field(default=0.2, le=1)

    sampling_seed: int = Field(default=0)


    model_source_type: ModelSourceType
    model_load_kwargs: ModelLoadKwargs

    loop_name: str # <loopname>
    loop_kwargs: dict

    
    dataset_implementation: DataSource = Field(default=DataSource.LAKEFSSOURCE)
    dataset_description: DatasetDescription = Field(
        default_factory=lambda: DatasetDescription(
            sample_column=["X", ColumnTypes.BYTES],
            target_column=["y", ColumnTypes.INT],
            order_column=["write_timestamp", ColumnTypes.DATETIME]
            )
        )
    run_name: str # TODO: remove
    model_register_name: str # TODO: remove?
    device: Literal["cpu", "cuda"] = Field(default="cpu")
    register_device: Literal["cpu", "cuda"] = Field(default="cpu")
    train_loader_kwargs: dict[str, Any] = Field(default_factory=dict)
    test_loader_kwargs: dict[str, Any] = Field(default_factory=dict)
    #train_test_split_seed: int = Field(default=0)

    optimizer: str = Field(default="Adam")
    optimizer_kwargs: dict[str, Any] = Field(default_factory=dict)
    loss_function: str = "CrossEntropyLoss"
    loss_function_kwargs: dict[str, Any] = Field(default_factory=dict)
    epochs: int = Field(default=10, ge=1)
    
    #artifact_path: str = Field(default="artifacts")

    @model_validator(mode="after")
    def validate_config(self):
        if self.train_size + self.test_size != 1:
            raise ValueError("train_size + test_size should sum to 1")
        return self

    @field_validator("n_training_samples")
    @classmethod
    def validate_n_training_samples(cls, v: int) -> int:
        if v == -1 or v > 0:
            return v
        raise ValueError("must be -1 or a positive integer")



















if __name__ == "__main__":
    # mlkcfg = {
    #     "python_file": "/home/mfijulje/Documents/masters-thesis-main/code/src/ml/models.py",
    #     "class_name": "SimpleTestModel"
    # }

    # mlkcfg = ModelLoadKwargs.model_validate(mlkcfg)

    # print(mlkcfg)


    training_config = {
            "dataset_location": "test-data1/expad7d9b2c016e7764/expdata/offline/",
            "model_source_type": "python",
            "model_load_kwargs": {
                "python_file": "/home/mfijulje/Documents/masters-thesis-main/code/src/ml/models.py",
                "class_name": "SimpleTestModel"
                },
            "trainer_config": {
                "train_loader_kwargs": {"batch_size": 8},
                "test_loader_kwargs": {"batch_size": 8},
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss",
            },
            "trainer_name": "BasicTrainer",
            "run_name": "simple-test",
            "model_register_name": "simple_test_nn"
        }

    trcfg = TrainingConfig.model_validate(training_config)
    print(trcfg)