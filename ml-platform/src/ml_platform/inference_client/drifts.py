import importlib
import logging
from typing import Callable

import numpy as np
import pandas as pd
from ml_platform.shared.data_processing import augmentation_funcs as aug_module
from ml_platform.shared.utils import info, warning
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class DriftFunc:
    """
    Base class and non transforming drift function
    """

    def __call__(self, X, y, test=False):
        if test:
            logger.info("Test call")

        return self._transform(X, y)

    def _transform(self, X, y) -> tuple[pd.Series, pd.Series, np.ndarray]:
        nx = len(X) if not isinstance(X, np.ndarray) else X.shape[0]
        t = self._build_parameter(nx)
        return X, y, t

    def _build_parameter(self, nx):
        return np.zeros(nx)


class CP2DFSettings(BaseModel):
    seed: int = 0,
    class_ratio: float = Field(default=0.5, gt=0.0, lt=1.0)
    augmentation_func: str | Callable
    augmentation_kwargs: dict = {}
    parameter_name: str
    target_replacement: int|float|str = 1
    replace_all_labels: bool = True
    param_values: tuple[int|float, int|float]
    params_cps: tuple[float, float]

    @field_validator("params_cps", mode="after")
    @classmethod
    def validate_cps(cls, value: tuple[float, float]) -> tuple[float, float]:
        if not (0 <= value[0] < value[1] <= 1):
            raise ValueError("params_cps must satisfy 0 <= cps0 < cps1 <= 1")
        return value

class CP2DriftFunc(DriftFunc):
    """
    piecewise linear drift function with two control points for binary targets using an augmentation function.
    Drift only affects samples from one class if the class balance is the same in the input data and therefore simulates concept drift.

    By defining class_ratio to be over the original class ratio and defining replace_all_labels to be False the original labels are kept
    but remaining input samples are affected by the augmentation function
    """

    def __init__(
        self,
        seed=0,
        class_ratio=0.5,
        augmentation_func: str | Callable = None,
        augmentation_kwargs: dict = {},
        parameter_name="",
        target_replacement=1,
        replace_all_labels=True,
        param_values=[],
        param_cps=[],
    ):
        super().__init__()

        assert len(param_values) == 2 and isinstance(param_values, list)
        assert len(param_cps) == 2 and isinstance(param_cps, list)

        assert (0 < param_cps[0] < 1) and (0 < param_cps[1] < 1)
        assert parameter_name not in augmentation_kwargs.keys()

        self.param_start_value = param_values[0]
        self.param_end_value = param_values[1]
        self.ramp_start = param_cps[0]
        self.ramp_end = param_cps[1]

        self.seed = seed
        self.min_samples = 5
        self.class_ratio = class_ratio

        if isinstance(augmentation_func, str):
            aug_cls = getattr(aug_module, augmentation_func)
            self.augmentation_func = aug_cls # actually a function
            
        elif isinstance(augmentation_func, Callable):
            self.augmentation_func = augmentation_func

        self.augmentation_kwargs = augmentation_kwargs
        logger.info(f"Augmentation function: {self.augmentation_func} with kwargs: {self.augmentation_kwargs}")

        self.parameter_name = parameter_name
        self.target_replacement = target_replacement
        self.replace_all_labels = replace_all_labels

        self.param = None  # generated

    def _transform(self, X, y) -> tuple[pd.Series, pd.Series, np.ndarray]:
        """
        Apply transformation to a fraction of samples according to the built parameter function.
        """
        if isinstance(X, np.ndarray):
            X = pd.Series([sample for sample in X])
        elif isinstance(X, list):
            X = pd.Series(X)

        if isinstance(y, (list, np.ndarray)):
            y = pd.Series(y)

        nx = len(X)
        ny = len(y)

        assert nx == ny, f"X and y have different number of samples {nx}, {ny}"

        self.param = pd.Series(self.get_parameter(nx), index=X.index)

        X_out = X.copy()
        y_out = y.copy()

        # Desired number of target_replacement samples
        N = max(1, round(nx * self.class_ratio))

        # Samples already having the target label
        existing_idx = y[y == self.target_replacement].index

        if len(existing_idx) >= N:
            # Already have enough; randomly choose N of them
            logger.info(
                f"Applying augmentation to {N} samples out of {len(y[y == self.target_replacement])} of class {self.target_replacement}"
            )
            selected_idx = (
                existing_idx.to_series()
                .sample(n=N, random_state=self.seed)
                .sort_values()
                .index
            )
        else:
            selected_idx = existing_idx  # Keep all existing positives
            logger.warning(
                f"Applying augmentation to samples from class(es) {np.unique(y[y != self.target_replacement])} instead of {self.target_replacement}, "
                f"{'not ' if self.replace_all_labels else ''}keeping original labels"
            )
            remaining_idx = y.index.difference(
                existing_idx
            )  # Sample additional indices from the remaining samples

            extra_n = N - len(existing_idx)
            extra_idx = (
                remaining_idx.to_series()
                .sample(n=extra_n, random_state=self.seed)
                .sort_values()
                .index
            )
            selected_idx = selected_idx.union(extra_idx).sort_values()

        # Apply augmentation
        for idx in selected_idx:
            param = self.param.loc[idx]
            kwargs = {**self.augmentation_kwargs, self.parameter_name: param}

            X_out.loc[idx] = self.augmentation_func( X.loc[idx], **kwargs, )

            # Only relabel newly selected samples
            if y.loc[idx] != self.target_replacement and self.replace_all_labels:
                y_out.loc[idx] = self.target_replacement

        return X_out, y_out, self.param

    def _build_parameter(self, nx) -> np.ndarray:
        """
        Build a gradual ramp of parameter values from self.param_start_value to self.param_end_value with length nx.
        The ramp starts at self.ramp_start and ends at self.ramp_end
        """
        if nx < self.min_samples:
            logger.warning(f"Drift not applied for sample size {nx}")
            return np.repeat(self.param_start_value, nx)

        param = np.zeros((nx))
        ramp_start_idx = int(self.ramp_start * nx)
        ramp_end_idx = int(self.ramp_end * nx)
        param[0:ramp_start_idx] = np.repeat(self.param_start_value, ramp_start_idx)
        param[ramp_start_idx:ramp_end_idx] = np.linspace(
            self.param_start_value, self.param_end_value, ramp_end_idx - ramp_start_idx
        )
        param[ramp_end_idx:nx] = np.repeat(self.param_end_value, nx - ramp_end_idx)
        return param

    def get_parameter(self, nx) -> np.ndarray:
        """
        Get the parameter array, build one if it doesnt exist yet
        """
        if self.param is None:
            self.param = self._build_parameter(nx)
        elif len(self.param) != nx or len(self.param) < self.min_samples:
            self.param = self._build_parameter(nx)

        return self.param



from collections.abc import Callable

from pydantic import BaseModel, Field, field_validator


class CPNDFSettings(BaseModel):
    seed: int = 0
    class_ratio: float | list[float]

    augmentation_func: str | Callable | list[str | Callable]
    augmentation_kwargs: dict | list[dict] = Field(default_factory=dict)
    parameter_names: str | list[str]
    target_replacement: int | float | str | list[int | float | str] = 1
    replace_all_labels: bool | list[bool] = True
    param_values: list[int | float] | list[list[int | float]]
    params_cps: list[float] | list[list[float]]

    @field_validator("params_cps")
    @classmethod
    def validate_cps( cls, value: list[float] | list[list[float]], ) -> list[float] | list[list[float]]:
        def validate_sequence(cps: list[float]) -> None:
            if not all(0 <= cp <= 1 for cp in cps):
                raise ValueError("All params_cps values must satisfy 0 <= cp <= 1")

            if any(a >= b for a, b in zip(cps, cps[1:])):
                raise ValueError("params_cps values must be strictly increasing")

        if isinstance(value[0], (int, float)):
            validate_sequence(value)
        else:
            for cps in value:
                validate_sequence(cps)

        return value

class CPNDriftFunc(DriftFunc):
    """
    Drift with N change points.
    """
    def __init__(
            self,
            seed=0,
            class_ratio: list[float] | float = 0.5,
            augmentation_func: str | Callable | list[str | Callable] = None,
            augmentation_kwargs: list[dict] | dict = {},
            parameter_names: list[str] | str="",
            target_replacement=1,
            replace_all_labels:list[bool] | bool=True,
            param_values:list[list[float]] | list[float]=[],
            param_cps: list[list[float]] | list[float]=[]
            ):
        
        super().__init__()
        self.param_values = param_values if isinstance(param_values[0], list) else [param_values]
        self.param_cps = sorted(param_cps) if isinstance(param_cps[0], list) else [param_cps]

        self.param = None
        self.min_samples = 10
        self.seed = seed
        self.class_ratio = class_ratio if isinstance(class_ratio, list) else [class_ratio]

        self.augmentation_funcs = []
        if isinstance(augmentation_func, (str, Callable)): augmentation_func = [augmentation_func]
        for af in augmentation_func:
            if isinstance(af, str):
                aug_cls = getattr(aug_module, af)
                self.augmentation_funcs.append(aug_cls) # actually a function
                
            elif isinstance(af, Callable):
                self.augmentation_funcs.append(af)

        self.augmentation_kwargs = augmentation_kwargs if isinstance(augmentation_kwargs, list) else [augmentation_kwargs]

        logger.info(f"Augmentation functions: {self.augmentation_funcs} with kwargs: {self.augmentation_kwargs}")

        self.parameter_names = parameter_names if isinstance(parameter_names, list) else [parameter_names]
        self.replace_all_labels = replace_all_labels if isinstance(replace_all_labels, list) else [replace_all_labels]
        self.target_replacement = target_replacement

        self.param = None # generated

    def _transform(self, X, y) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Apply transformation to a fraction of samples according to the built parameter function.
        """
        if isinstance(X, np.ndarray):
            X = pd.Series([sample for sample in X])
        elif isinstance(X, list):
            X = pd.Series(X)

        if isinstance(y, (list, np.ndarray)):
            y = pd.Series(y)

        assert X.index.equals(y.index), "X and y indices do not match"
        assert X.index.is_unique, "X index contains duplicates"
        assert y.index.is_unique, "y index contains duplicates"

        nx = len(X)
        ny = len(y)

        assert nx == ny, f"X and y have different number of samples {nx}, {ny}"

        params = []

        X_out = X.copy()
        y_out = y.copy()

        # Desired number of target_replacement samples
        for i, af in enumerate(self.augmentation_funcs):
            logger.info(f"Augmenting with {af.__name__}")

            self.param = self.get_parameter(nx, X.index, i)

            N = max(1, round(nx * self.class_ratio[i]))

            # Indices of samples that already have the target label
            existing_idx = y.index[y == self.target_replacement]

            if len(existing_idx) >= N:
                # We already have enough target samples; randomly select N of them
                logger.info(
                    f"Applying augmentation to {N} samples out of "
                    f"{len(existing_idx)} samples of class {self.target_replacement}"
                )

                selected_idx = existing_idx.to_series().sample(
                    n=N,
                    random_state=self.seed,
                ).index

            else:
                # Keep all existing target samples
                selected_idx = existing_idx

                logger.warning(
                    f"Applying augmentation to samples from class(es) "
                    f"{np.unique(y[y != self.target_replacement])} instead of "
                    f"{self.target_replacement}, "
                    f"{'not ' if self.replace_all_labels[i] else ''}"
                    f"keeping original labels"
                )

                # Select additional samples from the non-target samples
                remaining_idx = y.index.difference(existing_idx)
                extra_n = N - len(existing_idx)

                extra_idx = remaining_idx.to_series().sample(
                    n=extra_n,
                    random_state=self.seed,
                ).index

                selected_idx = existing_idx.union(extra_idx)

            # Keep processing order deterministic
            selected_idx = selected_idx.sort_values()

            for idx in selected_idx:
                param = self.param.loc[idx]

                kwargs = {**self.augmentation_kwargs[i], self.parameter_names[i]: param}

                if "rng" in kwargs:
                    kwargs["rng"] = np.random.default_rng(int(idx))

                X_out.loc[idx] = af(X.loc[idx], **kwargs)
                # Only relabel newly selected samples
                if y.loc[idx] != self.target_replacement and self.replace_all_labels[i]:
                    y_out.loc[idx] = self.target_replacement

            y = y_out.copy()
            X = X_out.copy()
            params.append(self.param)

        return X_out, y_out, pd.Series(params)

    def _build_parameter(self, nx, af_index) -> np.ndarray:
        if nx < self.min_samples:
            logger.warning(f"Drift not applied for sample size {nx}")
            return np.repeat(self.param_values[0], nx)

        param = np.zeros(nx)

        cps = self.param_cps[af_index]
        vls = self.param_values[af_index]
        param[:int(np.ceil(cps[0] * nx))] = vls[0]
        for i in range(1, len(vls)):
            start = int(np.ceil(cps[i-1] * nx))
            end = int(np.ceil(cps[i] * nx))

            if end > start:
                x = np.arange(start, end)
                param[start:end] = (vls[i] - vls[i-1]) / (end - start) * (x - start) + vls[i-1]

        start = int(np.ceil(cps[-1] * nx))
        param[start:] = vls[-1]

        return param

    
    def get_parameter(self, nx, index, af_index):
        self.param = pd.Series(
            self._build_parameter(nx, af_index),
            index=index,
        )
        return self.param


class SineDriftFunc(CPNDriftFunc):
    """
    Piecewise sine approximation
    """
    def __init__(self, seed=0, class_ratio=0.5, augmentation_func = None, augmentation_kwargs = {}, parameter_name="", target_replacement=1, replace_all_labels=True, frequency=1, amplitude=1, phase=np.pi, offset=1):
        param_cps = np.linspace(0.001, 0.999, 100)
        param_values = amplitude * np.sin(frequency*2*np.pi*param_cps + phase) + offset
        super().__init__(seed, class_ratio, augmentation_func, augmentation_kwargs, parameter_name, target_replacement, replace_all_labels, param_values, param_cps)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # N = 1000
    # D = 2

    def f(X, offset=0, constant=1):
        return constant * X + offset*5

    # X = np.random.normal(size=(N, D))  # np.ones((N, D))
    # y = np.zeros(N)


    X = np.random.uniform(size=(100, 2))
    y = np.random.choice([0, 1], size=(100))

    df = CPNDriftFunc(
        augmentation_func=["f"],
        augmentation_kwargs=[
        {
        "constant": [1]
        }
        ],
        parameter_names=["offset"],
        param_values=[0, 1],
        param_cps=[0.0, 1.0],
        class_ratio=1.0,
        replace_all_labels=[False]
    )

    x_out, y_out, param = df._transform(X, y)
    param = param[0]
    x_out = [xo[0] for xo in x_out]

    for i, offset in enumerate(np.linspace(0, 1, X.shape[0], endpoint=False)):
        X[i] = f(X[i], offset = offset) 


    plt.figure()
    plt.plot(X[:, 0], label="expected")
    plt.plot(x_out, color="red", label="actual")
    plt.legend()

    plt.figure()
    plt.plot(param, label="actual")
    plt.plot(np.arange(0, len(x_out)), np.linspace(0, 1, len(x_out), endpoint=False), label="expected")

    plt.legend()

    plt.show()

    # import matplotlib.pyplot as plt
    # from paths import OUT_DIR

    # plt.figure(figsize=(7, 6))
    # plt.scatter(
    #     Xt[yt == 0, 0],
    #     Xt[yt == 0, 1],
    #     c=np.arange(np.count_nonzero(yt == 0)),
    #     cmap="plasma",
    #     marker="x",
    #     s=20,
    # )
    # sc = plt.scatter(
    #     Xt[yt == 1, 0],
    #     Xt[yt == 1, 1],
    #     c=np.arange(np.count_nonzero(yt == 1)),
    #     cmap="plasma",
    #     s=20,
    # )
    # plt.colorbar(sc, label="Sample index")
    # plt.xlabel("Feature 1")
    # plt.ylabel("Feature 2")
    # plt.title("Gradual Drift")
    # plt.tight_layout()
    # plt.savefig(OUT_DIR / "xt.png")

    # plt.figure()
    # plt.plot(np.arange(len(ft)), ft)
    # plt.savefig(OUT_DIR / "ft.png")
