import hashlib
import json
import os
import subprocess
import time
from collections.abc import Sequence
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, model_validator

from ml_platform.shared.paths import LOG_DIR


def date_string(time, tf=0):
    """
    time formats:
    - 0: regular, [%H:%M:%S %d-%m-%Y]
    - 1: short, [%H:%M:%S]
    - 2: compressed, [%H%M%S %d%m%Y]
    """
    if not time:
        ds = ""
    elif tf == 0:
        ds = datetime.now().strftime(" [%H:%M:%S %d-%m-%Y]")
    elif tf == 1:
        ds = datetime.now().strftime(" [%H:%M:%S]")
    elif tf == 2:
        ds = datetime.now().strftime(" [%H%M%S %d%m%Y]")
    else:
        raise f"Unsupported time format {tf}"
    return ds


def info(message="", prefix="", time=True, tf=0, return_string=False):
    msg = f"[INFO]{date_string(time, tf=tf)} {prefix}: {message}"
    if return_string:
        return msg
    else:
        print(msg)


def warning(message, prefix="", time=True, tf=0):
    print(f"[WARNING]{date_string(time, tf=tf)} {prefix}: {message}")


def analyze_wafer_df(df):
    # --- helper ---
    def extract_label(x):
        try:
            return x[0][0]
        except Exception:
            return None

    df = df.copy()

    # --- ensure clean columns exist ---
    if "failure_label" not in df.columns:
        df["failure_label"] = df["failureType"].apply(extract_label)

    if "split_label" not in df.columns:
        df["split_label"] = df["trianTestLabel"].apply(extract_label)

    print("=== BASIC INFO ===")
    total_rows = len(df)
    print(f"Total rows: {total_rows}")

    # --- split stats ---
    print("\n=== TRAIN / TEST DISTRIBUTION ===")
    split_counts = df["split_label"].value_counts(dropna=False)
    print(split_counts)
    print(f"Unique split labels: {df['split_label'].nunique()}")

    # --- failure stats ---
    print("\n=== FAILURE TYPE DISTRIBUTION ===")
    failure_counts = df["failure_label"].value_counts(dropna=False)
    print(failure_counts)
    print(f"Unique failure types: {df['failure_label'].nunique()}")

    # --- filtered useful view (training + real failures) ---
    df_train_valid = df[
        (df["split_label"] == "Training")
        & (df["failure_label"].notna())
        & (df["failure_label"] != "none")
    ]

    print("\n=== TRAINING (EXCLUDING 'none') ===")
    print(f"Rows: {len(df_train_valid)}")

    print("\nFailure distribution (training only):")
    print(df_train_valid["failure_label"].value_counts())

    # --- cross-tab ---
    print("\n=== FAILURE TYPE vs SPLIT ===")
    print(pd.crosstab(df["failure_label"], df["split_label"]))

    # --- wafer map info ---
    print("\n=== WAFER MAP SHAPES ===")
    shapes = df["waferMap"].apply(lambda x: x.shape if hasattr(x, "shape") else None)
    print(shapes.value_counts().head())

    # --- die size ---
    print("\n=== DIE SIZE STATS ===")
    print(df["dieSize"].describe())

    # --- lot distribution ---
    print("\n=== LOT DISTRIBUTION (top 10) ===")
    print(df["lotName"].value_counts().head(10))

    # --- missing ---
    print("\n=== MISSING VALUES ===")
    print(df.isna().sum())

    # --- class imbalance (important) ---
    print("\n=== CLASS IMBALANCE (%) ===")
    imbalance = df_train_valid["failure_label"].value_counts(normalize=True)
    print((imbalance * 100).round(2))

    print("\n=== DONE ===")


def make_hash(o):
    if isinstance(o, dict):
        rep = tuple((k, make_hash(v)) for k, v in sorted(o.items()))
    elif isinstance(o, (list, tuple)):
        rep = tuple(make_hash(v) for v in o)
    elif isinstance(o, set):
        rep = tuple(sorted(make_hash(v) for v in o))
    else:
        rep = repr(o)

    return hashlib.sha256(repr(rep).encode()).hexdigest()[:16]


def count_nonzero_in_arrays(arrays):
    nan_counts = np.array([np.count_nonzero(np.isnan(a)) for a in arrays])

    print(f"Avg number of NaNs per array: {nan_counts.mean()}")
    print(f"Arrays containing NaNs: {np.flatnonzero(nan_counts)}")


def init_log_path(prefix, file_ext):
    log_path = Path(LOG_DIR) / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.{file_ext}"
    return log_path


def np_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_object(path: str, package="data_processing"):
    module_name, object_name = path.rsplit(".", 1)
    module = import_module(f"{package}.{module_name}")
    return getattr(module, object_name)




def get_healthy_endpoint(
    endpoints: Sequence[str],
    health_path: str | None = "/health",
    command: list[str] | None = None,
    timeout: float = 2.0,
) -> str:
    """
    Return the first healthy endpoint from `endpoints`, in priority order.

    If `health_path` is provided, performs an HTTP GET against
    `{endpoint}{health_path}`.

    If `command` is provided instead, runs the command with the endpoint
    appended as the final argument.

    Args:
        endpoints: Endpoint URLs to try, in priority order.
        health_path: Path used for the HTTP health check.
        command: Command used for a non-HTTP health check. The endpoint
            is appended as the final argument.
        timeout: Timeout in seconds for each health check.

    The total readiness window is controlled by
    ``SERVICE_READINESS_TIMEOUT_SECONDS`` (default: 30 seconds). Failed
    probes are retried at ``SERVICE_READINESS_POLL_INTERVAL_SECONDS``
    (default: 1 second). Keeping the per-probe timeout separate prevents a
    dependency that is still starting from failing its consumer immediately.

    Raises:
        RuntimeError: If none of the endpoints are healthy.
    """
    if not health_path and not command:
        raise RuntimeError("No valid healthcheck available")

    readiness_timeout = float(os.environ.get("SERVICE_READINESS_TIMEOUT_SECONDS", "30"))
    poll_interval = float(os.environ.get("SERVICE_READINESS_POLL_INTERVAL_SECONDS", "1"))
    if readiness_timeout <= 0 or poll_interval <= 0:
        raise ValueError("Service readiness timeout and poll interval must be positive")

    deadline = time.monotonic() + readiness_timeout
    attempted_endpoints = [endpoint.rstrip("/") for endpoint in endpoints if endpoint]
    if not attempted_endpoints:
        raise RuntimeError("No endpoints were provided for health checking")

    if health_path:
        health_path = "/" + health_path.lstrip("/")

    while True:
        for endpoint in attempted_endpoints:
            try:
                if health_path:
                    response = requests.get(f"{endpoint}{health_path}", timeout=timeout)
                    response.raise_for_status()
                    return endpoint

                result = subprocess.run(
                    [*command, endpoint], capture_output=True, text=True, timeout=timeout
                )
                if result.returncode == 0:
                    return endpoint
            except (requests.RequestException, subprocess.TimeoutExpired, FileNotFoundError):
                continue

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_interval, remaining))

    raise RuntimeError(
        f"No healthy endpoint found among: {list(endpoints)} "
        f"within {readiness_timeout:g} seconds"
    )


from io import BytesIO


def array_to_bytes(x: np.ndarray) -> bytes:
    np_bytes = BytesIO()
    np.save(np_bytes, x, allow_pickle=True)
    return np_bytes.getvalue()


def bytes_to_array(b: bytes) -> np.ndarray:
    return np.load(BytesIO(b), allow_pickle=True)
