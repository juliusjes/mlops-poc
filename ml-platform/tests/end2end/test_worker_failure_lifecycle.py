"""Opt-in validation of worker failure propagation through the orchestrator."""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pytest


RUN_LIVE_E2E = os.environ.get("RUN_LIVE_E2E") == "1"
pytestmark = [
    pytest.mark.live_e2e,
    pytest.mark.skipif(
        not RUN_LIVE_E2E,
        reason="requires a running Compose stack and writes temporary LakeFS/event state; set RUN_LIVE_E2E=1",
    ),
]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLATFORM_ROOT = PROJECT_ROOT / "ml-platform"
ORCHESTRATOR_URL = "http://localhost:8014"
DATA_CONFIG = PLATFORM_ROOT / "tests/shared/simple_data_config.json"


def request_json(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        f"{ORCHESTRATOR_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urlopen(request, timeout=30) as response:
        return response.status, json.load(response)


def generate_simple_dataset():
    environment = os.environ | {"ML_PLATFORM_DATA_DIR": str(PROJECT_ROOT)}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_platform.data_generation.generate_datasets",
            "--config",
            str(DATA_CONFIG),
        ],
        cwd=PLATFORM_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r"\[exp\]: exp stage done, artifact in (\S+)", result.stdout)
    assert match, result.stdout
    return f"{match.group(1)}0/offline/"


def lifecycle_payload(
    dataset_location,
    loop_name,
    deploy_criteria="no_deployment",
    minimum_accuracy=0.0,
):
    dataset_description = {
        "sample_column": ["X", "bytes"],
        "sample_shape": [8],
        "target_column": ["y", "int"],
        "target_shape": [1],
        "order_column": ["write_timestamp", "datetime"],
        "target_interp": "classification",
    }
    return {
        "training_config": {
            "dataset_location": dataset_location,
            "n_training_samples": 100,
            "model_source_type": "python",
            "model_load_kwargs": {"model_name": "simple_test_model"},
            "loop_name": loop_name,
            "loop_kwargs": {},
            "train_loader_kwargs": {"batch_size": 32},
            "test_loader_kwargs": {"batch_size": 32},
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "epochs": 1,
            "device": "cpu",
            "register_device": "cpu",
            "run_name": "p0-lifecycle-failure-check",
            "model_register_name": "p0_lifecycle_unused",
            "dataset_implementation": "lakefs_source",
            "dataset_description": dataset_description,
        },
        "validation_config": {
            "model_source": "mlflow",
            "model_load_kwargs": {},
            "device": "cpu",
            "dataset_implementation": "lakefs_source",
            "dataset_description": dataset_description,
            "validations": [
                {
                    "validation_type": "standalone",
                    "rules": [
                        {
                            "rule_name": "unused",
                            "data_source": dataset_location,
                            "n_samples": 10,
                            "metric": "accuracy_score",
                            "max_value": 1.0,
                            "min_value": minimum_accuracy,
                        }
                    ],
                }
            ],
        },
        "deploy_criteria": deploy_criteria,
    }


def wait_for_terminal_event(event_id):
    deadline = time.monotonic() + 20 * 60
    while time.monotonic() < deadline:
        _, event = request_json(f"/events/{event_id}")
        if event["status"] in {"failed", "complete"}:
            return event
        time.sleep(5)
    pytest.fail("worker event did not reach a terminal state within 20 minutes")


def test_failed_worker_event_is_persisted_as_failed():
    status, health = request_json("/health")
    assert status == 200
    assert health == {"status": "healthy"}

    dataset_location = generate_simple_dataset()
    status, queued = request_json(
        "/events/manual-start",
        lifecycle_payload(dataset_location, "intentionally_missing_loop"),
    )
    assert status == 202

    event = wait_for_terminal_event(queued["event_id"])

    assert event["status"] == "failed", event
    assert event["completed"] is not None
    assert event["error"]

    _, history = request_json(f"/events/{queued['event_id']}/history")
    assert any(state["new_state"] == "training" for state in history["history"])


def test_cpu_training_validation_completes_without_deployment():
    dataset_location = generate_simple_dataset()
    status, queued = request_json(
        "/events/manual-start",
        lifecycle_payload(dataset_location, "pytorch_classification_loop"),
    )
    assert status == 202

    event = wait_for_terminal_event(queued["event_id"])
    assert event["status"] == "complete", event
    assert event["error"] is None
    assert event["result"]["deployed"] is False

    _, history = request_json(f"/events/{queued['event_id']}/history")
    states = {(state["workflow"], state["new_state"]) for state in history["history"]}
    assert ("Training-workflow", "training") in states
    assert ("Training-workflow", "complete") in states
    assert ("Validation-workflow", "validating") in states
    assert ("Validation-workflow", "complete") in states


def test_failed_validation_does_not_deploy_when_criteria_is_validated():
    dataset_location = generate_simple_dataset()
    status, queued = request_json(
        "/events/manual-start",
        lifecycle_payload(
            dataset_location,
            "pytorch_classification_loop",
            deploy_criteria="validated",
            minimum_accuracy=1.1,
        ),
    )
    assert status == 202

    event = wait_for_terminal_event(queued["event_id"])
    assert event["status"] == "complete", event
    assert event["error"] is None
    assert event["result"]["passed"] is False
    assert event["result"]["deployed"] is False
