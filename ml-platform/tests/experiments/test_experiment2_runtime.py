import importlib.util
from pathlib import Path

import pytest


RUNTIME_PATH = (
    Path(__file__).resolve().parents[3] / "experiments/experiment2/runtime.py"
)
SPEC = importlib.util.spec_from_file_location("experiment2_runtime", RUNTIME_PATH)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def test_wait_for_terminal_event_returns_existing_complete_payload():
    responses = iter([{"status": "started"}, {"status": "complete", "result": {}}])

    result = runtime.wait_for_terminal_event(
        lambda _: next(responses),
        "event-id",
        poll_interval_seconds=0,
    )

    assert result == {"status": "complete", "result": {}}


def test_wait_for_terminal_event_returns_existing_failed_payload():
    result = runtime.wait_for_terminal_event(
        lambda _: {"status": "failed", "error": "worker failed"},
        "event-id",
        poll_interval_seconds=0,
    )

    assert result["status"] == "failed"
    assert result["error"] == "worker failed"


def test_wait_for_terminal_event_has_a_bounded_timeout():
    clock = iter([0, 0, 2])

    with pytest.raises(TimeoutError, match="event-id"):
        runtime.wait_for_terminal_event(
            lambda _: {"status": "started"},
            "event-id",
            timeout_seconds=1,
            poll_interval_seconds=0,
            monotonic=lambda: next(clock),
        )


def test_write_state_preserves_existing_json_data_shape(tmp_path):
    state = {
        "completed": {"subset": [0], "validation": {"metric": 0.5}},
        "waiting": {"subset": [1], "regime": ["gradual"]},
    }
    destination = tmp_path / "state.json"

    runtime.write_state(destination, state)

    assert destination.exists()
    assert destination.read_text(encoding="utf-8")
    import json

    assert json.loads(destination.read_text(encoding="utf-8")) == state
