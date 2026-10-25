"""Small runtime helpers for Experiment 2.

These helpers deliberately operate on the existing experiment state and API
payload shapes; they do not define a new experiment result format.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Callable


TERMINAL_EVENT_STATUSES = {"complete", "failed"}


def wait_for_terminal_event(
    get_event: Callable[[str], dict],
    event_id: str,
    *,
    timeout_seconds: float = 3600,
    poll_interval_seconds: float = 5,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Poll the existing event API until it reports a terminal status."""
    deadline = monotonic() + timeout_seconds
    while True:
        event = get_event(event_id)
        status = event.get("status")
        if status in TERMINAL_EVENT_STATUSES:
            return event
        if monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout_seconds} seconds waiting for event {event_id}"
            )
        sleep(poll_interval_seconds)


def write_state(path: Path, state: dict) -> None:
    """Atomically replace the existing state JSON without changing its schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(state, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
