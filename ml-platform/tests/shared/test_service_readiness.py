from types import SimpleNamespace

import pytest
import requests

from ml_platform.shared import utils


def test_get_healthy_endpoint_retries_until_an_endpoint_is_ready(monkeypatch):
    attempts = []

    def get(url, timeout):
        attempts.append((url, timeout))
        if len(attempts) < 3:
            raise requests.ConnectionError("service is starting")
        return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr(utils.requests, "get", get)
    monkeypatch.setattr(utils.time, "sleep", lambda _: None)
    monkeypatch.setenv("SERVICE_READINESS_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("SERVICE_READINESS_POLL_INTERVAL_SECONDS", "0.01")

    assert utils.get_healthy_endpoint(["http://service"], "/health", timeout=0.1) == "http://service"
    assert len(attempts) == 3


def test_get_healthy_endpoint_fails_after_the_configured_readiness_window(monkeypatch):
    clock = iter([0, 0, 1])

    monkeypatch.setattr(utils.requests, "get", lambda *_args, **_kwargs: (_ for _ in ()).throw(requests.ConnectionError()))
    monkeypatch.setattr(utils.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(utils.time, "sleep", lambda _: None)
    monkeypatch.setenv("SERVICE_READINESS_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("SERVICE_READINESS_POLL_INTERVAL_SECONDS", "0.1")

    with pytest.raises(RuntimeError, match="within 1 seconds"):
        utils.get_healthy_endpoint(["http://service"], "/health", timeout=0.1)


def test_get_healthy_endpoint_rejects_invalid_readiness_settings(monkeypatch):
    monkeypatch.setenv("SERVICE_READINESS_TIMEOUT_SECONDS", "0")

    with pytest.raises(ValueError, match="must be positive"):
        utils.get_healthy_endpoint(["http://service"])
