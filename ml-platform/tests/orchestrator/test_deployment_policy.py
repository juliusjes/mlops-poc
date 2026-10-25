import pytest

from ml_platform.orchestrator.deployment_policy import should_deploy


@pytest.mark.parametrize(
    ("criteria", "passed", "expected"),
    [
        ("none", True, True),
        ("none", False, True),
        ("validated", True, True),
        ("validated", False, False),
        ("no_deployment", True, False),
        ("no_deployment", False, False),
    ],
)
def test_should_deploy(criteria, passed, expected):
    assert should_deploy(criteria, {"passed": passed}) is expected


def test_should_deploy_rejects_unknown_criteria():
    with pytest.raises(ValueError, match="Unsupported deploy criterion"):
        should_deploy("unexpected", {"passed": True})
