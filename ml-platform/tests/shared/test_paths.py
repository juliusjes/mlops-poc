
import os
import subprocess
import sys

from ml_platform.shared.paths import (
    CONFIG_DIR,
    DATA_DIR,
    ENV_FILE,
    LOG_DIR,
    OUT_DIR,
    PACKAGE_ROOT,
    PLATFORM_ROOT,
    PROJECT_ROOT,
    REPOSITORY_ROOT,
    SOURCE_ROOT,
    get_repository_root,
)

class TestPaths:
    def test_is_path(self):
        paths = [PROJECT_ROOT, LOG_DIR, DATA_DIR, OUT_DIR, CONFIG_DIR]

        for p in paths:
            assert os.path.isdir(p), f"{p} should be a valid path"

        assert os.path.isfile(ENV_FILE), f".env file should be in {PROJECT_ROOT}"

    def test_repository_and_source_roots_are_distinct(self):
        assert PROJECT_ROOT == REPOSITORY_ROOT
        assert PLATFORM_ROOT == REPOSITORY_ROOT / "ml-platform"
        assert PACKAGE_ROOT == SOURCE_ROOT / "ml_platform"
        assert (PACKAGE_ROOT / "workers/trainer/Dockerfile").is_file()

    def test_explicit_repository_root_is_resolved(self, tmp_path):
        configured = tmp_path / "checkout"
        assert get_repository_root(str(configured)) == configured.resolve()

    def test_missing_repository_root_fails_clearly(self):
        environment = os.environ.copy()
        environment.pop("ML_PLATFORM_DATA_DIR", None)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from ml_platform.shared.paths import REPOSITORY_ROOT",
            ],
            env=environment,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "ML_PLATFORM_DATA_DIR must be set" in result.stderr

    def test_worker_runtime_uses_packaged_source_root(self):
        environment = os.environ.copy()
        environment.pop("ML_PLATFORM_DATA_DIR", None)
        environment["ML_PLATFORM_RUNTIME"] = "worker"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from ml_platform.shared.paths import "
                    "PLATFORM_ROOT, REPOSITORY_ROOT; "
                    "print(REPOSITORY_ROOT); print(PLATFORM_ROOT)"
                ),
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )

        repository_root, platform_root = result.stdout.splitlines()
        assert repository_root == str(SOURCE_ROOT)
        assert platform_root == str(SOURCE_ROOT)
