import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT.parent


def get_repository_root(configured: str | None = None) -> Path:
    """Return the checkout root configured by ``ML_PLATFORM_DATA_DIR``.

    The supported local-development contract is that this directory contains
    ``ml-platform/`` and ``configs/``.
    """
    configured = configured if configured is not None else os.environ.get(
        "ML_PLATFORM_DATA_DIR"
    )

    if configured:
        return Path(configured).expanduser().resolve()

    if os.environ.get("ML_PLATFORM_RUNTIME") == "worker":
        # Worker images contain the application package but not the local
        # checkout. Their service clients receive configuration through the
        # environment, so use the packaged source directory for local paths.
        return SOURCE_ROOT

    raise RuntimeError(
        "ML_PLATFORM_DATA_DIR must be set to the absolute repository root "
        "containing ml-platform/ and configs/"
    )


REPOSITORY_ROOT = get_repository_root()
PLATFORM_ROOT = (
    SOURCE_ROOT
    if os.environ.get("ML_PLATFORM_RUNTIME") == "worker"
    else REPOSITORY_ROOT / "ml-platform"
)

# Compatibility alias for existing callers.  New code should use the more
# specific constants above instead of treating one root as source and data.
PROJECT_ROOT = REPOSITORY_ROOT

DATA_DIR = REPOSITORY_ROOT / "data"
CONFIG_DIR = REPOSITORY_ROOT / "configs"
ENV_FILE = PLATFORM_ROOT / ".env"
LOG_DIR = REPOSITORY_ROOT / "outputs/logs"
OUT_DIR = REPOSITORY_ROOT / "outputs"
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"

DATA_FILE_PATH = DATA_DIR / "LSWMD.pkl"
DOWNLOAD_SCRIPT_PATH = SCRIPTS_DIR / "download_wafermap.sh"
