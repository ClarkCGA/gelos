from pathlib import Path
import os
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
# This loads GELOS_BUCKET and path configuration for GELOS
_env_file = os.getenv("GELOS_ENV_FILE")
if _env_file:
    load_dotenv(_env_file, override=False)
else:
    load_dotenv(override=False)

def _parse_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _env_path(var_name: str) -> Path | None:
    return _parse_path(os.getenv(var_name))


def _derive_child(base: Path | None, child: str) -> Path | None:
    if base is None:
        return None
    return base / child


def _require_paths(required: dict[str, Path | None]) -> None:
    missing = [name for name, value in required.items() if value is None]
    if missing:
        missing_vars = ", ".join(missing)
        raise ValueError(
            "Missing required GELOS path environment variables: "
            f"{missing_vars}. Set GELOS_DATA_ROOT or each specific path variable."
        )


# Paths (environment-authoritative)
PROJ_ROOT = _env_path("GELOS_PROJECT_ROOT")
DATA_DIR = _env_path("GELOS_DATA_ROOT")

RAW_DATA_DIR = _env_path("GELOS_RAW_DIR") or _derive_child(DATA_DIR, "raw")
INTERIM_DATA_DIR = _env_path("GELOS_INTERIM_DIR") or _derive_child(DATA_DIR, "interim")
PROCESSED_DATA_DIR = _env_path("GELOS_PROCESSED_DIR") or _derive_child(
    DATA_DIR, "processed"
)
EXTERNAL_DATA_DIR = _env_path("GELOS_EXTERNAL_DIR") or _derive_child(
    DATA_DIR, "external"
)

MODELS_DIR = _env_path("GELOS_MODELS_DIR") or _derive_child(PROJ_ROOT, "models")
REPORTS_DIR = _env_path("GELOS_REPORTS_DIR") or _derive_child(PROJ_ROOT, "reports")
FIGURES_DIR = _env_path("GELOS_FIGURES_DIR") or _derive_child(
    REPORTS_DIR, "figures"
)
CONFIG_DIR = _env_path("GELOS_CONFIG_DIR") or _derive_child(PROJ_ROOT, "gelos/configs")

DATA_VERSION = os.getenv("GELOS_DATA_VERSION", "v0.50.0")

_require_paths(
    {
        "GELOS_RAW_DIR": RAW_DATA_DIR,
        "GELOS_PROCESSED_DIR": PROCESSED_DATA_DIR,
        "GELOS_EXTERNAL_DIR": EXTERNAL_DATA_DIR,
    }
)

logger.info(
    "GELOS paths loaded: "
    f"raw={RAW_DATA_DIR}, processed={PROCESSED_DATA_DIR}, external={EXTERNAL_DATA_DIR}"
)

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
