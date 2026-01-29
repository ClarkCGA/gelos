import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()


def _get_required_path(var_name: str) -> Path:
    value = os.getenv(var_name)
    if not value:
        raise ValueError(
            "Missing required path environment variables: "
            "EXTERNAL_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT"
        )
    return Path(value).expanduser().resolve()


EXTERNAL_DATA_DIR = _get_required_path("EXTERNAL_DATA_DIR")
RAW_DATA_DIR = _get_required_path("RAW_DATA_DIR")
INTERIM_DATA_DIR = _get_required_path("INTERIM_DATA_DIR")
PROCESSED_DATA_DIR = _get_required_path("PROCESSED_DATA_DIR")
PROJECT_ROOT = _get_required_path("PROJECT_ROOT")

CONFIG_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

logger.info(
    "GELOS paths loaded: "
    f"raw={RAW_DATA_DIR}, processed={PROCESSED_DATA_DIR}, external={EXTERNAL_DATA_DIR}"
)
