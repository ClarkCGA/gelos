import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

logger.info(
    "GELOS paths loaded: "
    f"raw={RAW_DATA_DIR}, processed={PROCESSED_DATA_DIR}, external={EXTERNAL_DATA_DIR}"
)
