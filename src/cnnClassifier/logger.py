import logging
import sys
from pathlib import Path


LOG_FILE_PATH = Path("logs/running_logs.log")
LOG_FORMAT = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"


def get_logger(name: str = "cnnClassifier") -> logging.Logger:
    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


logger = get_logger()
