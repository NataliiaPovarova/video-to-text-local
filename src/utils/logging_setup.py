import logging
import os
import sys


def setup_logging(logs_dir: str, level: str, file_name: str, log_format: str) -> logging.Logger:
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger("video_to_text")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(log_format)
    file_handler = logging.FileHandler(os.path.join(logs_dir, file_name), encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger
