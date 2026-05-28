import logging
import os
import shutil

from .errors import ProcessingError


def ensure_directories(paths: list[str], logger: logging.Logger) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logger.debug("Ensured directory exists: %s", path)


def ensure_ffmpeg_available(ffmpeg_executable: str, logger: logging.Logger) -> None:
    if shutil.which(ffmpeg_executable) is None:
        raise ProcessingError(
            f"'{ffmpeg_executable}' was not found on PATH. Whisper relies on ffmpeg to decode audio. "
            "Install ffmpeg and restart your terminal/IDE. "
            "On Windows with Chocolatey: choco install ffmpeg. "
            "Or download from https://ffmpeg.org/download.html and add its 'bin' folder to PATH."
        )
    logger.info("Dependency check passed: %s is available.", ffmpeg_executable)
