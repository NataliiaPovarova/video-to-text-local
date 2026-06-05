from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path

from src.utils.errors import ProcessingError


def ensure_wav_16k_mono(
    audio_path: Path,
    work_dir: Path,
    ffmpeg_executable: str = "ffmpeg",
    logger: logging.Logger | None = None,
) -> Path:
    """Convert audio to 16 kHz mono WAV for diarization backends."""
    if audio_path.suffix.lower() == ".wav":
        return audio_path

    work_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(str(audio_path.resolve()).encode()).hexdigest()[:12]
    output_path = work_dir / f"{audio_path.stem}_{digest}_16k.wav"

    if output_path.exists() and output_path.stat().st_mtime >= audio_path.stat().st_mtime:
        return output_path

    command = [
        ffmpeg_executable,
        "-y",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    if logger:
        logger.debug("Converting audio for diarization: %s", " ".join(command))

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ProcessingError(
            f"ffmpeg not found ('{ffmpeg_executable}'). Install ffmpeg to run diarization."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ProcessingError(
            f"ffmpeg failed to convert audio for diarization: {exc.stderr or exc}"
        ) from exc

    return output_path
