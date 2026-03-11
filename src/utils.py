import argparse
import logging
import os
import shutil
import sys
import threading
import time

import torch
import yaml
from tqdm import tqdm


class ProcessingError(RuntimeError):
    """Raised when processing cannot continue safely."""


def parse_cli_args(
    videos_path: str,
    audios_path: str,
    cleaned_suffix: str,
    transcript_extension: str,
):
    parser = argparse.ArgumentParser(description="Transcribe videos or audios using local Whisper")
    parser.add_argument(
        "--type",
        choices=["video", "audio"],
        required=True,
        help=(
            f"Input type: 'video' to read from {videos_path}/ and extract audio; "
            f"'audio' to read from {audios_path}/"
        ),
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help=(
            "Run optional cleanup via local Ollama and save "
            f"{cleaned_suffix}{transcript_extension}"
        ),
    )
    return parser.parse_args()


def load_yaml_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ProcessingError(f"Expected YAML mapping in '{path}', got {type(data).__name__}.")
    return data


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


def select_device(logger: logging.Logger) -> str:
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("CUDA is available. Using GPU: %s", gpu_name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("CUDA initialization warning: %s", exc)
        return "cuda"

    diagnostics: list[str] = []
    try:
        if hasattr(torch.backends, "cuda") and not torch.backends.cuda.is_built():
            diagnostics.append("Current PyTorch build lacks CUDA support.")
    except Exception as exc:  # pragma: no cover - defensive
        diagnostics.append(f"Could not query torch.backends.cuda: {exc}")

    cuda_version = getattr(torch.version, "cuda", None)
    if not cuda_version:
        diagnostics.append(
            "torch.version.cuda is None. Install GPU-enabled wheels, "
            "for example: pip install --upgrade torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu121"
        )
    else:
        diagnostics.append(f"PyTorch reports CUDA runtime {cuda_version}.")

    try:
        gpu_count = torch.cuda.device_count()
        diagnostics.append(f"Detected CUDA devices: {gpu_count}")
        if gpu_count == 0:
            diagnostics.append(
                "No CUDA-capable GPUs detected. Ensure NVIDIA drivers are installed and `nvidia-smi` works."
            )
    except Exception as exc:  # pragma: no cover - defensive
        diagnostics.append(f"Could not enumerate CUDA devices: {exc}")

    logger.warning("CUDA not available, defaulting to CPU.")
    for line in diagnostics:
        logger.warning("Diagnostics: %s", line)
    return "cpu"


def get_audio_duration_seconds(audio_path: str) -> float | None:
    try:
        from moviepy import AudioFileClip  # type: ignore

        with AudioFileClip(audio_path) as clip:
            return float(clip.duration) if clip.duration else None
    except Exception:
        try:
            from moviepy.editor import AudioFileClip as EditorAudioFileClip  # type: ignore

            with EditorAudioFileClip(audio_path) as clip:
                return float(clip.duration) if clip.duration else None
        except Exception:
            return None


def transcribe_with_progress(
    model_obj,
    audio_path: str,
    language: str,
    description: str,
    progress_update_interval_seconds: float,
    logger: logging.Logger,
):
    total_seconds = get_audio_duration_seconds(audio_path)
    stop_event = threading.Event()
    pbar: tqdm | None = None

    def _run_progress_bar() -> None:
        nonlocal pbar
        if total_seconds is None or total_seconds <= 0:
            return
        pbar = tqdm(total=int(total_seconds), desc=description, unit="s")
        start_time = time.time()
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            current = int(min(elapsed, pbar.total))
            if current != pbar.n:
                pbar.n = current
                pbar.refresh()
            time.sleep(progress_update_interval_seconds)
        pbar.n = pbar.total
        pbar.refresh()
        pbar.close()

    thread: threading.Thread | None = None
    if total_seconds and total_seconds > 0:
        thread = threading.Thread(target=_run_progress_bar, daemon=True)
        thread.start()
    else:
        logger.debug("Could not estimate media duration for progress bar: %s", audio_path)

    try:
        return model_obj.transcribe(audio_path, language=language, fp16=False)
    finally:
        stop_event.set()
        if thread:
            thread.join(timeout=0.5)


def handle_error(context: str, exc: Exception, logger: logging.Logger) -> None:
    logger.exception("%s: %s", context, exc)
