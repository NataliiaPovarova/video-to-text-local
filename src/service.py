from __future__ import annotations

import functools
import gc
import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import whisper

from src.models import PipelineContext, TranscriptDocument
from src.pipeline import (
    AudioIngestionStep,
    CleanupStep,
    DiarizationStep,
    OutputStep,
    PipelineOrchestrator,
    TranscriptionStep,
    VideoIngestionStep,
)
from src.processing import cleanup_with_ollama
from src.transcription.diarization_config import load_diarization_config
from src.transcription.diarizer import create_diarization_backend
from src.utils import (
    MediaDecodeError,
    ensure_ffmpeg_available,
    ensure_ffmpeg_on_path,
    load_yaml_file,
    select_device,
)
from src.utils.naming import build_output_basename, sanitize_stem

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_AUDIO_EXTS = {".mp3", ".m4a"}
_device_cache: str | None = None

_MODEL_LOCK = threading.Lock()
_resident_model: tuple[str, object] | None = None  # (name, model)
_diarization_backend: tuple[str, str, object] | None = None  # (backend, model_id, obj)


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def get_whisper_model(name: str, device: str, logger: logging.Logger | None = None):
    """Return a resident Whisper model, holding at most one in memory.

    Reuses the resident model when ``name`` matches; otherwise evicts the
    previous model (drop ref + gc + empty CUDA cache) before loading the new
    one. Jobs run one at a time, so only one model is ever needed at once.
    """
    global _resident_model
    log = logger or logging.getLogger("video_to_text")
    with _MODEL_LOCK:
        if _resident_model is not None and _resident_model[0] == name:
            return _resident_model[1]
        if _resident_model is not None:
            log.info("Evicting Whisper model '%s' to load '%s'", _resident_model[0], name)
            _resident_model = None
            gc.collect()
            _empty_cuda_cache()
        log.info("Loading Whisper model: %s", name)
        model = whisper.load_model(name, device=device)
        _resident_model = (name, model)
        log.info("Whisper model loaded: %s", name)
        return model


def reset_caches() -> None:
    """Drop the resident model and diarization backend (tests/teardown)."""
    global _resident_model, _diarization_backend, _device_cache
    with _MODEL_LOCK:
        _resident_model = None
        _diarization_backend = None
        _device_cache = None
        gc.collect()
        _empty_cuda_cache()


@dataclass
class TranscribeResult:
    document: TranscriptDocument | None
    txt_path: str | None
    clean_path: str | None
    srt_path: str | None
    vtt_path: str | None
    model: str
    language: str | None
    options: dict
    status: str  # "success" | "warning" | "failed"
    warnings: list[str] = field(default_factory=list)
    message: str | None = None


def _resolve_device(logger: logging.Logger) -> str:
    global _device_cache
    with _MODEL_LOCK:
        if _device_cache is None:
            _device_cache = select_device(logger)
        return _device_cache


def get_diarization_backend(config, device: str, logger: logging.Logger):
    """Return a cached diarization backend, keyed by (backend, model)."""
    global _diarization_backend
    with _MODEL_LOCK:
        key = (config.backend, config.model)
        if _diarization_backend is not None and _diarization_backend[:2] == key:
            return _diarization_backend[2]
        backend = create_diarization_backend(config, device, logger)
        _diarization_backend = (config.backend, config.model, backend)
        return backend


def transcribe_file(
    source_path,
    *,
    model: str,
    language: str | None = None,
    cleanup: bool = False,
    diarize: bool = False,
    num_speakers: int | None = None,
    write_srt_vtt: bool = False,
    sanitize_output: bool = False,
    config_path: str = "configurations/general_config.yaml",
    logger: logging.Logger | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> TranscribeResult:
    """Build and run the existing step-pipeline for ONE file with per-job params.

    `params.yaml` is never mutated: `model`/`language`/`cleanup`/`diarize` are
    arguments. The same function backs both the CLI and the GUI.

    When ``write_srt_vtt=True`` and the transcript contains timestamped segments,
    SRT and VTT subtitle files are written alongside the plain-text transcript and
    their paths are returned in ``srt_path``/``vtt_path`` on the result.
    """
    source_path = Path(source_path)
    logger = logger or logging.getLogger("video_to_text")
    cfg = load_yaml_file(config_path)
    paths = cfg["paths"]
    output = cfg["output"]
    ollama = cfg["ollama"]
    processing = cfg["processing"]
    dependencies = cfg["dependencies"]

    # ffmpeg pre-flight (the shared CLI+GUI seam). Whisper's load_audio shells
    # out to a bare `ffmpeg`; without this the GUI died deep in the pipeline
    # with FileNotFoundError. Provision the bundled imageio-ffmpeg binary if no
    # system ffmpeg exists, then verify -- raising a clear, actionable
    # ProcessingError (surfaced by app.py) when ffmpeg is genuinely unavailable
    # instead of letting transcription fail mid-run.
    ensure_ffmpeg_on_path(logger)
    ensure_ffmpeg_available(dependencies["ffmpeg_executable"], logger)

    transcripts_dir = Path(paths["transcripts"])
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    audios_dir = Path(paths["audios"])
    audios_dir.mkdir(parents=True, exist_ok=True)
    ext = source_path.suffix.lower()
    transcript_extension = output["transcript_extension"]

    options = {"cleanup": cleanup, "diarize": diarize, "num_speakers": num_speakers}

    # --- classify input ---
    if ext in _VIDEO_EXTS:
        input_type = "video"
        ingestion_step = VideoIngestionStep(audios_dir, output["extracted_audio_extension"])
    elif ext in _AUDIO_EXTS:
        input_type = "audio"
        ingestion_step = AudioIngestionStep()
    else:
        raise MediaDecodeError(
            f"Unsupported file type '{ext or source_path.name}'. "
            f"Supported: {', '.join(sorted(_VIDEO_EXTS | _AUDIO_EXTS))}."
        )

    # --- output basename (GUI: sanitized + collision-safe; CLI: plain stem) ---
    job_id = uuid.uuid4().hex
    if sanitize_output:
        stem = sanitize_stem(source_path.stem, fallback=job_id[:8])
        basename = build_output_basename(
            stem,
            model,
            exists=lambda b: (transcripts_dir / (b + transcript_extension)).exists(),
            job_id=job_id,
        )
    else:
        basename = None

    device = _resolve_device(logger)
    whisper_model = get_whisper_model(model, device, logger)

    # --- assemble steps ---
    transcription_step = TranscriptionStep(
        whisper_model=whisper_model,
        progress_update_interval=processing["progress_update_interval_seconds"],
        progress_callback=progress_callback,
    )
    steps = [ingestion_step, transcription_step]

    if diarize:
        # Backend setup eagerly loads the pyannote pipeline (model download / HF
        # auth), which can fail before the pipeline ever runs. Diarization is an
        # optional enhancement, so degrade gracefully: on any setup failure log
        # a warning and skip the step -- the transcript is still produced, and
        # the outcome-classification below flags the missing speaker labels.
        try:
            diar_cfg = load_diarization_config(cfg["files"]["diarization"])
            diar_cfg.enabled = True
            backend = get_diarization_backend(diar_cfg, device, logger)
        except Exception as exc:  # noqa: BLE001 -- optional feature, degrade gracefully
            logger.warning(
                "Diarization unavailable (%s); transcribing without speaker labels.",
                exc,
            )
        else:
            steps.append(
                DiarizationStep(
                    backend=backend,
                    config=diar_cfg,
                    work_dir=audios_dir / ".diarization_cache",
                    ffmpeg_executable=dependencies["ffmpeg_executable"],
                    num_speakers_override=num_speakers,
                )
            )

    if cleanup:
        params = load_yaml_file(cfg["files"]["params"])  # read-only
        prompts = load_yaml_file(cfg["files"]["prompts"])
        cleanup_func = functools.partial(
            cleanup_with_ollama,
            cleanup_model_name=params["cleanup_model"],
            cleanup_prompt=prompts["cleanup_prompt"],
            device=device,
            ollama_url=ollama["url"],
            ollama_timeout_seconds=ollama["timeout_seconds"],
            ollama_request_content_type=ollama["request_content_type"],
            logger=logger,
        )
        steps.append(CleanupStep(cleanup_func))

    steps.append(
        OutputStep(
            transcripts_folder=transcripts_dir,
            transcript_extension=transcript_extension,
            cleaned_suffix=output["cleaned_suffix"],
            output_basename=basename,
            write_srt_vtt=write_srt_vtt,
        )
    )

    # --- run ---
    context = PipelineContext(source_path=source_path, input_type=input_type, language=language or "")
    pipeline = PipelineOrchestrator(steps=steps, logger=logger)
    context = pipeline.run(context)

    # --- classify outcome ---
    if context.exception is not None:
        if isinstance(context.exception, MediaDecodeError):
            raise context.exception
        return TranscribeResult(
            document=context.document, txt_path=None, clean_path=None, srt_path=None, vtt_path=None,
            model=model, language=language, options=options, status="failed",
            message=str(context.exception),
        )

    doc = context.document
    outputs = (doc.metadata.get("outputs") if doc else None) or {}
    warnings: list[str] = []
    status = "success"
    # Diarization was requested but produced no result -- either backend setup
    # failed (step skipped above) or diarize_document degraded inside the step.
    # Key on metadata["diarization"], which is written ONLY on full diarization
    # success and survives OutputStep (which clobbers pipeline_state to EXPORTED
    # regardless); its absence is the durable "diarization didn't happen"
    # signal and covers both failure paths. The raw transcript was still
    # written, so surface a warning instead of failing.
    if diarize and not (doc and doc.metadata.get("diarization")):
        warnings.append("Diarization unavailable; transcript saved without speaker labels.")
        status = "warning"
    # Cleanup was requested but produced nothing (e.g. Ollama down) -> warning.
    if cleanup and not outputs.get("clean"):
        warnings.append("Cleanup unavailable; raw transcript saved.")
        status = "warning"

    return TranscribeResult(
        document=doc,
        txt_path=outputs.get("txt"),
        clean_path=outputs.get("clean"),
        srt_path=outputs.get("srt"),
        vtt_path=outputs.get("vtt"),
        model=model,
        language=(doc.language if doc else language),
        options=options,
        status=status,
        warnings=warnings,
        message="; ".join(warnings) or None,
    )
