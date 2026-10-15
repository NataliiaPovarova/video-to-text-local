from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from src.models import PipelineContext, PipelineState
from src.transcription.diarization_backends.base import DiarizationBackend
from src.transcription.diarization_config import DiarizationConfig
from src.transcription.diarizer import diarize_document

from .steps import PipelineStep


class DiarizationStep(PipelineStep):
    """Run speaker diarization and align speakers to transcript segments."""

    def __init__(
        self,
        backend: DiarizationBackend | None = None,
        config: DiarizationConfig | None = None,
        *,
        backend_loader: Callable[[], DiarizationBackend] | None = None,
        work_dir: Path | None = None,
        ffmpeg_executable: str = "ffmpeg",
        num_speakers_override: int | None = None,
    ) -> None:
        # ``backend_loader`` defers loading (pyannote, GPU-resident) until the
        # step runs, i.e. after transcription -- see issue #27.
        if (backend is None) == (backend_loader is None):
            raise ValueError("Pass exactly one of backend or backend_loader")
        if config is None:
            raise ValueError("config is required")
        self._backend = backend
        self._backend_loader = backend_loader
        self._config = config
        self._work_dir = work_dir
        self._ffmpeg_executable = ffmpeg_executable
        self._num_speakers_override = num_speakers_override

    def should_skip(self, context: PipelineContext) -> bool:
        return super().should_skip(context) or not self._config.enabled

    def execute(self, context: PipelineContext, logger: logging.Logger) -> PipelineContext:
        if context.audio_path is None:
            context.fail("No audio path set before diarization step")
            return context
        if context.document is None:
            context.fail("No document available for diarization")
            return context
        if context.document.pipeline_state != PipelineState.TRANSCRIBED:
            logger.warning(
                "Diarization expected TRANSCRIBED state, got %s",
                context.document.pipeline_state,
            )

        # Diarization is an optional enhancement on top of an already-complete
        # transcript. Mirror CleanupStep: if it fails, keep the raw transcript
        # (diarize_document mutates the document only on success, so it stays
        # pristine here), record a warning, and let the pipeline continue to
        # OutputStep -- never abort the whole job. Catch broadly because the
        # pyannote/HF backend failure surface is wide (auth, download, CUDA,
        # audio decode, model format) -- lazy backend loading included.
        try:
            backend = self._backend if self._backend is not None else self._backend_loader()
            context.document = diarize_document(
                document=context.document,
                audio_path=context.audio_path,
                backend=backend,
                config=self._config,
                logger=logger,
                work_dir=self._work_dir,
                ffmpeg_executable=self._ffmpeg_executable,
                num_speakers_override=self._num_speakers_override,
            )
        except Exception as exc:  # noqa: BLE001 -- optional feature, degrade gracefully
            logger.warning(
                "Diarization unavailable for %s: %s; keeping raw transcript.",
                context.source_path.name,
                exc,
            )
            context.document.metadata.setdefault("warnings", []).append(
                f"diarization unavailable: {exc}"
            )
        return context
