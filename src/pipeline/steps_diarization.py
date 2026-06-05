from __future__ import annotations

import logging
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
        backend: DiarizationBackend,
        config: DiarizationConfig,
        *,
        work_dir: Path | None = None,
        ffmpeg_executable: str = "ffmpeg",
        num_speakers_override: int | None = None,
    ) -> None:
        self._backend = backend
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

        context.document = diarize_document(
            document=context.document,
            audio_path=context.audio_path,
            backend=self._backend,
            config=self._config,
            logger=logger,
            work_dir=self._work_dir,
            ffmpeg_executable=self._ffmpeg_executable,
            num_speakers_override=self._num_speakers_override,
        )
        return context
