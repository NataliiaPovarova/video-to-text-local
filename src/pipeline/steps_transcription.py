from __future__ import annotations

import logging

from src.models import PipelineContext
from src.transcription.asr_engine import transcribe_audio

from .steps import PipelineStep


class TranscriptionStep(PipelineStep):
    """Run Whisper ASR on the audio file."""

    def __init__(self, whisper_model, progress_update_interval: float) -> None:
        self._model = whisper_model
        self._interval = progress_update_interval

    def execute(self, context: PipelineContext, logger: logging.Logger) -> PipelineContext:
        if context.audio_path is None:
            context.fail("No audio path set before transcription step")
            return context

        document = transcribe_audio(
            model=self._model,
            audio_path=context.audio_path,
            language=context.language,
            progress_update_interval_seconds=self._interval,
            logger=logger,
        )
        document.source_file = str(context.source_path)
        context.document = document
        return context
