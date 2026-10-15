from __future__ import annotations

import logging

from src.models import PipelineContext
from src.transcription.asr_engine import transcribe_audio

from .steps import PipelineStep


class TranscriptionStep(PipelineStep):
    """Run Whisper ASR on the audio file."""

    def __init__(
        self,
        whisper_model=None,
        progress_update_interval: float = 1.0,
        *,
        progress_callback=None,
        model_loader=None,
    ) -> None:
        # ``model_loader`` (a zero-arg callable) lets the step hold no model
        # reference between runs, so the service can evict it from VRAM.
        if (whisper_model is None) == (model_loader is None):
            raise ValueError("Pass exactly one of whisper_model or model_loader")
        self._model = whisper_model
        self._model_loader = model_loader
        self._interval = progress_update_interval
        self._progress_callback = progress_callback

    def execute(self, context: PipelineContext, logger: logging.Logger) -> PipelineContext:
        if context.audio_path is None:
            context.fail("No audio path set before transcription step")
            return context

        document = transcribe_audio(
            model=self._model if self._model is not None else self._model_loader(),
            audio_path=context.audio_path,
            language=context.language,
            progress_update_interval_seconds=self._interval,
            logger=logger,
            progress_callback=self._progress_callback,
        )
        document.source_file = str(context.source_path)
        context.document = document
        return context
