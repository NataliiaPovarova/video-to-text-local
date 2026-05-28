from __future__ import annotations

import logging
from pathlib import Path

from src.ingestion.video_extractor import extract_audio_from_video
from src.models import PipelineContext, PipelineState, TranscriptDocument

from .steps import PipelineStep


class VideoIngestionStep(PipelineStep):
    """Extract audio from a video file."""

    def __init__(self, audios_path: Path, extracted_audio_extension: str) -> None:
        self._audios_path = audios_path
        self._extension = extracted_audio_extension

    def execute(self, context: PipelineContext, logger: logging.Logger) -> PipelineContext:
        audio_output = self._audios_path / (context.source_path.stem + self._extension)
        success = extract_audio_from_video(context.source_path, audio_output, logger)
        if not success:
            context.fail(f"No audio track in {context.source_path.name}")
            return context

        context.audio_path = audio_output
        context.document = TranscriptDocument(
            source_file=str(context.source_path),
            language=context.language,
            pipeline_state=PipelineState.INGESTED,
        )
        return context


class AudioIngestionStep(PipelineStep):
    """Prepare an audio file for transcription (no extraction needed)."""

    def execute(self, context: PipelineContext, logger: logging.Logger) -> PipelineContext:
        context.audio_path = context.source_path
        context.document = TranscriptDocument(
            source_file=str(context.source_path),
            language=context.language,
            pipeline_state=PipelineState.INGESTED,
        )
        logger.info("Audio file ready: %s", context.source_path.name)
        return context
