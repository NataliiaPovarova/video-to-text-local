from __future__ import annotations

import logging
from pathlib import Path

from src.models import PipelineContext, PipelineState, TranscriptDocument
from src.utils.errors import MediaDecodeError
from src.utils.ffmpeg_tools import ExtractionError, MediaInfo, classify_media, extract_audio, probe_media

from .steps import PipelineStep


def _require_audio(context: PipelineContext, ffmpeg: str, logger: logging.Logger) -> MediaInfo:
    """Return the source's probe, raising MediaDecodeError if it has no usable audio.

    The seam probes once and puts the result on ``context.media``; steps built
    without the seam (tests, future callers) probe here instead.
    """
    if context.media is None:
        context.media = probe_media(context.source_path, ffmpeg)
    logger.info("Media probe for %s: %s", context.source_path.name, context.media)
    classify_media(context.media, context.source_path.name)
    return context.media


class VideoIngestionStep(PipelineStep):
    """Extract the audio track of a video into ``audios_path``."""

    def __init__(self, audios_path: Path, extracted_audio_extension: str, ffmpeg_executable: str = "ffmpeg") -> None:
        self._audios_path = audios_path
        self._extension = extracted_audio_extension
        self._ffmpeg = ffmpeg_executable

    def execute(self, context: PipelineContext, logger: logging.Logger) -> PipelineContext:
        _require_audio(context, self._ffmpeg, logger)
        audio_output = self._audios_path / (context.source_path.stem + self._extension)
        logger.info("Extracting audio from %s", context.source_path.name)
        try:
            extract_audio(context.source_path, audio_output, self._ffmpeg)
        except ExtractionError as exc:
            raise MediaDecodeError(f"Could not extract audio from '{context.source_path.name}'.") from exc
        logger.info("Audio extracted: %s", audio_output)

        context.audio_path = audio_output
        context.document = TranscriptDocument(
            source_file=str(context.source_path),
            language=context.language,
            pipeline_state=PipelineState.INGESTED,
        )
        return context


class AudioIngestionStep(PipelineStep):
    """Prepare an audio file for transcription (no extraction needed)."""

    def __init__(self, ffmpeg_executable: str = "ffmpeg") -> None:
        self._ffmpeg = ffmpeg_executable

    def execute(self, context: PipelineContext, logger: logging.Logger) -> PipelineContext:
        # Fail up front with a clear MediaDecodeError instead of an opaque
        # ffmpeg failure mid-transcription.
        _require_audio(context, self._ffmpeg, logger)
        context.audio_path = context.source_path
        context.document = TranscriptDocument(
            source_file=str(context.source_path),
            language=context.language,
            pipeline_state=PipelineState.INGESTED,
        )
        logger.info("Audio file ready: %s", context.source_path.name)
        return context
