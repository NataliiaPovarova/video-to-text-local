from __future__ import annotations

import logging
from pathlib import Path

from src.ingestion.video_extractor import extract_audio_from_video
from src.models import PipelineContext, PipelineState, TranscriptDocument
from src.transcription.asr_engine import _get_audio_duration_seconds
from src.utils.errors import MediaDecodeError

from .steps import PipelineStep


def _probe_duration(path: Path) -> float | None:
    """Return decodable duration in seconds, or None if the file can't be read."""
    return _get_audio_duration_seconds(str(path))


class VideoIngestionStep(PipelineStep):
    """Extract audio from a video file."""

    def __init__(self, audios_path: Path, extracted_audio_extension: str) -> None:
        self._audios_path = audios_path
        self._extension = extracted_audio_extension

    def execute(self, context: PipelineContext, logger: logging.Logger) -> PipelineContext:
        audio_output = self._audios_path / (context.source_path.stem + self._extension)
        try:
            success = extract_audio_from_video(context.source_path, audio_output, logger)
        except Exception as exc:  # moviepy/ffmpeg decode failure
            raise MediaDecodeError(
                f"Could not read/decode '{context.source_path.name}': it may be corrupt, "
                "truncated, or use an unsupported codec."
            ) from exc
        if not success:
            raise MediaDecodeError(
                f"Could not read/decode '{context.source_path.name}': it has no audio track."
            )

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
        # Probe decodability up front so corrupt/unsupported audio yields a clear
        # MediaDecodeError instead of an opaque ffmpeg failure mid-transcription.
        if _probe_duration(context.source_path) is None:
            raise MediaDecodeError(
                f"Could not read/decode '{context.source_path.name}': it may be corrupt, "
                "truncated, missing an audio track, or use an unsupported codec."
            )
        context.audio_path = context.source_path
        context.document = TranscriptDocument(
            source_file=str(context.source_path),
            language=context.language,
            pipeline_state=PipelineState.INGESTED,
        )
        logger.info("Audio file ready: %s", context.source_path.name)
        return context
