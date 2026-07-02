from __future__ import annotations

import logging
from pathlib import Path

from src.models import PipelineContext, PipelineState
from src.output.formatter import write_srt, write_text_file, write_transcript, write_vtt

from .steps import PipelineStep


class OutputStep(PipelineStep):
    """Write transcript and (optionally) cleaned text to disk."""

    def __init__(
        self,
        transcripts_folder: Path,
        transcript_extension: str,
        cleaned_suffix: str,
        *,
        output_basename: str | None = None,
        write_srt_vtt: bool = False,
    ) -> None:
        self._transcripts_folder = transcripts_folder
        self._extension = transcript_extension
        self._cleaned_suffix = cleaned_suffix
        self._output_basename = output_basename
        self._write_srt_vtt = write_srt_vtt

    def execute(self, context: PipelineContext, logger: logging.Logger) -> PipelineContext:
        if context.document is None:
            context.fail("No document to write")
            return context

        stem = self._output_basename or context.source_path.stem
        transcript_path = self._transcripts_folder / (stem + self._extension)
        write_transcript(context.document, transcript_path)
        logger.info("Transcript saved: %s", transcript_path)

        clean_path: Path | None = None
        if context.cleaned_text:
            clean_path = self._transcripts_folder / (stem + self._cleaned_suffix + self._extension)
            write_text_file(clean_path, context.cleaned_text)
            logger.info("Cleaned transcript saved: %s", clean_path)

        srt_path: Path | None = None
        vtt_path: Path | None = None
        if self._write_srt_vtt and any(
            s.start_time is not None and s.end_time is not None for s in context.document.segments
        ):
            srt_path = self._transcripts_folder / (stem + ".srt")
            vtt_path = self._transcripts_folder / (stem + ".vtt")
            write_srt(context.document, srt_path)
            write_vtt(context.document, vtt_path)
            logger.info("Subtitles saved: %s, %s", srt_path, vtt_path)

        outputs = context.document.metadata.setdefault("outputs", {})
        outputs["txt"] = str(transcript_path)
        outputs["clean"] = str(clean_path) if clean_path else None
        outputs["srt"] = str(srt_path) if srt_path else None
        outputs["vtt"] = str(vtt_path) if vtt_path else None

        context.document.pipeline_state = PipelineState.EXPORTED
        return context
