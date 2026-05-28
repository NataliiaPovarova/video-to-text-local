from __future__ import annotations

import logging
from pathlib import Path

from src.models import PipelineContext, PipelineState
from src.output.formatter import write_text_file, write_transcript

from .steps import PipelineStep


class OutputStep(PipelineStep):
    """Write transcript and (optionally) cleaned text to disk."""

    def __init__(
        self,
        transcripts_folder: Path,
        transcript_extension: str,
        cleaned_suffix: str,
    ) -> None:
        self._transcripts_folder = transcripts_folder
        self._extension = transcript_extension
        self._cleaned_suffix = cleaned_suffix

    def execute(self, context: PipelineContext, logger: logging.Logger) -> PipelineContext:
        if context.document is None:
            context.fail("No document to write")
            return context

        stem = context.source_path.stem
        transcript_path = self._transcripts_folder / (stem + self._extension)
        write_transcript(context.document, transcript_path)
        logger.info("Transcript saved: %s", transcript_path)

        if context.cleaned_text:
            cleaned_path = self._transcripts_folder / (
                stem + self._cleaned_suffix + self._extension
            )
            write_text_file(cleaned_path, context.cleaned_text)
            logger.info("Cleaned transcript saved: %s", cleaned_path)

        context.document.pipeline_state = PipelineState.EXPORTED
        return context
