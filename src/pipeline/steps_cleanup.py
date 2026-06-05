from __future__ import annotations

import logging
from collections.abc import Callable

from src.models import PipelineContext, PipelineState
from src.output.formatter import format_document_with_speakers

from .steps import PipelineStep


class CleanupStep(PipelineStep):
    """Run LLM-based text cleanup (Ollama) on the transcript."""

    def __init__(self, cleanup_func: Callable[[str], str]) -> None:
        self._cleanup_func = cleanup_func

    def execute(self, context: PipelineContext, logger: logging.Logger) -> PipelineContext:
        if context.document is None:
            context.fail("No document available for cleanup")
            return context

        if context.document.pipeline_state == PipelineState.DIARIZED:
            full_text = format_document_with_speakers(context.document)
        else:
            full_text = context.document.full_text
        if not full_text.strip():
            logger.warning("Empty transcript, skipping cleanup for %s", context.source_path.name)
            return context

        logger.info("Running cleanup for %s", context.source_path.name)
        context.cleaned_text = self._cleanup_func(full_text)
        return context
