from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from src.models import PipelineContext


class PipelineStep(ABC):
    """Base class for all pipeline steps.

    Each step receives a PipelineContext, performs its work, and returns
    the (possibly modified) context. Steps can be composed into a pipeline
    via the PipelineOrchestrator.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def execute(self, context: PipelineContext, logger: logging.Logger) -> PipelineContext:
        ...

    def should_skip(self, context: PipelineContext) -> bool:
        """Override to conditionally skip this step."""
        return bool(context.errors)
