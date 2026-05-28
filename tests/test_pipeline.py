import logging
from pathlib import Path
from unittest.mock import MagicMock

from src.models import PipelineContext, PipelineState, TranscriptDocument, TranscriptSegment
from src.pipeline import PipelineOrchestrator, PipelineStep


class PassthroughStep(PipelineStep):
    """A no-op step for testing."""

    def __init__(self, name: str = "Passthrough"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def execute(self, context: PipelineContext, logger: logging.Logger) -> PipelineContext:
        return context


class FailingStep(PipelineStep):
    """A step that always raises an exception."""

    def execute(self, context: PipelineContext, logger: logging.Logger) -> PipelineContext:
        raise RuntimeError("Intentional failure")


class EnrichingStep(PipelineStep):
    """A step that adds a document to context."""

    def execute(self, context: PipelineContext, logger: logging.Logger) -> PipelineContext:
        context.document = TranscriptDocument(
            source_file=str(context.source_path),
            segments=[TranscriptSegment(text="Hello from test")],
            pipeline_state=PipelineState.TRANSCRIBED,
        )
        return context


class TestPipelineOrchestrator:
    def _make_context(self) -> PipelineContext:
        return PipelineContext(
            source_path=Path("/tmp/test.mp3"),
            input_type="audio",
            language="ru",
        )

    def _make_logger(self) -> logging.Logger:
        return logging.getLogger("test_pipeline")

    def test_empty_pipeline(self):
        logger = self._make_logger()
        pipeline = PipelineOrchestrator(steps=[], logger=logger)
        ctx = self._make_context()
        result = pipeline.run(ctx)
        assert result.errors == []

    def test_single_step_success(self):
        logger = self._make_logger()
        pipeline = PipelineOrchestrator(steps=[PassthroughStep()], logger=logger)
        ctx = self._make_context()
        result = pipeline.run(ctx)
        assert result.errors == []

    def test_multi_step_chain(self):
        logger = self._make_logger()
        pipeline = PipelineOrchestrator(
            steps=[PassthroughStep("A"), EnrichingStep(), PassthroughStep("B")],
            logger=logger,
        )
        ctx = self._make_context()
        result = pipeline.run(ctx)
        assert result.errors == []
        assert result.document is not None
        assert result.document.full_text == "Hello from test"

    def test_failing_step_stops_pipeline(self):
        logger = self._make_logger()
        pipeline = PipelineOrchestrator(
            steps=[PassthroughStep("Before"), FailingStep(), PassthroughStep("After")],
            logger=logger,
        )
        ctx = self._make_context()
        result = pipeline.run(ctx)
        assert len(result.errors) == 1
        assert "Intentional failure" in result.errors[0]

    def test_skip_on_prior_errors(self):
        logger = self._make_logger()
        step_after = MagicMock(spec=PipelineStep)
        step_after.name = "MockAfter"
        step_after.should_skip.return_value = True

        pipeline = PipelineOrchestrator(
            steps=[FailingStep(), step_after],
            logger=logger,
        )
        ctx = self._make_context()
        pipeline.run(ctx)
        step_after.execute.assert_not_called()
