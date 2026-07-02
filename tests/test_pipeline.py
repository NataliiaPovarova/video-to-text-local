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

    def test_logging_failure_does_not_mask_step_error(self):
        """A failure while LOGGING a step error must never replace the real
        step exception or crash the pipeline.

        Real-world trigger (the GUI-hang bug): logging an exception formats its
        traceback, which runs ``linecache.checkcache`` over ``sys.modules``;
        a broken lazy module there (speechbrain's ``k2_fsa``, missing the
        optional ``k2`` dep) raises a *new* ImportError mid-format. Unguarded,
        that secondary error escaped ``logger.exception``, skipped the
        orchestrator's ``break``, propagated out of ``transcribe_file``
        uncaught, and left the Gradio handler hanging at 100% with no
        transcript and no error. The orchestrator must isolate the logging
        call so the genuine step failure is always what reaches the caller as
        a failed context.
        """

        class ExplodingLogger:
            """Mimics the landmine: ``.exception()`` (the traceback-formatting
            path) raises; the traceback-free ``.error()`` fallback is safe."""

            def __init__(self):
                self.error_calls = []
                self.exception_attempted = False

            def info(self, *a, **k):
                pass

            def warning(self, *a, **k):
                pass

            def exception(self, *a, **k):
                self.exception_attempted = True
                raise ImportError(
                    "Lazy import of LazyModule(...speechbrain...k2_fsa) failed"
                )

            def error(self, *a, **k):
                self.error_calls.append((a, k))

        logger = ExplodingLogger()
        after = MagicMock(spec=PipelineStep)
        after.name = "After"
        after.should_skip.return_value = True
        pipeline = PipelineOrchestrator(steps=[FailingStep(), after], logger=logger)
        ctx = self._make_context()

        # Must NOT raise, even though logger.exception() raises mid-format.
        result = pipeline.run(ctx)

        # The REAL step error survives -- not the ImportError from logging.
        assert isinstance(result.exception, RuntimeError)
        assert "Intentional failure" in str(result.exception)
        assert len(result.errors) == 1 and "Intentional failure" in result.errors[0]
        # It tried the full-traceback exception() path FIRST (the useful one),
        # then fell back to the traceback-free logger.error path -- not an escape,
        # and not skipping straight to error() (which would lose tracebacks when
        # logging works normally).
        assert logger.exception_attempted
        assert logger.error_calls
        # The pipeline still stopped: the step after the failure never ran.
        after.execute.assert_not_called()
