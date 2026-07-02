from __future__ import annotations

import logging

from src.models import PipelineContext

from .steps import PipelineStep


class PipelineOrchestrator:
    """Executes a sequence of PipelineSteps on a PipelineContext."""

    def __init__(self, steps: list[PipelineStep], logger: logging.Logger) -> None:
        self._steps = steps
        self._logger = logger

    @property
    def steps(self) -> list[PipelineStep]:
        return list(self._steps)

    def run(self, context: PipelineContext) -> PipelineContext:
        self._logger.info(
            "Pipeline started for: %s (%d steps)", context.source_path.name, len(self._steps)
        )
        for step in self._steps:
            if step.should_skip(context):
                self._logger.warning(
                    "Skipping step '%s' due to prior errors.", step.name
                )
                continue

            self._logger.info("Executing step: %s", step.name)
            try:
                context = step.execute(context, self._logger)
            except Exception as exc:
                context.exception = exc
                context.fail(f"Step '{step.name}' failed: {exc}")
                self._log_step_failure(step.name, exc)
                break

        if context.errors:
            self._logger.error(
                "Pipeline finished with errors for %s: %s",
                context.source_path.name,
                context.errors,
            )
        else:
            self._logger.info("Pipeline completed successfully for: %s", context.source_path.name)

        return context

    def _log_step_failure(self, step_name: str, exc: BaseException) -> None:
        """Record a step failure without letting the logging call itself escape.

        ``logger.exception`` formats the exception's traceback, which runs
        ``linecache.checkcache`` over ``sys.modules``; a broken lazy module
        there (e.g. speechbrain's ``k2_fsa``, missing the optional ``k2`` dep)
        raises a *new* exception mid-format. Unguarded, that secondary error
        replaced the real step failure and crashed the pipeline -- in the GUI it
        left the progress bar pinned at 100%% with no transcript and no error.
        A logging failure must never mask the genuine failure, so fall back to a
        traceback-free record (which does not touch linecache) and, failing
        that, swallow it -- the real exception is already on the context.
        """
        try:
            self._logger.exception("Step '%s' failed: %s", step_name, exc)
        except Exception:  # noqa: BLE001 -- logging must never break the pipeline
            try:
                self._logger.error("Step '%s' failed: %s", step_name, exc)
            except Exception:  # noqa: BLE001
                pass
