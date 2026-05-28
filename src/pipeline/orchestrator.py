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
                context.fail(f"Step '{step.name}' failed: {exc}")
                self._logger.exception("Step '%s' failed: %s", step.name, exc)
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
