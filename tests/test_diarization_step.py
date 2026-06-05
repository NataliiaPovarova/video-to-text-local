import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.models import PipelineContext, PipelineState, TranscriptDocument, TranscriptSegment
from src.pipeline.steps_diarization import DiarizationStep
from src.transcription.diarization_backends.base import DiarizationTurn
from src.transcription.diarization_config import DiarizationConfig


def _config(enabled: bool = True) -> DiarizationConfig:
    return DiarizationConfig(
        enabled=enabled,
        backend="pyannote",
        model="pyannote/speaker-diarization-3.1",
        hf_token_env="HF_TOKEN",
        num_speakers=None,
        min_speakers=2,
        max_speakers=10,
        key_speakers=3,
        overlap_threshold=0.2,
        min_segment_duration=0.3,
        progress_update_interval_seconds=0.25,
    )


class MockBackend:
    name = "mock"

    def diarize(self, audio_path, *, num_speakers=None, min_speakers=None, max_speakers=None):
        return [
            DiarizationTurn(speaker_id="spk_0", start_time=0.0, end_time=10.0),
        ]


class TestDiarizationStep:
    def _context(self) -> PipelineContext:
        return PipelineContext(
            source_path=Path("/tmp/test.mp3"),
            input_type="audio",
            language="ru",
            audio_path=Path("/tmp/test.mp3"),
            document=TranscriptDocument(
                segments=[TranscriptSegment(text="hello", start_time=0.0, end_time=5.0)],
                pipeline_state=PipelineState.TRANSCRIBED,
            ),
        )

    @patch("src.pipeline.steps_diarization.diarize_document")
    def test_execute_updates_document(self, mock_diarize):
        doc = TranscriptDocument(
            segments=[TranscriptSegment(text="hello", start_time=0.0, end_time=5.0)],
            pipeline_state=PipelineState.DIARIZED,
            speakers=[],
        )
        mock_diarize.return_value = doc

        step = DiarizationStep(MockBackend(), _config())
        result = step.execute(self._context(), logging.getLogger("test"))

        assert result.document.pipeline_state == PipelineState.DIARIZED
        mock_diarize.assert_called_once()

    def test_should_skip_when_disabled(self):
        step = DiarizationStep(MockBackend(), _config(enabled=False))
        assert step.should_skip(self._context()) is True

    def test_should_skip_on_prior_errors(self):
        ctx = self._context()
        ctx.fail("error")
        step = DiarizationStep(MockBackend(), _config())
        assert step.should_skip(ctx) is True
