import logging
from pathlib import Path

import pytest

from src.models import PipelineContext
from src.pipeline import PipelineOrchestrator, PipelineStep
from src.utils import MediaDecodeError, ProcessingError
from src.pipeline import AudioIngestionStep, VideoIngestionStep


class RaisingStep(PipelineStep):
    def __init__(self, exc: Exception):
        self._exc = exc

    def execute(self, context, logger):
        raise self._exc


def _ctx():
    return PipelineContext(source_path=Path("/tmp/x.mp3"), input_type="audio")


def test_media_decode_error_is_processing_error():
    assert issubclass(MediaDecodeError, ProcessingError)


def test_orchestrator_records_exception_on_context():
    logger = logging.getLogger("test")
    exc = MediaDecodeError("bad file")
    pipeline = PipelineOrchestrator(steps=[RaisingStep(exc)], logger=logger)
    ctx = pipeline.run(_ctx())
    assert ctx.exception is exc
    assert ctx.errors  # string error still recorded


def test_orchestrator_exception_none_on_success():
    logger = logging.getLogger("test")
    pipeline = PipelineOrchestrator(steps=[], logger=logger)
    ctx = pipeline.run(_ctx())
    assert ctx.exception is None


def test_audio_ingestion_raises_on_undecodable(monkeypatch, tmp_path):
    fake = tmp_path / "broken.mp3"
    fake.write_bytes(b"not really audio")
    # Force the duration probe to report "cannot decode".
    monkeypatch.setattr("src.pipeline.steps_ingestion._probe_duration", lambda p: None)
    step = AudioIngestionStep()
    ctx = PipelineContext(source_path=fake, input_type="audio")
    with pytest.raises(MediaDecodeError):
        step.execute(ctx, logging.getLogger("t"))


def test_video_ingestion_raises_on_no_audio_track(monkeypatch, tmp_path):
    fake = tmp_path / "silent.mp4"
    fake.write_bytes(b"x")
    monkeypatch.setattr(
        "src.pipeline.steps_ingestion.extract_audio_from_video",
        lambda src, out, logger: False,  # simulate no audio track
    )
    step = VideoIngestionStep(tmp_path, ".mp3")
    ctx = PipelineContext(source_path=fake, input_type="video")
    with pytest.raises(MediaDecodeError):
        step.execute(ctx, logging.getLogger("t"))
