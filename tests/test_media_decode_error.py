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


from src.utils.ffmpeg_tools import ExtractionError, MediaInfo  # noqa: E402

_AUDIO = MediaInfo(True, True, 1.0, 1, 0, "mp3", 44100, "mono")
_VIDEO = MediaInfo(True, True, 1.0, 1, 1, "aac", 44100, "mono")
_MUTE_VIDEO = MediaInfo(True, False, 1.0, 0, 1, None, None, None)


def test_audio_ingestion_raises_on_undecodable(tmp_path):
    fake = tmp_path / "broken.mp3"
    fake.write_bytes(b"not really audio")
    ctx = PipelineContext(source_path=fake, input_type="audio", media=MediaInfo.unreadable())
    with pytest.raises(MediaDecodeError, match="Could not read"):
        AudioIngestionStep().execute(ctx, logging.getLogger("t"))


def test_audio_ingestion_probes_when_context_has_no_media(monkeypatch, tmp_path):
    # Steps built outside the seam (no context.media) probe the file themselves.
    fake = tmp_path / "broken.mp3"
    fake.write_bytes(b"x")
    monkeypatch.setattr("src.pipeline.steps_ingestion.probe_media", lambda p, ffmpeg="ffmpeg": MediaInfo.unreadable())
    ctx = PipelineContext(source_path=fake, input_type="audio")
    with pytest.raises(MediaDecodeError):
        AudioIngestionStep().execute(ctx, logging.getLogger("t"))
    assert ctx.media == MediaInfo.unreadable()


def test_video_ingestion_raises_on_no_audio_track(monkeypatch, tmp_path):
    fake = tmp_path / "silent.mp4"
    fake.write_bytes(b"x")
    calls = []
    monkeypatch.setattr("src.pipeline.steps_ingestion.extract_audio", lambda *a, **k: calls.append(a))
    step = VideoIngestionStep(tmp_path, ".mp3")
    ctx = PipelineContext(source_path=fake, input_type="video", media=_MUTE_VIDEO)
    with pytest.raises(MediaDecodeError, match="no audio track"):
        step.execute(ctx, logging.getLogger("t"))
    assert calls == []  # no extraction attempted


def test_video_ingestion_wraps_extraction_failure(monkeypatch, tmp_path):
    fake = tmp_path / "clip.mp4"
    fake.write_bytes(b"x")

    def boom(*a, **k):
        raise ExtractionError("ffmpeg died")

    monkeypatch.setattr("src.pipeline.steps_ingestion.extract_audio", boom)
    ctx = PipelineContext(source_path=fake, input_type="video", media=_VIDEO)
    with pytest.raises(MediaDecodeError, match="Could not extract audio"):
        VideoIngestionStep(tmp_path, ".mp3").execute(ctx, logging.getLogger("t"))
