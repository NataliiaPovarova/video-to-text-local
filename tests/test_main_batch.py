import logging
from pathlib import Path

import main
from src.service import TranscribeResult
from src.utils import MediaDecodeError


def _ok():
    return TranscribeResult(
        document=None, txt_path=None, clean_path=None, srt_path=None, vtt_path=None,
        model="base", language="en", options={}, status="success",
    )


def test_batch_continues_after_one_file_raises(monkeypatch, tmp_path):
    attempted = []

    def fake_transcribe(file_path, **kwargs):
        attempted.append(Path(file_path).name)
        if len(attempted) == 2:  # second of three files blows up
            raise MediaDecodeError("corrupt")
        return _ok()

    monkeypatch.setattr(main, "transcribe_file", fake_transcribe)
    files = [
        ("a.mp3", tmp_path / "a.mp3"),
        ("b.mp3", tmp_path / "b.mp3"),
        ("c.mp3", tmp_path / "c.mp3"),
    ]
    main._process_batch(
        files, model_name="base", language="en", cleanup=False,
        diarize=False, num_speakers=None, logger=logging.getLogger("t"),
    )
    assert attempted == ["a.mp3", "b.mp3", "c.mp3"]  # all three attempted despite #2 raising


def test_batch_continues_after_unexpected_exception(monkeypatch, tmp_path):
    attempted = []

    def fake_transcribe(file_path, **kwargs):
        attempted.append(Path(file_path).name)
        if len(attempted) == 1:
            raise RuntimeError("boom")  # non-MediaDecodeError must also be caught
        return _ok()

    monkeypatch.setattr(main, "transcribe_file", fake_transcribe)
    files = [("a.mp3", tmp_path / "a.mp3"), ("b.mp3", tmp_path / "b.mp3")]
    main._process_batch(
        files, model_name="base", language="en", cleanup=False,
        diarize=False, num_speakers=None, logger=logging.getLogger("t"),
    )
    assert attempted == ["a.mp3", "b.mp3"]
