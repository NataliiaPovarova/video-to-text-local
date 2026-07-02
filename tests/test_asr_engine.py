import logging
from pathlib import Path
from unittest.mock import MagicMock

from src.transcription.asr_engine import transcribe_audio


def _fake_model(segments):
    model = MagicMock()
    model.transcribe.return_value = {
        "segments": segments,
        "text": " ".join(s["text"] for s in segments),
    }
    return model


def test_transcribe_disables_condition_on_previous_text():
    """Regression guard for the trailing-hallucination bug.

    With Whisper's default ``condition_on_previous_text=True`` the large-v3
    decoder ran away into a repeated/garbled loop, emitting ~18s of phantom
    segments past the true end of audio. The fix passes
    ``condition_on_previous_text=False``; this test pins that contract.
    """
    model = _fake_model([{"text": "hello world", "start": 0.0, "end": 1.0}])

    transcribe_audio(
        model=model,
        audio_path=Path("/tmp/does_not_exist.mp3"),
        language="en",
        progress_update_interval_seconds=0.25,
        logger=logging.getLogger("test_asr"),
    )

    model.transcribe.assert_called_once()
    _, kwargs = model.transcribe.call_args
    assert kwargs.get("condition_on_previous_text") is False, (
        "condition_on_previous_text must be False to prevent runaway "
        "trailing hallucination"
    )
    assert kwargs.get("language") == "en"
    assert kwargs.get("fp16") is False


def test_transcribe_preserves_segment_data():
    """Segment text/timestamps from Whisper survive into the document."""
    model = _fake_model(
        [
            {"text": " first ", "start": 0.0, "end": 1.5},
            {"text": "second", "start": 1.5, "end": 3.0},
        ]
    )

    doc = transcribe_audio(
        model=model,
        audio_path=Path("/tmp/does_not_exist.mp3"),
        language="en",
        progress_update_interval_seconds=0.25,
        logger=logging.getLogger("test_asr"),
    )

    assert [s.text for s in doc.segments] == ["first", "second"]
    assert doc.segments[0].start_time == 0.0
    assert doc.segments[-1].end_time == 3.0


class TestProgressAndLanguage:
    def _model(self, result):
        from unittest.mock import MagicMock

        model = MagicMock()
        model.transcribe.return_value = result
        return model

    def test_progress_callback_called_with_completion(self):
        import logging
        from pathlib import Path

        from src.transcription.asr_engine import transcribe_audio

        calls = []
        model = self._model({"text": "hi", "segments": [], "language": "en"})
        transcribe_audio(
            model=model,
            audio_path=Path("/tmp/nope.mp3"),
            language="en",
            progress_update_interval_seconds=0.01,
            logger=logging.getLogger("t"),
            progress_callback=lambda f: calls.append(f),
        )
        assert calls and calls[-1] == 1.0  # always signals completion

    def test_detected_language_captured_when_auto(self):
        import logging
        from pathlib import Path

        from src.transcription.asr_engine import transcribe_audio

        model = self._model({"text": "hola", "segments": [], "language": "es"})
        doc = transcribe_audio(
            model=model,
            audio_path=Path("/tmp/nope.mp3"),
            language=None,
            progress_update_interval_seconds=0.01,
            logger=logging.getLogger("t"),
        )
        assert doc.language == "es"
        # language=None means we pass no language kwarg to Whisper (auto-detect)
        _, kwargs = model.transcribe.call_args
        assert "language" not in kwargs

    def test_empty_string_language_routes_to_auto(self):
        import logging
        from pathlib import Path

        from src.transcription.asr_engine import transcribe_audio

        model = self._model({"text": "hi", "segments": [], "language": "fr"})
        transcribe_audio(
            model=model,
            audio_path=Path("/tmp/nope.mp3"),
            language="",  # empty string must also auto-detect (regression guard)
            progress_update_interval_seconds=0.01,
            logger=logging.getLogger("t"),
        )
        _, kwargs = model.transcribe.call_args
        assert "language" not in kwargs  # no language kwarg passed at all


class TestProgressEstimateAndContext:
    """Covers the two-fold progress-bar fix (follow-up 'A')."""

    def test_estimate_fraction_is_honest_and_asymptotic(self):
        # fold 1: the wallclock heuristic must NOT pin at 100% once wallclock
        # passes audio duration (the old `min(elapsed/total, 1.0)` returned 1.0
        # at elapsed==total, i.e. the 1x-realtime mark, then sat there).
        from src.transcription.asr_engine import _estimate_fraction

        assert _estimate_fraction(0.0, 60.0) == 0.0
        assert _estimate_fraction(-5.0, 60.0) == 0.0  # never negative
        assert _estimate_fraction(5.0, 0.0) == 0.0  # unknown/zero duration guard
        # elapsed == audio duration: old formula gave 1.0 (the bug); new gives 0.5
        assert _estimate_fraction(60.0, 60.0) == 0.5
        assert _estimate_fraction(180.0, 60.0) == 0.75
        # strictly monotonic increasing in elapsed
        xs = [_estimate_fraction(e, 60.0) for e in (1, 10, 60, 120, 600)]
        assert xs == sorted(xs) and len(set(xs)) == len(xs)
        # asymptotic: approaches but NEVER reaches 1.0, even for absurd elapsed
        assert _estimate_fraction(1e9, 60.0) < 1.0
        assert _estimate_fraction(1e9, 60.0) <= 0.99

    def test_progress_callback_runs_in_copied_context(self, monkeypatch):
        """fold 2: the ticker thread must inherit the handler thread's
        contextvars (via copy_context) so Gradio's gr.Progress -- which reads
        LocalContext.blocks/event_id ContextVars -- actually delivers
        ticker-thread updates instead of silently dropping them. A raw
        threading.Thread does NOT inherit contextvars, so without the fix the
        ticker sees the default sentinel value, not the handler thread's.

        Deterministic by construction: the fake transcribe blocks until the
        ticker has fired at least once (gated on a threading.Event the callback
        sets), so the assertion never races thread scheduling. The timeout is a
        safety net, not a timing assumption -- if the ticker never fires the
        test fails on the empty-list assert rather than hanging or flaking.
        """
        import contextvars
        import threading

        monkeypatch.setattr(
            "src.transcription.asr_engine._get_audio_duration_seconds", lambda p: 10.0
        )
        sentinel = contextvars.ContextVar("sentinel_asr_test", default="UNSET")
        sentinel.set("MAIN")
        main_thread = threading.current_thread()
        ticker_fired = threading.Event()
        ticker_views: list[str] = []

        def cb(_frac):
            # record ONLY ticker-thread observations (ignore the main-thread
            # finally(1.0) call, which trivially sees "MAIN")
            if threading.current_thread() is not main_thread:
                ticker_views.append(sentinel.get())
                ticker_fired.set()

        model = MagicMock()

        def slow_transcribe(*_a, **_k):
            # Block until the ticker has fired once, so the observation below is
            # guaranteed without relying on wallclock sleeps / scheduling luck.
            ticker_fired.wait(timeout=5.0)
            return {"text": "hi", "segments": [], "language": "en"}

        model.transcribe.side_effect = slow_transcribe
        transcribe_audio(
            model=model,
            audio_path=Path("/tmp/nope.mp3"),
            language="en",
            progress_update_interval_seconds=0.01,
            logger=logging.getLogger("t"),
            progress_callback=cb,
        )
        assert ticker_views, "ticker thread should have called progress_callback at least once"
        assert all(v == "MAIN" for v in ticker_views), (
            f"ticker thread must run inside a copied context; saw {set(ticker_views)} "
            "(raw threads do NOT inherit contextvars -> Gradio progress is a no-op)"
        )
