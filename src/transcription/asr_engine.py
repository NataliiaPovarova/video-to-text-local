from __future__ import annotations

import contextvars
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

from src.models import PipelineState, TranscriptDocument, TranscriptSegment

try:
    from moviepy import AudioFileClip
except ImportError:
    try:
        from moviepy.editor import AudioFileClip  # type: ignore[no-redef]
    except ImportError:
        AudioFileClip = None  # type: ignore[misc, assignment]


def _get_audio_duration_seconds(audio_path: str) -> float | None:
    if AudioFileClip is None:
        return None
    try:
        with AudioFileClip(audio_path) as clip:
            return float(clip.duration) if clip.duration else None
    except Exception:
        return None


def _estimate_fraction(elapsed_seconds: float, total_seconds: float) -> float:
    """Return a smooth, honest 0.0-1.0 progress estimate for a Whisper run.

    Whisper's blocking ``transcribe()`` exposes no real progress signal, and on
    CPU it runs slower than realtime. The naive ``elapsed / audio_duration``
    estimate therefore reached 1.0 (100%) at the 1x-realtime mark and then sat
    there for the rest of the decode. Instead use ``elapsed / (elapsed +
    duration)``, which rises quickly early, slows as it goes, and asymptotically
    approaches -- but never reaches -- 1.0, so the bar always shows forward
    motion and only the real completion (the ``finally`` 1.0) marks the run
    done. Capped at 0.99 as a float-safety floor strictly below 1.0.
    """
    if total_seconds <= 0 or elapsed_seconds <= 0:
        return 0.0
    return min(elapsed_seconds / (elapsed_seconds + total_seconds), 0.99)


def transcribe_audio(
    model,
    audio_path: Path,
    language: str | None,
    progress_update_interval_seconds: float,
    logger: logging.Logger,
    *,
    progress_callback: Callable[[float], None] | None = None,
) -> TranscriptDocument:
    """Run Whisper transcription and return a structured TranscriptDocument.

    Preserves segment-level data (text, timestamps) from Whisper output. When
    ``language`` is falsy (None or ""), Whisper auto-detects and the detected
    language is stored on the returned document. ``progress_callback`` (optional)
    receives a 0.0-1.0 fraction during the run and 1.0 on completion.
    """
    audio_str = str(audio_path)
    total_seconds = _get_audio_duration_seconds(audio_str)

    stop_event = threading.Event()
    pbar: tqdm | None = None

    def _run_progress_bar() -> None:
        nonlocal pbar
        if total_seconds is None or total_seconds <= 0:
            return
        pbar = tqdm(total=int(total_seconds), desc=f"Transcribing {audio_path.name}", unit="s")
        start_time = time.time()
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            current = int(min(elapsed, pbar.total))
            if current != pbar.n:
                pbar.n = current
                pbar.refresh()
            if progress_callback is not None:
                progress_callback(_estimate_fraction(elapsed, total_seconds))
            time.sleep(progress_update_interval_seconds)
        pbar.n = pbar.total
        pbar.refresh()
        pbar.close()

    thread: threading.Thread | None = None
    if total_seconds and total_seconds > 0:
        # Run the ticker inside a COPY of the current context so it inherits the
        # caller's ContextVars. Gradio's gr.Progress reads request-scoped
        # ContextVars (blocks + event_id) on every call; a raw threading.Thread
        # does NOT inherit them, so ticker-thread progress updates would be
        # silently dropped (only the main-thread finally(1.0) would land).
        # copy_context() captures the handler thread's vars here and ctx.run()
        # applies them inside the ticker thread. Harmless when no relevant vars
        # are set (e.g. the CLI, where progress_callback is None).
        ctx = contextvars.copy_context()
        thread = threading.Thread(target=lambda: ctx.run(_run_progress_bar), daemon=True)
        thread.start()
    else:
        logger.debug("Could not estimate media duration for progress bar: %s", audio_path)

    transcribe_kwargs = {"fp16": False, "condition_on_previous_text": False}
    if language:  # None or "" => auto-detect (omit the language kwarg entirely)
        transcribe_kwargs["language"] = language

    try:
        # condition_on_previous_text=False stops Whisper's self-conditioning
        # decode loop from running away into repeated/garbled phantom segments
        # on low-confidence trailing audio (silence, outro, background noise).
        # With the default (True), large-v3 emitted ~18s of hallucinated
        # content past the true end of audio. See tests/test_asr_engine.py.
        result = model.transcribe(audio_str, **transcribe_kwargs)
    finally:
        stop_event.set()
        if thread:
            thread.join(timeout=0.5)
        if progress_callback is not None:
            progress_callback(1.0)

    segments = [
        TranscriptSegment(
            text=seg.get("text", "").strip(),
            start_time=seg.get("start"),
            end_time=seg.get("end"),
        )
        for seg in result.get("segments", [])
    ]

    if not segments and result.get("text"):
        segments = [TranscriptSegment(text=result["text"])]

    detected_language = result.get("language") or language or "unknown"

    doc = TranscriptDocument(
        source_file=str(audio_path),
        segments=segments,
        language=detected_language,
        pipeline_state=PipelineState.TRANSCRIBED,
    )

    logger.info(
        "Transcription complete: %d segments, %.1fs total",
        len(segments),
        doc.duration_seconds or 0.0,
    )
    return doc
