from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from tqdm import tqdm

from src.models import PipelineState, TranscriptDocument, TranscriptSegment


def _get_audio_duration_seconds(audio_path: str) -> float | None:
    try:
        from moviepy import AudioFileClip

        with AudioFileClip(audio_path) as clip:
            return float(clip.duration) if clip.duration else None
    except Exception:
        try:
            from moviepy.editor import AudioFileClip as EditorAudioFileClip

            with EditorAudioFileClip(audio_path) as clip:
                return float(clip.duration) if clip.duration else None
        except Exception:
            return None


def transcribe_audio(
    model,
    audio_path: Path,
    language: str,
    progress_update_interval_seconds: float,
    logger: logging.Logger,
) -> TranscriptDocument:
    """Run Whisper transcription and return a structured TranscriptDocument.

    Preserves segment-level data (text, timestamps) from Whisper output.
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
            time.sleep(progress_update_interval_seconds)
        pbar.n = pbar.total
        pbar.refresh()
        pbar.close()

    thread: threading.Thread | None = None
    if total_seconds and total_seconds > 0:
        thread = threading.Thread(target=_run_progress_bar, daemon=True)
        thread.start()
    else:
        logger.debug("Could not estimate media duration for progress bar: %s", audio_path)

    try:
        result = model.transcribe(audio_str, language=language, fp16=False)
    finally:
        stop_event.set()
        if thread:
            thread.join(timeout=0.5)

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

    doc = TranscriptDocument(
        source_file=str(audio_path),
        segments=segments,
        language=language,
        pipeline_state=PipelineState.TRANSCRIBED,
    )

    logger.info(
        "Transcription complete: %d segments, %.1fs total",
        len(segments),
        doc.duration_seconds or 0.0,
    )
    return doc
