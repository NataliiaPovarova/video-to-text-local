from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from src.models import TranscriptDocument, TranscriptSegment


def format_segment_with_speaker(segment: TranscriptSegment) -> str:
    """Format a segment with optional speaker label and timestamp."""
    text = segment.text.strip()
    if not text:
        return ""

    if segment.speaker is not None:
        label = segment.speaker.label or segment.speaker.id
        if segment.start_time is not None:
            return f"[{label} {segment.start_time:.1f}s] {text}"
        return f"[{label}] {text}"

    if segment.start_time is not None and segment.end_time is not None:
        return f"[{segment.start_time:.1f}-{segment.end_time:.1f}] {text}"

    return text


def format_document_with_speakers(document: TranscriptDocument) -> str:
    """Join segments into readable text with speaker labels when available."""
    has_speakers = any(seg.speaker is not None for seg in document.segments)
    if not has_speakers:
        return document.full_text

    lines = [
        formatted
        for seg in document.segments
        if (formatted := format_segment_with_speaker(seg))
    ]
    return "\n".join(lines)


def write_transcript(document: TranscriptDocument, output_path: Path) -> None:
    """Write a TranscriptDocument to a text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = format_document_with_speakers(document)
    output_path.write_text(content, encoding="utf-8")


def write_text_file(path: str | Path, text: str) -> None:
    """Write raw text to a file (backward-compatible helper)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _format_timestamp(seconds: float, *, sep: str) -> str:
    """Format seconds as HH:MM:SS<sep>mmm (sep ',' for SRT, '.' for VTT)."""
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def _timed_segments(document: TranscriptDocument) -> Iterator[tuple[float, float, str]]:
    for seg in document.segments:
        if seg.start_time is None or seg.end_time is None:
            continue
        text = seg.text.strip()
        if not text:
            continue
        if seg.speaker is not None:
            label = seg.speaker.label or seg.speaker.id
            text = f"{label}: {text}"
        yield seg.start_time, seg.end_time, text


def format_srt(document: TranscriptDocument) -> str:
    blocks = []
    for index, (start, end, text) in enumerate(_timed_segments(document), start=1):
        blocks.append(
            f"{index}\n"
            f"{_format_timestamp(start, sep=',')} --> {_format_timestamp(end, sep=',')}\n"
            f"{text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def format_vtt(document: TranscriptDocument) -> str:
    lines = ["WEBVTT", ""]
    for start, end, text in _timed_segments(document):
        lines.append(f"{_format_timestamp(start, sep='.')} --> {_format_timestamp(end, sep='.')}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def write_srt(document: TranscriptDocument, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_srt(document), encoding="utf-8")


def write_vtt(document: TranscriptDocument, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_vtt(document), encoding="utf-8")
