from __future__ import annotations

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
