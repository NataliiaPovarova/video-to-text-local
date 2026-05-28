from __future__ import annotations

from pathlib import Path

from src.models import TranscriptDocument


def write_transcript(document: TranscriptDocument, output_path: Path) -> None:
    """Write a TranscriptDocument to a text file.

    Currently writes plain text (full_text). Future phases will add
    markdown formatting, timestamps, speaker labels, and hyperlinks.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document.full_text, encoding="utf-8")


def write_text_file(path: str | Path, text: str) -> None:
    """Write raw text to a file (backward-compatible helper)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
