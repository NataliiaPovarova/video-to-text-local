from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class PipelineState(str, Enum):
    PENDING = "pending"
    INGESTED = "ingested"
    TRANSCRIBED = "transcribed"
    PROCESSED = "processed"
    EXPORTED = "exported"
    FAILED = "failed"


@dataclass
class Speaker:
    id: str
    label: str | None = None


@dataclass
class TranscriptSegment:
    """A single segment of transcribed audio."""

    text: str
    start_time: float | None = None
    end_time: float | None = None
    speaker: Speaker | None = None
    confidence: float = 1.0
    is_unclear: bool = False


@dataclass
class TranscriptDocument:
    """Full transcript with metadata and provenance."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_file: str = ""
    segments: list[TranscriptSegment] = field(default_factory=list)
    speakers: list[Speaker] = field(default_factory=list)
    language: str = "ru"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    pipeline_state: PipelineState = PipelineState.PENDING
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return " ".join(seg.text for seg in self.segments if seg.text)

    @property
    def duration_seconds(self) -> float | None:
        if not self.segments:
            return None
        ends = [s.end_time for s in self.segments if s.end_time is not None]
        return max(ends) if ends else None


@dataclass
class PipelineContext:
    """Carries state through the processing pipeline."""

    source_path: Path
    input_type: str  # "video" | "audio"
    language: str = "ru"
    audio_path: Path | None = None
    document: TranscriptDocument | None = None
    cleaned_text: str | None = None
    errors: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.errors.append(message)
        if self.document:
            self.document.pipeline_state = PipelineState.FAILED
