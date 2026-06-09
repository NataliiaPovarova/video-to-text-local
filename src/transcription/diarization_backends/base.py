from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class DiarizationTurn:
    speaker_id: str
    start_time: float
    end_time: float
    confidence: float | None = None


_SPEAKER_PATTERN = re.compile(r"SPEAKER_(\d+)", re.IGNORECASE)


def normalize_speaker_id(raw_id: str) -> str:
    """Map pyannote labels (SPEAKER_00) to stable ids (spk_0)."""
    match = _SPEAKER_PATTERN.match(raw_id.strip())
    if match:
        return f"spk_{int(match.group(1))}"
    if raw_id.startswith("spk_"):
        return raw_id
    return f"spk_{raw_id}"


class DiarizationBackend(Protocol):
    """Protocol for speaker diarization backends (local pyannote or cloud APIs)."""

    @property
    def name(self) -> str: ...

    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        logger: logging.Logger | None = None,
    ) -> list[DiarizationTurn]: ...
