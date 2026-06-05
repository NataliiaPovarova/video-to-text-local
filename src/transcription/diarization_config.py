from __future__ import annotations

import os
from dataclasses import dataclass

from src.utils.config import load_yaml_file
from src.utils.errors import ProcessingError


@dataclass
class DiarizationConfig:
    enabled: bool
    backend: str
    model: str
    hf_token_env: str
    num_speakers: int | None
    min_speakers: int
    max_speakers: int
    key_speakers: int
    overlap_threshold: float
    min_segment_duration: float
    progress_update_interval_seconds: float

    def resolve_hf_token(self) -> str:
        token = os.environ.get(self.hf_token_env, "").strip()
        if not token:
            raise ProcessingError(
                f"Diarization requires HuggingFace token in environment variable "
                f"'{self.hf_token_env}'. Create a token at "
                "https://huggingface.co/settings/tokens and accept the user conditions "
                "for every gated model used by pyannote.audio: "
                "pyannote/segmentation-3.0, pyannote/speaker-diarization-3.1, and "
                "pyannote/speaker-diarization-community-1 "
                "(the last one is required by pyannote.audio >= 4.0, which loads PLDA "
                "weights from that repository even when the configured model is "
                "speaker-diarization-3.1)."
            )
        return token


def load_diarization_config(path: str) -> DiarizationConfig:
    data = load_yaml_file(path)
    speakers = data.get("speakers", {})
    alignment = data.get("alignment", {})
    processing = data.get("processing", {})

    num_speakers = speakers.get("num_speakers")
    if num_speakers is not None:
        num_speakers = int(num_speakers)

    return DiarizationConfig(
        enabled=bool(data.get("enabled", False)),
        backend=str(data.get("backend", "pyannote")),
        model=str(data.get("model", "pyannote/speaker-diarization-3.1")),
        hf_token_env=str(data.get("hf_token_env", "HF_TOKEN")),
        num_speakers=num_speakers,
        min_speakers=int(speakers.get("min_speakers", 2)),
        max_speakers=int(speakers.get("max_speakers", 10)),
        key_speakers=int(speakers.get("key_speakers", 3)),
        overlap_threshold=float(alignment.get("overlap_threshold", 0.2)),
        min_segment_duration=float(alignment.get("min_segment_duration", 0.3)),
        progress_update_interval_seconds=float(
            processing.get("progress_update_interval_seconds", 0.25)
        ),
    )
