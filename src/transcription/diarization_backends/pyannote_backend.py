from __future__ import annotations

import logging
from pathlib import Path

import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from src.transcription.diarization_config import DiarizationConfig

from .base import DiarizationTurn, normalize_speaker_id


class PyannoteBackend:
    """Local speaker diarization via pyannote.audio."""

    def __init__(self, pipeline, model_id: str) -> None:
        self._pipeline = pipeline
        self._model_id = model_id

    @property
    def name(self) -> str:
        return "pyannote"

    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[DiarizationTurn]:
        kwargs: dict = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers is not None:
                kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                kwargs["max_speakers"] = max_speakers

        with ProgressHook() as hook:
            output = self._pipeline(str(audio_path), hook=hook, **kwargs)
        turns: list[DiarizationTurn] = []
        for turn, _, speaker in output.itertracks(yield_label=True):
            turns.append(
                DiarizationTurn(
                    speaker_id=normalize_speaker_id(str(speaker)),
                    start_time=float(turn.start),
                    end_time=float(turn.end),
                )
            )
        return turns


def create_pyannote_backend(
    config: DiarizationConfig,
    device: str,
    logger: logging.Logger,
) -> PyannoteBackend:
    token = config.resolve_hf_token()
    logger.info("Loading pyannote diarization model: %s", config.model)

    try:
        pipeline = Pipeline.from_pretrained(config.model, use_auth_token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(config.model, token=token)

    pipeline.to(torch.device(device))
    logger.info("Pyannote diarization model loaded on %s", device)
    return PyannoteBackend(pipeline=pipeline, model_id=config.model)
