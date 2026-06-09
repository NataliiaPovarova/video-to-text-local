from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from scipy.io import wavfile
from whisper.audio import load_audio as _load_audio_mono16k

from src.transcription.diarization_config import DiarizationConfig

from .base import DiarizationTurn, normalize_speaker_id


# Pyannote's diarization models operate internally at 16 kHz mono, so we
# load the audio at that rate up-front and skip any pyannote-side resampling.
_PYANNOTE_SAMPLE_RATE = 16000


class PyannoteBackend:
    """Local speaker diarization via pyannote.audio.

    Audio is loaded into memory and handed to the pipeline as a
    ``{"waveform": Tensor, "sample_rate": int, "uri": str}`` mapping. This
    makes pyannote's ``Audio.validate_file`` take its in-memory code path,
    which bypasses ``torchcodec`` entirely. On Windows ``torchcodec`` is
    fragile -- its native extension dynamically loads FFmpeg's shared
    libraries and frequently fails to find compatible exports across FFmpeg
    minor versions, crashing pyannote deep in the pipeline with
    ``NameError: name 'AudioDecoder' is not defined``. Loading audio
    ourselves removes that whole class of failures.

    Fast path: when the input is already a 16 kHz mono PCM WAV (the normal
    case, since ``ensure_wav_16k_mono`` is run upstream), the file is read
    via ``scipy.io.wavfile`` in well under a second even for multi-hour
    recordings. Slow fallback: any other format is decoded through
    :func:`whisper.audio.load_audio`, which shells out to ``ffmpeg.exe``.
    """

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
        logger: logging.Logger | None = None,
    ) -> list[DiarizationTurn]:
        kwargs: dict = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers is not None:
                kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                kwargs["max_speakers"] = max_speakers

        file_input = self._load_in_memory(audio_path, logger=logger)

        with ProgressHook() as hook:
            output = self._pipeline(file_input, hook=hook, **kwargs)

        # pyannote.audio >= 4.0 wraps the result in a `DiarizeOutput` dataclass
        # exposing `speaker_diarization` (overlap-preserving) and
        # `exclusive_speaker_diarization` (overlap-free) Annotations. Older
        # versions, and pipelines instantiated with `legacy=True`, return the
        # Annotation directly. We pick the overlap-preserving form because
        # the downstream alignment in `src/transcription/alignment.py`
        # already tracks overlap_segments_count in document metadata.
        annotation = getattr(output, "speaker_diarization", output)

        turns: list[DiarizationTurn] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(
                DiarizationTurn(
                    speaker_id=normalize_speaker_id(str(speaker)),
                    start_time=float(turn.start),
                    end_time=float(turn.end),
                )
            )
        return turns

    @classmethod
    def _load_in_memory(
        cls,
        audio_path: Path,
        *,
        logger: logging.Logger | None = None,
    ) -> dict:
        start = time.time()
        audio_np, sample_rate, source = cls._decode_audio(audio_path)
        # pyannote requires shape (channel, time) with channel <= time; the
        # mono float32 array becomes (1, num_samples) after unsqueeze(0).
        waveform = torch.from_numpy(np.ascontiguousarray(audio_np)).unsqueeze(0)
        if logger is not None:
            duration_s = waveform.shape[1] / float(sample_rate)
            logger.info(
                "Loaded audio for diarization (%s): %.1fs of audio in %.2fs wall time",
                source,
                duration_s,
                time.time() - start,
            )
        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "uri": audio_path.stem,
        }

    @staticmethod
    def _decode_audio(audio_path: Path) -> tuple[np.ndarray, int, str]:
        """Return (mono float32 waveform, sample_rate, source-label).

        Fast path uses scipy on the cached 16k mono WAV produced by
        ``ensure_wav_16k_mono``; slow path falls back to an ffmpeg subprocess.
        """
        if audio_path.suffix.lower() == ".wav":
            try:
                sample_rate, raw = wavfile.read(str(audio_path))
            except Exception:
                # Fall through to the ffmpeg path on any WAV-read failure.
                pass
            else:
                if raw.ndim == 2:
                    raw = raw.mean(axis=1)
                if raw.dtype == np.int16:
                    audio_np = raw.astype(np.float32) / 32768.0
                elif raw.dtype == np.int32:
                    audio_np = raw.astype(np.float32) / 2147483648.0
                elif raw.dtype == np.uint8:
                    audio_np = (raw.astype(np.float32) - 128.0) / 128.0
                elif raw.dtype == np.float32:
                    audio_np = raw
                else:
                    audio_np = raw.astype(np.float32)
                return audio_np, int(sample_rate), f"wav fast-path @ {sample_rate} Hz"

        audio_np = _load_audio_mono16k(str(audio_path), sr=_PYANNOTE_SAMPLE_RATE)
        return audio_np, _PYANNOTE_SAMPLE_RATE, "ffmpeg subprocess @ 16000 Hz"


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
