from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path

from src.models import PipelineState, TranscriptDocument, TranscriptSegment
from src.transcription.alignment import (
    assign_speakers_to_segments,
    build_speaker_registry,
    detect_speaker_changes,
)
from src.transcription.asr_engine import _get_audio_duration_seconds
from src.transcription.audio_prep import ensure_wav_16k_mono
from src.transcription.diarization_backends.base import DiarizationBackend, DiarizationTurn
from src.transcription.diarization_backends.pyannote_backend import create_pyannote_backend
from src.transcription.diarization_config import DiarizationConfig
from src.utils.errors import ProcessingError
from src.utils.progress import duration_progress_bar


def create_diarization_backend(
    config: DiarizationConfig,
    device: str,
    logger: logging.Logger,
) -> DiarizationBackend:
    if config.backend == "pyannote":
        return create_pyannote_backend(config, device, logger)
    raise ProcessingError(f"Unsupported diarization backend: {config.backend}")


def _remap_segment_speakers(
    segments: list[TranscriptSegment],
    speakers: list,
) -> None:
    registry = {speaker.id: speaker for speaker in speakers}
    for segment in segments:
        if segment.speaker is not None:
            segment.speaker = registry.get(segment.speaker.id, segment.speaker)


def merge_chunk_diarizations(
    chunk_results: list[tuple[list[DiarizationTurn], list[TranscriptSegment]]],
    overlap_seconds: float,
) -> tuple[list[DiarizationTurn], list[TranscriptSegment]]:
    """Stub for Phase 2.4: remap speaker IDs across chunk boundaries."""
    raise NotImplementedError(
        "Chunk diarization merge is not implemented yet (see Phase 2.4)."
    )


def diarize_document(
    document: TranscriptDocument,
    audio_path: Path,
    backend: DiarizationBackend,
    config: DiarizationConfig,
    logger: logging.Logger,
    *,
    work_dir: Path | None = None,
    ffmpeg_executable: str = "ffmpeg",
    num_speakers_override: int | None = None,
) -> TranscriptDocument:
    """Run diarization and align speakers to existing transcript segments."""
    if not document.segments:
        raise ProcessingError("Cannot diarize document with no transcript segments")

    # Compute source duration up-front so the prep step shows a real-time
    # progress bar (ffmpeg decode of the source can take 30-60 s on first run
    # for hour-long inputs; later runs hit the cached WAV and finish instantly).
    duration = _get_audio_duration_seconds(str(audio_path))
    if duration and duration > 30 * 60:
        logger.warning(
            "Long audio (%.0f min): diarization may take significant time on CPU.",
            duration / 60,
        )

    prep_dir = work_dir or audio_path.parent / ".diarization_cache"
    with duration_progress_bar(
        desc=f"Preparing audio {audio_path.name}",
        duration_seconds=duration,
        update_interval_seconds=config.progress_update_interval_seconds,
    ):
        prepared_audio = ensure_wav_16k_mono(
            audio_path,
            prep_dir,
            ffmpeg_executable=ffmpeg_executable,
            logger=logger,
        )

    num_speakers = num_speakers_override if num_speakers_override is not None else config.num_speakers
    min_speakers = None if num_speakers is not None else config.min_speakers
    max_speakers = None if num_speakers is not None else config.max_speakers

    # Pyannote drives its own rich.progress bars through ProgressHook; other
    # backends get the duration-based fallback bar.
    use_duration_fallback = backend.name != "pyannote"
    diarize_desc = f"Diarizing {audio_path.name}"

    if use_duration_fallback:
        progress_ctx = duration_progress_bar(
            desc=diarize_desc,
            duration_seconds=duration,
            update_interval_seconds=config.progress_update_interval_seconds,
        )
    else:
        progress_ctx = nullcontext()

    with progress_ctx:
        turns = backend.diarize(
            prepared_audio,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            logger=logger,
        )

    segments, overlap_segments_count = assign_speakers_to_segments(
        document.segments,
        turns,
        config,
        logger=logger,
    )
    speakers = build_speaker_registry(turns, segments, key_speakers=config.key_speakers)
    _remap_segment_speakers(segments, speakers)

    change_points = detect_speaker_changes(segments, turns)
    document.segments = segments
    document.speakers = speakers
    document.pipeline_state = PipelineState.DIARIZED
    document.metadata["diarization"] = {
        "backend": backend.name,
        "model": getattr(backend, "_model_id", config.model),
        "num_speakers_detected": len(speakers),
        "key_speaker_ids": [s.id for s in speakers if s.is_key],
        "speaker_change_points": change_points,
        "turns_count": len(turns),
        "overlap_segments_count": overlap_segments_count,
    }

    logger.info(
        "Diarization complete: %d speakers, %d turns, %d overlap segments",
        len(speakers),
        len(turns),
        overlap_segments_count,
    )
    return document
