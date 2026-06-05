from __future__ import annotations

import logging
from copy import deepcopy

from tqdm import tqdm

from src.models import Speaker, TranscriptSegment
from src.transcription.diarization_backends.base import DiarizationTurn
from src.transcription.diarization_config import DiarizationConfig


def _overlap_duration(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _segment_duration(segment: TranscriptSegment) -> float | None:
    if segment.start_time is None or segment.end_time is None:
        return None
    duration = segment.end_time - segment.start_time
    return duration if duration > 0 else None


def _find_turns_at_midpoint(turns: list[DiarizationTurn], midpoint: float) -> list[DiarizationTurn]:
    return [t for t in turns if t.start_time <= midpoint < t.end_time]


def assign_speakers_to_segments(
    segments: list[TranscriptSegment],
    turns: list[DiarizationTurn],
    config: DiarizationConfig,
    logger: logging.Logger | None = None,
) -> tuple[list[TranscriptSegment], int]:
    """Assign speakers to transcript segments by temporal overlap with diarization turns."""
    result: list[TranscriptSegment] = []
    overlap_segments_count = 0
    speaker_cache: dict[str, Speaker] = {}

    for segment in tqdm(
        segments,
        desc="Aligning speakers",
        unit="seg",
        leave=False,
        disable=len(segments) < 2,
    ):
        seg = deepcopy(segment)
        duration = _segment_duration(seg)

        if duration is None:
            if logger:
                logger.warning("Skipping speaker assignment for segment without timestamps")
            result.append(seg)
            continue

        start = seg.start_time
        end = seg.end_time
        assert start is not None and end is not None

        use_midpoint = duration < config.min_segment_duration
        if use_midpoint:
            midpoint = (start + end) / 2.0
            midpoint_turns = _find_turns_at_midpoint(turns, midpoint)
            if len(midpoint_turns) == 1:
                winner_id = midpoint_turns[0].speaker_id
                seg.speaker = speaker_cache.setdefault(winner_id, Speaker(id=winner_id))
                seg.confidence_details["diarization"] = 1.0
                seg.confidence_details["overlap_speakers"] = 1
                result.append(seg)
                continue

        overlaps: dict[str, float] = {}
        for turn in turns:
            overlap = _overlap_duration(start, end, turn.start_time, turn.end_time)
            if overlap > 0:
                overlaps[turn.speaker_id] = overlaps.get(turn.speaker_id, 0.0) + overlap

        if not overlaps:
            seg.speaker = None
            seg.confidence_details["diarization"] = 0.0
            seg.confidence_details["overlap_speakers"] = 0
            seg.is_unclear = True
            result.append(seg)
            continue

        sorted_overlaps = sorted(overlaps.items(), key=lambda item: item[1], reverse=True)
        winner_id, winner_overlap = sorted_overlaps[0]
        diar_conf = min(1.0, winner_overlap / duration)

        significant_speakers = sum(
            1 for _, overlap in sorted_overlaps if overlap / duration > config.overlap_threshold
        )
        if significant_speakers > 1:
            overlap_segments_count += 1

        seg.speaker = speaker_cache.setdefault(winner_id, Speaker(id=winner_id))
        seg.confidence_details["diarization"] = round(diar_conf, 4)
        seg.confidence_details["overlap_speakers"] = significant_speakers
        if diar_conf < config.overlap_threshold:
            seg.is_unclear = True
        result.append(seg)

    return result, overlap_segments_count


def build_speaker_registry(
    turns: list[DiarizationTurn],
    segments: list[TranscriptSegment],
    key_speakers: int,
) -> list[Speaker]:
    """Build document-level speaker list with speaking-time stats and key-speaker flags."""
    speaking_seconds: dict[str, float] = {}
    for turn in turns:
        duration = turn.end_time - turn.start_time
        if duration > 0:
            speaking_seconds[turn.speaker_id] = speaking_seconds.get(turn.speaker_id, 0.0) + duration

    segment_ids = {seg.speaker.id for seg in segments if seg.speaker is not None}
    all_ids = sorted(set(speaking_seconds) | segment_ids)

    speakers = [
        Speaker(
            id=speaker_id,
            total_speaking_seconds=round(speaking_seconds.get(speaker_id, 0.0), 3),
        )
        for speaker_id in all_ids
    ]
    speakers.sort(key=lambda s: s.total_speaking_seconds, reverse=True)

    for index, speaker in enumerate(speakers):
        speaker.is_key = index < key_speakers

    return speakers


def detect_speaker_changes(
    segments: list[TranscriptSegment],
    turns: list[DiarizationTurn] | None = None,
) -> list[float]:
    """Return sorted unique timestamps where the active speaker changes."""
    change_points: set[float] = set()

    sorted_segments = sorted(
        [s for s in segments if s.start_time is not None and s.speaker is not None],
        key=lambda s: s.start_time or 0.0,
    )
    for index in range(1, len(sorted_segments)):
        prev = sorted_segments[index - 1]
        curr = sorted_segments[index]
        if prev.speaker and curr.speaker and prev.speaker.id != curr.speaker.id:
            change_points.add(curr.start_time or 0.0)

    if turns:
        sorted_turns = sorted(turns, key=lambda t: t.start_time)
        for index in range(1, len(sorted_turns)):
            prev_turn = sorted_turns[index - 1]
            curr_turn = sorted_turns[index]
            if prev_turn.speaker_id != curr_turn.speaker_id:
                change_points.add(curr_turn.start_time)

    return sorted(change_points)
