from src.models import Speaker, TranscriptSegment
from src.transcription.alignment import (
    assign_speakers_to_segments,
    build_speaker_registry,
    detect_speaker_changes,
)
from src.transcription.diarization_backends.base import DiarizationTurn
from src.transcription.diarization_config import DiarizationConfig


def _config() -> DiarizationConfig:
    return DiarizationConfig(
        enabled=True,
        backend="pyannote",
        model="pyannote/speaker-diarization-3.1",
        hf_token_env="HF_TOKEN",
        num_speakers=None,
        min_speakers=2,
        max_speakers=10,
        key_speakers=3,
        overlap_threshold=0.2,
        min_segment_duration=0.3,
        progress_update_interval_seconds=0.25,
    )


class TestAssignSpeakersToSegments:
    def test_single_speaker_full_overlap(self):
        segments = [
            TranscriptSegment(text="a", start_time=0.0, end_time=5.0),
            TranscriptSegment(text="b", start_time=5.0, end_time=10.0),
        ]
        turns = [DiarizationTurn(speaker_id="spk_0", start_time=0.0, end_time=10.0)]

        result, overlap_count = assign_speakers_to_segments(segments, turns, _config())
        assert overlap_count == 0
        assert all(seg.speaker and seg.speaker.id == "spk_0" for seg in result)
        assert all(seg.confidence_details["diarization"] == 1.0 for seg in result)

    def test_speaker_change_between_segments(self):
        segments = [
            TranscriptSegment(text="a", start_time=0.0, end_time=5.0),
            TranscriptSegment(text="b", start_time=5.0, end_time=10.0),
        ]
        turns = [
            DiarizationTurn(speaker_id="spk_0", start_time=0.0, end_time=5.0),
            DiarizationTurn(speaker_id="spk_1", start_time=5.0, end_time=10.0),
        ]

        result, _ = assign_speakers_to_segments(segments, turns, _config())
        assert result[0].speaker.id == "spk_0"
        assert result[1].speaker.id == "spk_1"

    def test_overlap_speech_two_speakers(self):
        segments = [TranscriptSegment(text="both", start_time=0.0, end_time=10.0)]
        turns = [
            DiarizationTurn(speaker_id="spk_0", start_time=0.0, end_time=10.0),
            DiarizationTurn(speaker_id="spk_1", start_time=0.0, end_time=10.0),
        ]

        result, overlap_count = assign_speakers_to_segments(segments, turns, _config())
        assert overlap_count == 1
        assert result[0].confidence_details["overlap_speakers"] == 2

    def test_no_turns_for_segment(self):
        segments = [TranscriptSegment(text="gap", start_time=20.0, end_time=25.0)]
        turns = [DiarizationTurn(speaker_id="spk_0", start_time=0.0, end_time=10.0)]

        result, _ = assign_speakers_to_segments(segments, turns, _config())
        assert result[0].speaker is None
        assert result[0].is_unclear is True
        assert result[0].confidence_details["diarization"] == 0.0

    def test_short_segment_midpoint_lookup(self):
        segments = [TranscriptSegment(text="hi", start_time=1.0, end_time=1.2)]
        turns = [DiarizationTurn(speaker_id="spk_0", start_time=0.0, end_time=5.0)]

        result, _ = assign_speakers_to_segments(segments, turns, _config())
        assert result[0].speaker.id == "spk_0"


class TestBuildSpeakerRegistry:
    def test_key_speaker_ranking(self):
        turns = [
            DiarizationTurn(speaker_id="spk_0", start_time=0.0, end_time=100.0),
            DiarizationTurn(speaker_id="spk_1", start_time=0.0, end_time=50.0),
            DiarizationTurn(speaker_id="spk_2", start_time=0.0, end_time=30.0),
            DiarizationTurn(speaker_id="spk_3", start_time=0.0, end_time=10.0),
        ]
        segments = [
            TranscriptSegment(
                text="x",
                start_time=0.0,
                end_time=1.0,
                speaker=Speaker(id="spk_0"),
            )
        ]

        speakers = build_speaker_registry(turns, segments, key_speakers=3)
        key_ids = [s.id for s in speakers if s.is_key]
        assert key_ids == ["spk_0", "spk_1", "spk_2"]
        assert speakers[3].is_key is False


class TestDetectSpeakerChanges:
    def test_speaker_change_detection(self):
        segments = [
            TranscriptSegment(
                text="a",
                start_time=0.0,
                end_time=5.0,
                speaker=Speaker(id="spk_0"),
            ),
            TranscriptSegment(
                text="b",
                start_time=5.0,
                end_time=10.0,
                speaker=Speaker(id="spk_1"),
            ),
        ]
        turns = [
            DiarizationTurn(speaker_id="spk_0", start_time=0.0, end_time=5.0),
            DiarizationTurn(speaker_id="spk_1", start_time=5.0, end_time=10.0),
        ]

        changes = detect_speaker_changes(segments, turns)
        assert 5.0 in changes
