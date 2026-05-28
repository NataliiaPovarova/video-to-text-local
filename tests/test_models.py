from pathlib import Path

from src.models import (
    PipelineContext,
    PipelineState,
    Speaker,
    TranscriptDocument,
    TranscriptSegment,
)


class TestTranscriptSegment:
    def test_basic_creation(self):
        seg = TranscriptSegment(text="Hello world")
        assert seg.text == "Hello world"
        assert seg.start_time is None
        assert seg.end_time is None
        assert seg.speaker is None
        assert seg.confidence == 1.0
        assert seg.is_unclear is False

    def test_with_timestamps(self):
        seg = TranscriptSegment(text="test", start_time=1.5, end_time=3.2)
        assert seg.start_time == 1.5
        assert seg.end_time == 3.2

    def test_with_speaker(self):
        speaker = Speaker(id="spk_1", label="Alice")
        seg = TranscriptSegment(text="test", speaker=speaker)
        assert seg.speaker.label == "Alice"


class TestTranscriptDocument:
    def test_empty_document(self):
        doc = TranscriptDocument()
        assert doc.full_text == ""
        assert doc.duration_seconds is None
        assert doc.pipeline_state == PipelineState.PENDING

    def test_full_text_concatenation(self):
        doc = TranscriptDocument(
            segments=[
                TranscriptSegment(text="Hello"),
                TranscriptSegment(text="world"),
            ]
        )
        assert doc.full_text == "Hello world"

    def test_duration_from_segments(self):
        doc = TranscriptDocument(
            segments=[
                TranscriptSegment(text="a", start_time=0.0, end_time=5.0),
                TranscriptSegment(text="b", start_time=5.0, end_time=12.3),
            ]
        )
        assert doc.duration_seconds == 12.3

    def test_id_is_generated(self):
        doc1 = TranscriptDocument()
        doc2 = TranscriptDocument()
        assert doc1.id != doc2.id
        assert len(doc1.id) == 12


class TestPipelineContext:
    def test_creation(self):
        ctx = PipelineContext(
            source_path=Path("/tmp/test.mp3"),
            input_type="audio",
            language="ru",
        )
        assert ctx.source_path == Path("/tmp/test.mp3")
        assert ctx.input_type == "audio"
        assert ctx.audio_path is None
        assert ctx.document is None
        assert ctx.errors == []

    def test_fail_records_error(self):
        ctx = PipelineContext(
            source_path=Path("/tmp/test.mp3"),
            input_type="audio",
        )
        ctx.document = TranscriptDocument()
        ctx.fail("Something went wrong")
        assert "Something went wrong" in ctx.errors
        assert ctx.document.pipeline_state == PipelineState.FAILED
