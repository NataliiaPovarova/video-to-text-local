from src.models import Speaker, TranscriptDocument, TranscriptSegment
from src.output.formatter import format_document_with_speakers, format_segment_with_speaker


class TestFormatter:
    def test_format_segment_with_speaker(self):
        seg = TranscriptSegment(
            text="Hello",
            start_time=1.0,
            end_time=2.0,
            speaker=Speaker(id="spk_0", label="Alice"),
        )
        assert format_segment_with_speaker(seg) == "[Alice 1.0s] Hello"

    def test_format_document_with_speakers(self):
        doc = TranscriptDocument(
            segments=[
                TranscriptSegment(
                    text="Hi",
                    start_time=0.0,
                    end_time=1.0,
                    speaker=Speaker(id="spk_0"),
                ),
                TranscriptSegment(
                    text="Bye",
                    start_time=1.0,
                    end_time=2.0,
                    speaker=Speaker(id="spk_1"),
                ),
            ]
        )
        text = format_document_with_speakers(doc)
        assert "[spk_0 0.0s] Hi" in text
        assert "[spk_1 1.0s] Bye" in text

    def test_format_document_without_speakers_uses_full_text(self):
        doc = TranscriptDocument(
            segments=[
                TranscriptSegment(text="Hello"),
                TranscriptSegment(text="world"),
            ]
        )
        assert format_document_with_speakers(doc) == "Hello world"
