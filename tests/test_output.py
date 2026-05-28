from pathlib import Path

from src.models import PipelineState, TranscriptDocument, TranscriptSegment
from src.output.formatter import write_text_file, write_transcript


class TestWriteTranscript:
    def test_writes_full_text(self, tmp_path: Path):
        doc = TranscriptDocument(
            segments=[
                TranscriptSegment(text="First segment."),
                TranscriptSegment(text="Second segment."),
            ]
        )
        output = tmp_path / "transcript.txt"
        write_transcript(doc, output)
        assert output.read_text(encoding="utf-8") == "First segment. Second segment."

    def test_creates_parent_directories(self, tmp_path: Path):
        doc = TranscriptDocument(segments=[TranscriptSegment(text="test")])
        output = tmp_path / "sub" / "dir" / "out.txt"
        write_transcript(doc, output)
        assert output.exists()
        assert output.read_text(encoding="utf-8") == "test"


class TestWriteTextFile:
    def test_basic_write(self, tmp_path: Path):
        path = tmp_path / "out.txt"
        write_text_file(path, "hello world")
        assert path.read_text(encoding="utf-8") == "hello world"

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "a" / "b" / "c.txt"
        write_text_file(path, "nested")
        assert path.read_text(encoding="utf-8") == "nested"
