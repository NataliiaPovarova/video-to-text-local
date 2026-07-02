import logging
from pathlib import Path

from src.models import PipelineContext, PipelineState, TranscriptDocument, TranscriptSegment
from src.output.formatter import write_text_file, write_transcript
from src.pipeline import OutputStep


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


def test_output_step_uses_basename_override(tmp_path):
    doc = TranscriptDocument(segments=[TranscriptSegment(text="hello")])
    ctx = PipelineContext(source_path=tmp_path / "orig name.mp3", input_type="audio", document=doc)
    step = OutputStep(tmp_path, ".txt", "_clean", output_basename="orig_name__base")
    step.execute(ctx, logging.getLogger("t"))
    written = tmp_path / "orig_name__base.txt"
    assert written.exists()
    assert ctx.document.metadata["outputs"]["txt"] == str(written)
    assert ctx.document.metadata["outputs"]["clean"] is None


def test_output_step_defaults_to_source_stem(tmp_path):
    doc = TranscriptDocument(segments=[TranscriptSegment(text="hi")])
    ctx = PipelineContext(source_path=tmp_path / "clip.mp3", input_type="audio", document=doc)
    OutputStep(tmp_path, ".txt", "_clean").execute(ctx, logging.getLogger("t"))
    assert (tmp_path / "clip.txt").exists()


def test_output_step_writes_clean_when_present(tmp_path):
    doc = TranscriptDocument(segments=[TranscriptSegment(text="hi")])
    ctx = PipelineContext(source_path=tmp_path / "clip.mp3", input_type="audio", document=doc)
    ctx.cleaned_text = "cleaned"
    OutputStep(tmp_path, ".txt", "_clean", output_basename="clip__base").execute(ctx, logging.getLogger("t"))
    assert (tmp_path / "clip__base_clean.txt").read_text(encoding="utf-8") == "cleaned"
    assert ctx.document.metadata["outputs"]["clean"] == str(tmp_path / "clip__base_clean.txt")


def test_output_step_writes_srt_vtt_when_enabled(tmp_path):
    doc = TranscriptDocument(segments=[TranscriptSegment(text="hi", start_time=0.0, end_time=1.0)])
    ctx = PipelineContext(source_path=tmp_path / "c.mp3", input_type="audio", document=doc)
    OutputStep(tmp_path, ".txt", "_clean", output_basename="c__base", write_srt_vtt=True).execute(
        ctx, logging.getLogger("t")
    )
    assert (tmp_path / "c__base.srt").exists()
    assert (tmp_path / "c__base.vtt").exists()
    assert ctx.document.metadata["outputs"]["srt"] == str(tmp_path / "c__base.srt")
    assert ctx.document.metadata["outputs"]["vtt"] == str(tmp_path / "c__base.vtt")
