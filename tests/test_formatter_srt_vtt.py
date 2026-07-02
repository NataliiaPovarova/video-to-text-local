from src.models import Speaker, TranscriptDocument, TranscriptSegment
from src.output.formatter import format_srt, format_vtt, write_srt, write_vtt


def _doc():
    return TranscriptDocument(segments=[
        TranscriptSegment(text="Hello.", start_time=0.0, end_time=1.5),
        TranscriptSegment(text="World.", start_time=1.5, end_time=3.25),
    ])


def test_srt_structure():
    srt = format_srt(_doc())
    assert "1\n00:00:00,000 --> 00:00:01,500\nHello." in srt
    assert "2\n00:00:01,500 --> 00:00:03,250\nWorld." in srt


def test_vtt_header_and_timestamps():
    vtt = format_vtt(_doc())
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in vtt


def test_speaker_label_prefixed():
    doc = TranscriptDocument(segments=[
        TranscriptSegment(text="Hi.", start_time=0.0, end_time=1.0, speaker=Speaker(id="SPEAKER_00", label="Alice")),
    ])
    assert "Alice: Hi." in format_srt(doc)


def test_segments_without_timestamps_skipped():
    doc = TranscriptDocument(segments=[TranscriptSegment(text="no times")])
    assert format_srt(doc).strip() == ""


def test_speaker_id_used_when_label_absent():
    doc = TranscriptDocument(segments=[
        TranscriptSegment(text="Hi.", start_time=0.0, end_time=1.0,
                          speaker=Speaker(id="SPEAKER_00", label=None)),
    ])
    assert "SPEAKER_00: Hi." in format_srt(doc)


def test_write_helpers(tmp_path):
    write_srt(_doc(), tmp_path / "a.srt")
    write_vtt(_doc(), tmp_path / "a.vtt")
    assert (tmp_path / "a.srt").read_text(encoding="utf-8").startswith("1")
    assert (tmp_path / "a.vtt").read_text(encoding="utf-8").startswith("WEBVTT")


def test_empty_document_srt_vtt_asymmetry():
    # Intentional: an empty SRT is legitimately empty, but a valid empty VTT
    # must still carry its WEBVTT header. Do NOT "fix" these to match.
    doc = TranscriptDocument(segments=[TranscriptSegment(text="no times")])
    assert format_srt(doc) == ""
    assert format_vtt(doc) == "WEBVTT\n"
