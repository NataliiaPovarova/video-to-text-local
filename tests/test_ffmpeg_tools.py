"""Tests for src/utils/ffmpeg_tools.py (C-005, docs/AUDIO-PIPELINE-PLAN.md §4).

Media fixtures are generated with ffmpeg's lavfi sources at test time -- no
binary files live in the repo. Parser tests run on canned ffmpeg output and
need no ffmpeg at all.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.utils.errors import MediaDecodeError, ProcessingError
from src.utils.ffmpeg_tools import (
    ExtractionError,
    MediaInfo,
    _parse_probe_output,
    decode_to_array,
    extract_audio,
    get_media_duration_seconds,
    probe_media,
)


def _resolve_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


FFMPEG = _resolve_ffmpeg()
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="no ffmpeg binary available")

_VIDEO_SRC = ["-f", "lavfi", "-i", "testsrc=duration=1:size=64x48:rate=5"]
_SINE = ["-f", "lavfi", "-i", "sine=frequency=440:duration=1"]


def _make(path: Path, *args: str) -> Path:
    r = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args, str(path)],
        capture_output=True,
    )
    if r.returncode != 0:
        pytest.skip(f"this ffmpeg build cannot generate {path.name}: {r.stderr[-200:]!r}")
    return path


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict[str, Path]:
    if FFMPEG is None:
        pytest.skip("no ffmpeg binary available")
    d = tmp_path_factory.mktemp("media")
    m = {
        "video_audio": _make(d / "v_audio.mp4", *_VIDEO_SRC, *_SINE, "-shortest", "-c:a", "aac"),
        "video_mute": _make(d / "v_mute.mp4", *_VIDEO_SRC),
        "wav_stereo": _make(d / "a_stereo.wav", *_SINE, "-ac", "2"),
        "flac": _make(d / "a.flac", *_SINE),
        "aac": _make(d / "a.aac", *_SINE, "-c:a", "aac"),
        "unicode": _make(d / "запись 1 — тест.wav", *_SINE),
        "ogg": _make(d / "a.ogg", *_SINE),
        # Header/metadata text that looks like probe structure must not fool the parser.
        "output_in_name": _make(d / "Output #1.mp3", *_SINE),
        "output_in_tag": _make(d / "tag_output.mp3", *_SINE, "-metadata", "title=Output #2 mix"),
        "duration_in_tag": _make(d / "tag_duration.m4a", *_SINE, "-c:a", "aac",
                                 "-metadata", "comment=Duration: 00:45:00.00"),
    }
    cover = _make(d / "cover.png", "-f", "lavfi", "-i", "color=c=red:s=32x32", "-frames:v", "1")
    plain_mp3 = _make(d / "plain.mp3", *_SINE)
    m["mp3_cover"] = _make(
        d / "a_cover.mp3",
        "-i", str(plain_mp3), "-i", str(cover),
        "-map", "0", "-map", "1", "-c", "copy", "-disposition:v", "attached_pic",
    )
    broken = d / "broken.mp4"
    broken.write_bytes(bytes(range(256)) * 12)
    m["broken"] = broken
    text = d / "notes.txt"
    text.write_text("not media\n", encoding="utf-8")
    m["text"] = text
    return m


# --------------------------------------------------------------- parser (no ffmpeg)

_STDERR_VIDEO = """\
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'clip.mp4':
  Duration: 01:02:03.50, start: 0.000000, bitrate: 100 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p, 64x48, 5 fps (default)
  Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, 5.1(side), fltp, 70 kb/s (default)
At least one output file must be specified
"""

_STDERR_RAW_AAC_NA = """\
Input #0, aac, from 'a.aac':
  Duration: N/A, bitrate: 71 kb/s
  Stream #0:0: Audio: aac (LC), 44100 Hz, mono, fltp, 71 kb/s
"""

_STDERR_COVER = """\
Input #0, mp3, from 'a.mp3':
  Duration: 00:00:01.00, start: 0.025057, bitrate: 64 kb/s
  Stream #0:0: Audio: mp3 (mp3float), 44100 Hz, stereo, fltp, 64 kb/s
  Stream #0:1: Video: mjpeg (Baseline), yuvj420p(pc), 300x300, 90k tbr, 90k tbn (attached pic)
"""

_STDERR_INVALID = """\
[in#0 @ 000001febfd2d100] Error opening input: Invalid data found when processing input
Error opening input files: Invalid data found when processing input
"""


class TestParseProbeOutput:
    def test_video_with_audio(self):
        info = _parse_probe_output(_STDERR_VIDEO)
        assert info == MediaInfo(
            readable=True, has_audio=True, duration=3723.5,
            audio_streams=1, video_streams=1,
            audio_codec="aac", sample_rate=48000, channel_layout="5.1(side)",
        )
        assert info.is_video

    def test_duration_na_is_none_but_still_readable(self):
        info = _parse_probe_output(_STDERR_RAW_AAC_NA)
        assert info.readable and info.has_audio
        assert info.duration is None
        assert info.channel_layout == "mono"

    def test_attached_pic_is_not_a_video_stream(self):
        info = _parse_probe_output(_STDERR_COVER)
        assert info.video_streams == 0
        assert not info.is_video
        assert info.has_audio and info.audio_codec == "mp3"

    def test_invalid_input_is_unreadable(self):
        info = _parse_probe_output(_STDERR_INVALID)
        assert info == MediaInfo.unreadable()

    def test_empty_output_is_unreadable(self):
        assert _parse_probe_output("") == MediaInfo.unreadable()

    def test_output_text_in_filename_or_tags_does_not_truncate_streams(self):
        stderr = (
            "Input #0, mp3, from 'Output #1.mp3':\n"
            "  Metadata:\n"
            "    title           : Output #2 mix\n"
            "  Duration: 00:00:01.00, start: 0.025057, bitrate: 64 kb/s\n"
            "  Stream #0:0: Audio: mp3 (mp3float), 44100 Hz, mono, fltp, 64 kb/s\n"
        )
        info = _parse_probe_output(stderr)
        assert info.has_audio and info.audio_streams == 1
        assert info.duration == pytest.approx(1.0)

    def test_duration_in_metadata_is_ignored(self):
        stderr = (
            "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'a.m4a':\n"
            "  Metadata:\n"
            "    comment         : Duration: 00:45:00.00\n"
            "  Duration: 00:00:01.02, start: 0.000000, bitrate: 70 kb/s\n"
            "  Stream #0:0[0x1](und): Audio: aac (LC), 44100 Hz, mono, fltp, 69 kb/s (default)\n"
        )
        assert _parse_probe_output(stderr).duration == pytest.approx(1.02)

    def test_output_streams_after_input_are_ignored(self):
        # Only the Input #0 section describes the file; an "Output #0" block
        # (never produced by a bare `ffmpeg -i`, but defensive) must not count.
        stderr = _STDERR_RAW_AAC_NA + "Output #0, wav, to 'x.wav':\n  Stream #0:0: Audio: pcm_s16le, 16000 Hz, mono\n"
        assert _parse_probe_output(stderr).audio_streams == 1


# --------------------------------------------------------------- probe_media (real ffmpeg)

@needs_ffmpeg
class TestProbeMedia:
    def test_video_with_audio(self, media):
        info = probe_media(media["video_audio"], FFMPEG)
        assert info.readable and info.has_audio and info.is_video
        assert info.audio_streams == 1 and info.video_streams == 1
        assert info.audio_codec == "aac"
        assert info.duration == pytest.approx(1.0, abs=0.1)

    def test_mute_video_has_no_audio(self, media):
        info = probe_media(media["video_mute"], FFMPEG)
        assert info.readable and info.is_video
        assert not info.has_audio

    @pytest.mark.parametrize("key", ["output_in_name", "output_in_tag"])
    def test_output_text_in_name_or_tag_keeps_audio(self, media, key):
        info = probe_media(media[key], FFMPEG)
        assert info.has_audio and info.duration == pytest.approx(1.0, abs=0.1)

    def test_duration_tag_does_not_override_real_duration(self, media):
        assert probe_media(media["duration_in_tag"], FFMPEG).duration == pytest.approx(1.0, abs=0.1)

    @pytest.mark.parametrize("key", ["wav_stereo", "flac", "aac", "ogg"])
    def test_audio_formats_are_audio_not_video(self, media, key):
        info = probe_media(media[key], FFMPEG)
        assert info.readable and info.has_audio
        assert not info.is_video

    def test_wav_details(self, media):
        info = probe_media(media["wav_stereo"], FFMPEG)
        assert info.audio_codec == "pcm_s16le"
        assert info.sample_rate == 44100
        assert info.channel_layout == "stereo"

    def test_mp3_with_cover_art_is_audio(self, media):
        info = probe_media(media["mp3_cover"], FFMPEG)
        assert info.has_audio
        assert info.video_streams == 0 and not info.is_video

    @pytest.mark.parametrize("key", ["broken", "text"])
    def test_garbage_is_unreadable(self, media, key):
        assert probe_media(media[key], FFMPEG) == MediaInfo.unreadable()

    def test_missing_file_is_unreadable(self, tmp_path):
        assert probe_media(tmp_path / "nope.mp3", FFMPEG) == MediaInfo.unreadable()

    def test_missing_ffmpeg_binary_is_unreadable_not_crash(self, media, tmp_path):
        assert probe_media(media["wav_stereo"], str(tmp_path / "no-ffmpeg.exe")) == MediaInfo.unreadable()

    def test_unicode_path(self, media):
        info = probe_media(media["unicode"], FFMPEG)
        assert info.readable and info.has_audio

    def test_get_media_duration_seconds(self, media):
        assert get_media_duration_seconds(media["flac"], FFMPEG) == pytest.approx(1.0, abs=0.1)
        assert get_media_duration_seconds(media["broken"], FFMPEG) is None


# --------------------------------------------------------------- extract_audio

@needs_ffmpeg
class TestExtractAudio:
    def test_wav_is_16k_mono_pcm(self, media, tmp_path):
        out = extract_audio(media["video_audio"], tmp_path / "sub" / "out.wav", FFMPEG)
        assert out == tmp_path / "sub" / "out.wav"
        info = probe_media(out, FFMPEG)
        assert (info.audio_codec, info.sample_rate, info.channel_layout) == ("pcm_s16le", 16000, "mono")
        assert not info.is_video

    def test_flac_is_16k_mono(self, media, tmp_path):
        info = probe_media(extract_audio(media["video_audio"], tmp_path / "out.flac", FFMPEG), FFMPEG)
        assert (info.audio_codec, info.sample_rate, info.channel_layout) == ("flac", 16000, "mono")

    def test_mp3(self, media, tmp_path):
        info = probe_media(extract_audio(media["video_audio"], tmp_path / "out.mp3", FFMPEG), FFMPEG)
        assert info.audio_codec == "mp3" and not info.is_video

    def test_extension_is_case_insensitive(self, media, tmp_path):
        out = extract_audio(media["video_audio"], tmp_path / "OUT.WAV", FFMPEG)
        assert probe_media(out, FFMPEG).audio_codec == "pcm_s16le"

    def test_unsupported_extension_raises_and_writes_nothing(self, media, tmp_path):
        with pytest.raises(ValueError, match=".ogg"):
            extract_audio(media["video_audio"], tmp_path / "out.ogg", FFMPEG)
        assert list(tmp_path.iterdir()) == []

    def test_failure_leaves_no_output_and_no_part_file(self, media, tmp_path):
        with pytest.raises(ExtractionError):
            extract_audio(media["video_mute"], tmp_path / "out.wav", FFMPEG)
        assert list(tmp_path.iterdir()) == []

    def test_partial_part_file_is_removed_when_ffmpeg_fails_midway(self, monkeypatch, tmp_path):
        # ffmpeg may write part of the output before failing (e.g. a truncated
        # source). The .part file must not survive and must never be promoted.
        def fake_run(cmd, **kwargs):
            Path(cmd[-1]).write_bytes(b"half-written")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="decode error midway")

        monkeypatch.setattr("src.utils.ffmpeg_tools.subprocess.run", fake_run)
        with pytest.raises(ExtractionError, match="midway"):
            extract_audio(tmp_path / "in.mp4", tmp_path / "out.wav", "ffmpeg")
        assert sorted(p.name for p in tmp_path.iterdir()) == []

    def test_extraction_error_is_a_processing_error(self):
        assert issubclass(ExtractionError, ProcessingError)

    def test_replaces_stale_output(self, media, tmp_path):
        out = tmp_path / "out.wav"
        out.write_bytes(b"stale")
        extract_audio(media["video_audio"], out, FFMPEG)
        assert probe_media(out, FFMPEG).sample_rate == 16000


# --------------------------------------------------------------- decode_to_array

@needs_ffmpeg
class TestDecodeToArray:
    @pytest.mark.parametrize("key", ["video_audio", "wav_stereo", "aac"])
    def test_matches_whisper_load_audio_exactly(self, media, key, monkeypatch):
        # INV-1 groundwork: with no filters the samples must be bit-identical
        # to what openai-whisper decodes itself, so moving Whisper onto arrays
        # later (C-008) cannot change audio-file transcripts.
        whisper_audio = pytest.importorskip("whisper.audio")
        monkeypatch.setenv("PATH", str(Path(FFMPEG).parent), prepend=";" if "\\" in FFMPEG else ":")
        expected = whisper_audio.load_audio(str(media[key]))
        got = decode_to_array(media[key], FFMPEG)
        assert got.dtype == np.float32 and got.ndim == 1
        assert np.array_equal(got, expected)

    def test_length_matches_duration(self, media):
        samples = decode_to_array(media["flac"], FFMPEG)
        assert len(samples) == pytest.approx(16000, abs=16000 * 0.05)

    def test_custom_sample_rate(self, media):
        assert len(decode_to_array(media["flac"], FFMPEG, sample_rate=8000)) == pytest.approx(8000, abs=400)

    def test_filters_are_applied_and_preserve_length(self, media):
        plain = decode_to_array(media["flac"], FFMPEG)
        quiet = decode_to_array(media["flac"], FFMPEG, filters=["volume=0.5"])
        assert len(quiet) == len(plain)
        assert np.abs(quiet).max() == pytest.approx(np.abs(plain).max() / 2, rel=0.02)

    def test_result_is_writable(self, media):
        # torch.from_numpy warns/errs on read-only buffers (np.frombuffer).
        assert decode_to_array(media["flac"], FFMPEG).flags.writeable

    @pytest.mark.parametrize("key", ["video_mute", "broken"])
    def test_undecodable_raises_media_decode_error(self, media, key):
        with pytest.raises(MediaDecodeError):
            decode_to_array(media[key], FFMPEG)

    def test_unicode_path(self, media):
        assert len(decode_to_array(media["unicode"], FFMPEG)) > 0
