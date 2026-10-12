"""All ffmpeg subprocess interaction in one place (docs/AUDIO-PIPELINE-PLAN.md §4).

Uses only the ``ffmpeg`` binary (no ffprobe -- imageio-ffmpeg does not ship
it). Every call passes an argument list (no shell), so paths with spaces,
Cyrillic or other non-ASCII characters work the same on Windows, macOS and
Linux.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import MediaDecodeError, ProcessingError

SAMPLE_RATE = 16_000

# Anchored to line start: metadata lines ("    comment : Duration: ...", or a
# filename/title containing "Output #") must not be mistaken for structure.
_DURATION_RE = re.compile(r"^\s*Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", re.M)
_INPUT_RE = re.compile(r"^Input #0", re.M)
_OUTPUT_RE = re.compile(r"^Output #", re.M)
# "Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, ..."
_STREAM_RE = re.compile(r"^\s*Stream #\d+:\d+\S*:\s*(Audio|Video):\s*(.*)$")
_SAMPLE_RATE_RE = re.compile(r"(\d+) Hz")
_LAYOUT_RE = re.compile(r"\d+ Hz,\s*([^,]+)")

# Codec arguments by output extension. 16 kHz mono is exactly what Whisper and
# pyannote consume, so the extracted file needs no further conversion.
_EXTRACT_CODEC_ARGS = {
    ".wav": ["-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le"],
    ".flac": ["-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "flac"],
    ".mp3": ["-c:a", "libmp3lame", "-q:a", "2"],
}


class ExtractionError(ProcessingError):
    """ffmpeg could not extract the audio track."""


@dataclass(frozen=True)
class MediaInfo:
    readable: bool  # ffmpeg opened the file
    has_audio: bool  # at least one audio stream
    duration: float | None  # seconds; None when ffmpeg reports none (e.g. raw .aac) -- not an error
    audio_streams: int
    video_streams: int  # real video only; cover art ("attached pic") is not counted
    audio_codec: str | None  # first audio stream
    sample_rate: int | None
    channel_layout: str | None  # as ffmpeg prints it: "mono", "stereo", "5.1(side)"

    @property
    def is_video(self) -> bool:
        return self.video_streams > 0

    @classmethod
    def unreadable(cls) -> MediaInfo:
        return cls(False, False, None, 0, 0, None, None, None)


def _parse_probe_output(stderr: str) -> MediaInfo:
    """Parse the file description ``ffmpeg -i <file>`` prints to stderr.

    ``readable`` comes from the ``Input #0`` header, not the exit code: a bare
    ``ffmpeg -i`` exits 1 for good and bad files alike.
    """
    start = _INPUT_RE.search(stderr)
    if start is None:
        return MediaInfo.unreadable()
    section = stderr[start.start():]
    end = _OUTPUT_RE.search(section)
    if end is not None:
        section = section[: end.start()]

    duration = None
    m = _DURATION_RE.search(section)
    if m:
        h, mnt, s = m.groups()
        duration = int(h) * 3600 + int(mnt) * 60 + float(s)

    audio_streams = video_streams = 0
    codec = layout = None
    sample_rate = None
    for line in section.splitlines():
        sm = _STREAM_RE.match(line)
        if not sm:
            continue
        kind, details = sm.groups()
        if kind == "Video":
            if "(attached pic)" not in details:
                video_streams += 1
            continue
        audio_streams += 1
        if audio_streams == 1:
            codec = re.split(r"[\s,]", details, maxsplit=1)[0] or None
            rate = _SAMPLE_RATE_RE.search(details)
            sample_rate = int(rate.group(1)) if rate else None
            lay = _LAYOUT_RE.search(details)
            layout = lay.group(1).strip() if lay else None

    return MediaInfo(
        readable=True,
        has_audio=audio_streams > 0,
        duration=duration,
        audio_streams=audio_streams,
        video_streams=video_streams,
        audio_codec=codec,
        sample_rate=sample_rate,
        channel_layout=layout,
    )


def probe_media(path: Path | str, ffmpeg: str = "ffmpeg") -> MediaInfo:
    """Describe a media file via ``ffmpeg -i``. Never raises for bad input."""
    try:
        r = subprocess.run(
            [ffmpeg, "-hide_banner", "-nostdin", "-i", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # tags/filenames are not always valid UTF-8
        )
    except OSError:
        return MediaInfo.unreadable()
    return _parse_probe_output(r.stderr or "")


def get_media_duration_seconds(path: Path | str, ffmpeg: str = "ffmpeg") -> float | None:
    return probe_media(path, ffmpeg).duration


def extract_audio(video_path: Path | str, output_path: Path | str, ffmpeg: str = "ffmpeg") -> Path:
    """Extract the first audio track to ``output_path``; the extension picks the codec.

    Writes to a ``.part`` file first and renames it on success, so an
    interrupted run never leaves a truncated file that looks finished.
    """
    output_path = Path(output_path)
    ext = output_path.suffix.lower()
    codec_args = _EXTRACT_CODEC_ARGS.get(ext)
    if codec_args is None:
        raise ValueError(
            f"Unsupported extracted audio extension: {ext or '(none)'}; "
            f"use one of {', '.join(sorted(_EXTRACT_CODEC_ARGS))}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.stem + ".part" + output_path.suffix)
    cmd = [
        ffmpeg, "-hide_banner", "-nostdin", "-y",
        "-i", str(video_path),
        "-map", "0:a:0",  # first audio track
        "-vn",
        *codec_args,
        str(tmp_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ExtractionError(f"Could not run ffmpeg ('{ffmpeg}'): {exc}") from exc
    if r.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise ExtractionError((r.stderr or "").strip()[-800:])
    tmp_path.replace(output_path)
    return output_path


def decode_to_array(
    path: Path | str,
    ffmpeg: str = "ffmpeg",
    *,
    sample_rate: int = SAMPLE_RATE,
    filters: list[str] | None = None,
) -> np.ndarray:
    """Decode to a mono float32 array in [-1, 1].

    With no filters this reproduces ``whisper.audio.load_audio`` exactly
    (s16le, then / 32768), so feeding Whisper the array instead of the path
    leaves transcripts unchanged. ``filters`` are joined into one ``-af``
    chain; ffmpeg resamples to ``sample_rate`` after the chain.
    """
    cmd = [ffmpeg, "-nostdin", "-threads", "0", "-i", str(path)]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-"]
    try:
        r = subprocess.run(cmd, capture_output=True)
    except OSError as exc:
        raise MediaDecodeError(f"Could not run ffmpeg ('{ffmpeg}'): {exc}") from exc
    if r.returncode != 0:
        detail = r.stderr.decode("utf-8", errors="replace").strip()[-800:]
        raise MediaDecodeError(f"Could not decode '{Path(path).name}': {detail}")
    # .astype() copies, so the result is writable (np.frombuffer alone is read-only).
    return np.frombuffer(r.stdout, np.int16).flatten().astype(np.float32) / 32768.0
