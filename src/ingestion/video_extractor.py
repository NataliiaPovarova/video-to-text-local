from __future__ import annotations

import logging
from pathlib import Path

from moviepy import VideoFileClip


def extract_audio_from_video(
    video_path: Path,
    audio_output_path: Path,
    logger: logging.Logger,
) -> bool:
    """Extract audio track from a video file.

    Returns True on success, False if no audio track found.
    """
    logger.info("Extracting audio from %s", video_path.name)
    with VideoFileClip(str(video_path)) as video:
        if video.audio is None:
            logger.warning("No audio track found in %s. Skipping.", video_path.name)
            return False
        video.audio.write_audiofile(str(audio_output_path))
    logger.info("Audio extracted: %s", audio_output_path)
    return True
