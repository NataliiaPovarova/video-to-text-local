import logging
import os
from collections.abc import Callable

from moviepy import VideoFileClip

from .utils import handle_error, transcribe_with_progress


def iter_supported_media_files(
    folder_path: str,
    supported_extensions: tuple[str, ...],
) -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    for media_file in os.listdir(folder_path):
        media_path = os.path.join(folder_path, media_file)
        if not os.path.isfile(media_path):
            continue
        extension = os.path.splitext(media_file)[1].lower()
        if extension not in supported_extensions:
            continue
        files.append((media_file, media_path))
    return files


def write_text_file(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def process_video_files(
    *,
    videos_path: str,
    audios_path: str,
    transcripts_folder: str,
    video_extensions: tuple[str, ...],
    transcript_extension: str,
    cleaned_suffix: str,
    extracted_audio_extension: str,
    whisper_model,
    language: str,
    enable_cleanup: bool,
    cleanup_func: Callable[[str], str] | None,
    progress_update_interval_seconds: float,
    logger: logging.Logger,
) -> None:
    media_files = iter_supported_media_files(videos_path, video_extensions)
    logger.info("Found %d supported video files in %s.", len(media_files), videos_path)

    for media_file, media_path in media_files:
        logger.info("Processing video: %s", media_file)
        audio_output_path = os.path.join(
            audios_path,
            os.path.splitext(media_file)[0] + extracted_audio_extension,
        )

        logger.info("Extracting audio from %s.", media_file)
        with VideoFileClip(media_path) as video:
            if video.audio is None:
                logger.warning("No audio track found in %s. Skipping file.", media_file)
                continue
            video.audio.write_audiofile(audio_output_path)
        logger.info("Audio extracted: %s", audio_output_path)

        logger.info("Transcribing audio for %s.", media_file)
        result = transcribe_with_progress(
            whisper_model,
            audio_output_path,
            language,
            f"Transcribing {media_file}",
            progress_update_interval_seconds,
            logger,
        )
        logger.info("Transcription complete for %s.", media_file)

        transcript_filename = os.path.splitext(media_file)[0] + transcript_extension
        transcript_path = os.path.join(transcripts_folder, transcript_filename)
        write_text_file(transcript_path, result["text"])
        logger.info("Transcript saved: %s", transcript_path)

        if enable_cleanup and cleanup_func:
            cleaned_filename = (
                os.path.splitext(media_file)[0] + cleaned_suffix + transcript_extension
            )
            cleaned_path = os.path.join(transcripts_folder, cleaned_filename)
            logger.info("Running cleanup for %s.", media_file)
            try:
                cleaned_text = cleanup_func(result["text"])
                write_text_file(cleaned_path, cleaned_text)
                logger.info("Cleaned transcript saved: %s", cleaned_path)
            except Exception as exc:
                handle_error(f"Cleanup failed for {media_file}", exc, logger)


def process_audio_files(
    *,
    audios_path: str,
    transcripts_folder: str,
    audio_extensions: tuple[str, ...],
    transcript_extension: str,
    cleaned_suffix: str,
    whisper_model,
    language: str,
    enable_cleanup: bool,
    cleanup_func: Callable[[str], str] | None,
    progress_update_interval_seconds: float,
    logger: logging.Logger,
) -> None:
    media_files = iter_supported_media_files(audios_path, audio_extensions)
    logger.info("Found %d supported audio files in %s.", len(media_files), audios_path)

    for media_file, media_path in media_files:
        logger.info("Processing audio: %s", media_file)
        logger.info("Transcribing audio for %s.", media_file)
        result = transcribe_with_progress(
            whisper_model,
            media_path,
            language,
            f"Transcribing {media_file}",
            progress_update_interval_seconds,
            logger,
        )
        logger.info("Transcription complete for %s.", media_file)

        transcript_filename = os.path.splitext(media_file)[0] + transcript_extension
        transcript_path = os.path.join(transcripts_folder, transcript_filename)
        write_text_file(transcript_path, result["text"])
        logger.info("Transcript saved: %s", transcript_path)

        if enable_cleanup and cleanup_func:
            cleaned_filename = (
                os.path.splitext(media_file)[0] + cleaned_suffix + transcript_extension
            )
            cleaned_path = os.path.join(transcripts_folder, cleaned_filename)
            logger.info("Running cleanup for %s.", media_file)
            try:
                cleaned_text = cleanup_func(result["text"])
                write_text_file(cleaned_path, cleaned_text)
                logger.info("Cleaned transcript saved: %s", cleaned_path)
            except Exception as exc:
                handle_error(f"Cleanup failed for {media_file}", exc, logger)
