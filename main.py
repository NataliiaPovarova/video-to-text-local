import functools
import logging
import sys
import whisper

from src.cleanup import cleanup_with_ollama
from src.file_preprocessing import process_audio_files, process_video_files
from src.utils import (
    ProcessingError,
    ensure_directories,
    ensure_ffmpeg_available,
    load_yaml_file,
    parse_cli_args,
    select_device,
    setup_logging,
)


CONFIG_PATH = "configurations/general_config.yaml"


def orchestrate() -> None:
    general_config = load_yaml_file(CONFIG_PATH)
    paths = general_config["paths"]
    files = general_config["files"]
    output = general_config["output"]
    ollama = general_config["ollama"]
    processing = general_config["processing"]
    dependencies = general_config["dependencies"]
    logging_config = general_config["logging"]

    logger = setup_logging(
        logs_dir=paths["logs"],
        level=logging_config["level"],
        file_name=logging_config["file_name"],
        log_format=logging_config["format"],
    )
    logger.info("Application started.")

    videos_path = paths["videos"]
    audios_path = paths["audios"]
    transcripts_folder = paths["transcripts"]
    params_path = files["params"]
    prompts_path = files["prompts"]
    transcript_extension = output["transcript_extension"]
    cleaned_suffix = output["cleaned_suffix"]
    extracted_audio_extension = output["extracted_audio_extension"]

    ensure_directories([audios_path, transcripts_folder], logger)
    ensure_ffmpeg_available(dependencies["ffmpeg_executable"], logger)

    args = parse_cli_args(videos_path, audios_path, cleaned_suffix, transcript_extension)
    logger.info("CLI arguments parsed: type=%s, language=%s, cleanup=%s", args.type, args.language, args.cleanup)

    params = load_yaml_file(params_path)
    prompts = load_yaml_file(prompts_path)

    language = args.language
    transcription_model_name = params["transcription_model"]
    cleanup_model_name = params["cleanup_model"]
    cleanup_prompt = prompts["cleanup_prompt"]
    if not isinstance(cleanup_prompt, str) or not cleanup_prompt.strip():
        raise ProcessingError(f"{prompts_path} must define a non-empty cleanup_prompt")

    device = select_device(logger)
    logger.info("Using device: %s", device)

    logger.info("Loading Whisper model: %s", transcription_model_name)
    whisper_model = whisper.load_model(transcription_model_name, device=device)
    logger.info("Whisper model loaded.")

    cleanup_func = None
    if args.cleanup:
        cleanup_func = functools.partial(
            cleanup_with_ollama,
            cleanup_model_name=cleanup_model_name,
            cleanup_prompt=cleanup_prompt,
            device=device,
            ollama_url=ollama["url"],
            ollama_timeout_seconds=ollama["timeout_seconds"],
            ollama_request_content_type=ollama["request_content_type"],
            logger=logger,
        )
        logger.info("Cleanup mode is enabled with model: %s", cleanup_model_name)

    if args.type == "video":
        process_video_files(
            videos_path=videos_path,
            audios_path=audios_path,
            transcripts_folder=transcripts_folder,
            video_extensions=tuple(general_config["extensions"]["video"]),
            transcript_extension=transcript_extension,
            cleaned_suffix=cleaned_suffix,
            extracted_audio_extension=extracted_audio_extension,
            whisper_model=whisper_model,
            language=language,
            enable_cleanup=args.cleanup,
            cleanup_func=cleanup_func,
            progress_update_interval_seconds=processing["progress_update_interval_seconds"],
            logger=logger,
        )
    else:
        process_audio_files(
            audios_path=audios_path,
            transcripts_folder=transcripts_folder,
            audio_extensions=tuple(general_config["extensions"]["audio"]),
            transcript_extension=transcript_extension,
            cleaned_suffix=cleaned_suffix,
            whisper_model=whisper_model,
            language=language,
            enable_cleanup=args.cleanup,
            cleanup_func=cleanup_func,
            progress_update_interval_seconds=processing["progress_update_interval_seconds"],
            logger=logger,
        )

    logger.info("All processing completed.")


if __name__ == "__main__":
    try:
        orchestrate()
    except ProcessingError as exc:
        logging.error("Processing failed: %s", exc)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - defensive
        logging.exception("Unexpected error: %s", exc)
        sys.exit(1)
