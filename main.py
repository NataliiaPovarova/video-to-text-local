import functools
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from the project's .env file before any module
# reads os.environ. Existing OS-level variables take precedence (override=False
# by default), so CI/CD and shell exports still win over a local .env.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

import whisper

from src.ingestion import discover_media_files
from src.models import PipelineContext
from src.pipeline import (
    AudioIngestionStep,
    CleanupStep,
    DiarizationStep,
    OutputStep,
    PipelineOrchestrator,
    TranscriptionStep,
    VideoIngestionStep,
)
from src.transcription.diarization_config import load_diarization_config
from src.transcription.diarizer import create_diarization_backend
from src.processing import cleanup_with_ollama
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
    diarization_path = files["diarization"]
    transcript_extension = output["transcript_extension"]
    cleaned_suffix = output["cleaned_suffix"]
    extracted_audio_extension = output["extracted_audio_extension"]

    ensure_directories([audios_path, transcripts_folder], logger)
    ensure_ffmpeg_available(dependencies["ffmpeg_executable"], logger)

    args = parse_cli_args(videos_path, audios_path, cleaned_suffix, transcript_extension)
    logger.info(
        "CLI arguments parsed: type=%s, language=%s, cleanup=%s, diarize=%s, num_speakers=%s",
        args.type,
        args.language,
        args.cleanup,
        args.diarize,
        args.num_speakers,
    )

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

    # --- Build pipeline steps ---
    transcripts_path = Path(transcripts_folder)
    audios_path_obj = Path(audios_path)

    if args.type == "video":
        ingestion_step = VideoIngestionStep(audios_path_obj, extracted_audio_extension)
        source_folder = videos_path
        extensions = tuple(general_config["extensions"]["video"])
    else:
        ingestion_step = AudioIngestionStep()
        source_folder = audios_path
        extensions = tuple(general_config["extensions"]["audio"])

    transcription_step = TranscriptionStep(
        whisper_model=whisper_model,
        progress_update_interval=processing["progress_update_interval_seconds"],
    )

    diarization_config = load_diarization_config(diarization_path)
    if args.no_diarize:
        diarization_config.enabled = False
    elif args.diarize:
        diarization_config.enabled = True

    output_step = OutputStep(
        transcripts_folder=transcripts_path,
        transcript_extension=transcript_extension,
        cleaned_suffix=cleaned_suffix,
    )

    steps = [ingestion_step, transcription_step]

    if diarization_config.enabled:
        diarization_backend = create_diarization_backend(
            diarization_config,
            device,
            logger,
        )
        diarization_work_dir = Path(audios_path) / ".diarization_cache"
        steps.append(
            DiarizationStep(
                backend=diarization_backend,
                config=diarization_config,
                work_dir=diarization_work_dir,
                ffmpeg_executable=dependencies["ffmpeg_executable"],
                num_speakers_override=args.num_speakers,
            )
        )
        logger.info("Diarization mode is enabled (backend=%s)", diarization_config.backend)

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
        steps.append(CleanupStep(cleanup_func))
        logger.info("Cleanup mode is enabled with model: %s", cleanup_model_name)

    steps.append(output_step)

    pipeline = PipelineOrchestrator(steps=steps, logger=logger)

    # --- Discover and process files ---
    media_files = discover_media_files(source_folder, extensions)
    logger.info("Found %d supported files in %s.", len(media_files), source_folder)

    for filename, file_path in media_files:
        context = PipelineContext(
            source_path=file_path,
            input_type=args.type,
            language=language,
        )
        pipeline.run(context)

    logger.info("All processing completed.")


if __name__ == "__main__":
    try:
        orchestrate()
    except ProcessingError as exc:
        logging.error("Processing failed: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logging.exception("Unexpected error: %s", exc)
        sys.exit(1)
