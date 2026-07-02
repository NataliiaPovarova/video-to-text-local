import sys

if sys.platform == "win32":
    # Suppress Windows DLL loader modal dialogs (e.g. "the procedure entry
    # point ... could not be located in the dynamic link library ...") so
    # benign import-time failures inside transitive dependencies such as
    # torchcodec do not interrupt batch runs with a popup that requires a
    # click. The failures still propagate as Python exceptions and are
    # surfaced via the pyannote stderr warning; only the modal UI is
    # suppressed. SEM_FAILCRITICALERRORS = 0x0001.
    # This MUST run before any import that may load torch / pyannote.audio /
    # torchcodec (so before `whisper`, `src.pipeline`, etc.), otherwise the
    # dialog has already fired by the time we get here.
    import ctypes

    ctypes.windll.kernel32.SetErrorMode(0x0001)

import logging
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from the project's .env file before any module
# reads os.environ. Existing OS-level variables take precedence (override=False
# by default), so CI/CD and shell exports still win over a local .env.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

import whisper  # noqa: F401  (kept so model weights resolve identically; optional)

from src.ingestion import discover_media_files
from src.service import transcribe_file
from src.transcription.diarization_config import load_diarization_config
from src.utils import (
    MediaDecodeError,
    ProcessingError,
    ensure_directories,
    ensure_ffmpeg_available,
    ensure_ffmpeg_on_path,
    load_yaml_file,
    parse_cli_args,
    setup_logging,
)


CONFIG_PATH = "configurations/general_config.yaml"


def orchestrate() -> None:
    general_config = load_yaml_file(CONFIG_PATH)
    paths = general_config["paths"]
    files = general_config["files"]
    output = general_config["output"]
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
    cleaned_suffix = output["cleaned_suffix"]
    transcript_extension = output["transcript_extension"]

    ensure_directories([audios_path, transcripts_folder], logger)
    # Provision the bundled imageio-ffmpeg binary when no system ffmpeg exists,
    # so the pre-flight check below passes without a separate install.
    ensure_ffmpeg_on_path(logger)
    ensure_ffmpeg_available(dependencies["ffmpeg_executable"], logger)

    args = parse_cli_args(videos_path, audios_path, cleaned_suffix, transcript_extension)
    logger.info(
        "CLI arguments parsed: type=%s, language=%s, cleanup=%s, diarize=%s, num_speakers=%s",
        args.type, args.language, args.cleanup, args.diarize, args.num_speakers,
    )

    # Resolve diarization on/off from config + flags (CLI semantics preserved).
    diarization_config = load_diarization_config(files["diarization"])
    if args.no_diarize:
        diarize = False
    elif args.diarize:
        diarize = True
    else:
        diarize = diarization_config.enabled

    params = load_yaml_file(files["params"])
    model_name = params["transcription_model"]

    if args.type == "video":
        source_folder = videos_path
        extensions = tuple(general_config["extensions"]["video"])
    else:
        source_folder = audios_path
        extensions = tuple(general_config["extensions"]["audio"])

    media_files = discover_media_files(source_folder, extensions)
    logger.info("Found %d supported files in %s.", len(media_files), source_folder)

    _process_batch(
        media_files,
        model_name=model_name,
        language=args.language,
        cleanup=args.cleanup,
        diarize=diarize,
        num_speakers=args.num_speakers,
        logger=logger,
    )

    logger.info("All processing completed.")


def _process_batch(
    media_files,
    *,
    model_name: str,
    language: str,
    cleanup: bool,
    diarize: bool,
    num_speakers,
    logger,
) -> None:
    """Transcribe each discovered file, continuing past any single-file failure.

    A corrupt/no-audio file raises MediaDecodeError (re-raised by the service);
    defensively, any other exception is caught too. We log it and move on so one
    bad file never kills the batch (matches Task 4's CLI-parity note).
    """
    for _filename, file_path in media_files:
        try:
            result = transcribe_file(
                file_path,
                model=model_name,
                language=language,
                cleanup=cleanup,
                diarize=diarize,
                num_speakers=num_speakers,
                config_path=CONFIG_PATH,
                logger=logger,
            )
        except MediaDecodeError as exc:
            logger.error("Skipping %s: %s", file_path.name, exc)
            continue
        except Exception as exc:  # one bad file must not kill the batch
            logger.exception("Unexpected error on %s: %s", file_path.name, exc)
            continue
        if result.status == "failed":
            logger.error("Failed: %s (%s)", file_path.name, result.message)


if __name__ == "__main__":
    try:
        orchestrate()
    except ProcessingError as exc:
        logging.error("Processing failed: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logging.exception("Unexpected error: %s", exc)
        sys.exit(1)
