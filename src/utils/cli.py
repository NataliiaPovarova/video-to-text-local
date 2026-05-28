import argparse


def parse_cli_args(
    videos_path: str,
    audios_path: str,
    cleaned_suffix: str,
    transcript_extension: str,
):
    parser = argparse.ArgumentParser(description="Transcribe videos or audios using local Whisper")
    parser.add_argument(
        "--type",
        choices=["video", "audio"],
        required=True,
        help=(
            f"Input type: 'video' to read from {videos_path}/ and extract audio; "
            f"'audio' to read from {audios_path}/"
        ),
    )
    parser.add_argument(
        "--language",
        default="ru",
        help="Language code for Whisper transcription (default: %(default)s)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help=(
            "Run optional cleanup via local Ollama and save "
            f"{cleaned_suffix}{transcript_extension}"
        ),
    )
    return parser.parse_args()
