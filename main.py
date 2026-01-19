import whisper
from moviepy import VideoFileClip
import os
import yaml
import torch
import argparse
import shutil
import sys
import threading
import time
from tqdm import tqdm

VIDEOS_PATH = "videos"
AUDIOS_PATH = "audios"
TRANSCRIPTS_FOLDER = "transcripts"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
AUDIO_EXTENSIONS = (".mp3", ".m4a")

# Create directories if they don't exist
os.makedirs(AUDIOS_PATH, exist_ok=True)
os.makedirs(TRANSCRIPTS_FOLDER, exist_ok=True)

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

#parameters
language = params["language"]
model_name = params["model"]

# --- Device Checking ---

def _select_device() -> str:
    """
    Prefer CUDA when available and emit actionable diagnostics otherwise.
    """
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"CUDA is available. Using GPU: {gpu_name}")
        except Exception as exc:
            print(f"CUDA initialization warning: {exc}")
        return "cuda"

    diagnostics: list[str] = []
    try:
        if hasattr(torch.backends, "cuda") and not torch.backends.cuda.is_built():
            diagnostics.append("Current PyTorch build lacks CUDA support.")
    except Exception as exc:  # pragma: no cover - defensive
        diagnostics.append(f"Could not query torch.backends.cuda: {exc}")

    cuda_version = getattr(torch.version, "cuda", None)
    if not cuda_version:
        diagnostics.append(
            "torch.version.cuda is None. Install GPU-enabled wheels, "
            "for example:\n"
            "  pip install --upgrade torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu121"
        )
    else:
        diagnostics.append(f"PyTorch reports CUDA runtime {cuda_version}.")

    try:
        gpu_count = torch.cuda.device_count()
        diagnostics.append(f"Detected CUDA devices: {gpu_count}")
        if gpu_count == 0:
            diagnostics.append(
                "No CUDA-capable GPUs detected. Ensure NVIDIA drivers are installed "
                "and `nvidia-smi` works."
            )
    except Exception as exc:  # pragma: no cover - defensive
        diagnostics.append(f"Could not enumerate CUDA devices: {exc}")

    print("CUDA not available, defaulting to CPU.")
    if diagnostics:
        print("Diagnostics:")
        for line in diagnostics:
            print(f" - {line}")
    return "cpu"


device = _select_device()
print(f"Using device: {device}")
# ---------------------

# --- CLI args ---
parser = argparse.ArgumentParser(description="Transcribe videos or audios using local Whisper")
parser.add_argument(
    "--type",
    choices=["video", "audio"],
    required=True,
    help="Input type: 'video' to read from videos/ and extract audio; 'audio' to read from audios/",
)
args = parser.parse_args()

# --- Dependencies check ---
def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        print(
            "ERROR: 'ffmpeg' was not found on PATH. Whisper relies on ffmpeg to decode audio.\n"
            "Install ffmpeg and restart your terminal/IDE. On Windows with Chocolatey:\n"
            "  choco install ffmpeg\n"
            "Or download from https://ffmpeg.org/download.html and add its 'bin' folder to PATH."
        )
        sys.exit(1)

ensure_ffmpeg_available()

print("Loading whisper model...")
model = whisper.load_model(model_name, device=device)
print("Model loaded.")

def _get_audio_duration_seconds(audio_path: str) -> float | None:
    """
    Try to determine audio duration using MoviePy. Returns None if unavailable.
    """
    try:
        # Prefer top-level import (MoviePy v2 style)
        from moviepy import AudioFileClip  # type: ignore
        with AudioFileClip(audio_path) as clip:
            return float(clip.duration) if clip.duration else None
    except Exception:
        try:
            # Fallback for MoviePy v1 style
            from moviepy.editor import AudioFileClip as EditorAudioFileClip  # type: ignore
            with EditorAudioFileClip(audio_path) as clip:
                return float(clip.duration) if clip.duration else None
        except Exception:
            return None

def transcribe_with_progress(model_obj, audio_path: str, language: str, description: str):
    """
    Display a progress bar based on wall-clock time vs. media duration
    while Whisper is running. This is an estimate (there is no callback API).
    """
    total_seconds = _get_audio_duration_seconds(audio_path)
    stop_event = threading.Event()
    pbar: tqdm | None = None

    def _run_progress_bar():
        if total_seconds is None or total_seconds <= 0:
            return  # no progress bar if we can't estimate duration
        nonlocal pbar
        pbar = tqdm(total=int(total_seconds), desc=description, unit="s")
        start_time = time.time()
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            # Clamp to total
            current = int(min(elapsed, pbar.total))
            if current != pbar.n:
                pbar.n = current
                pbar.refresh()
            time.sleep(0.25)
        # complete the bar
        pbar.n = pbar.total
        pbar.refresh()
        pbar.close()

    thread: threading.Thread | None = None
    if total_seconds and total_seconds > 0:
        thread = threading.Thread(target=_run_progress_bar, daemon=True)
        thread.start()

    try:
        return model_obj.transcribe(audio_path, language=language, fp16=False)
    finally:
        stop_event.set()
        if thread:
            thread.join(timeout=0.5)

if args.type == "video":
    # Process each video file in the videos folder
    for media_file in os.listdir(VIDEOS_PATH):
        media_path = os.path.join(VIDEOS_PATH, media_file)
        if not os.path.isfile(media_path):
            continue

        extension = os.path.splitext(media_file)[1].lower()
        if extension not in VIDEO_EXTENSIONS:
            continue

        print(f"Processing video {media_file}...")
        audio_output_path = os.path.join(
            AUDIOS_PATH, os.path.splitext(media_file)[0] + ".mp3"
        )
        print(f"Extracting audio from {media_file}...")
        with VideoFileClip(media_path) as video:
            if video.audio is None:
                print(f"No audio track found in {media_file}. Skipping file.")
                continue
            video.audio.write_audiofile(audio_output_path)
        print("Audio extracted.")

        print(f"Transcribing audio for {media_file}...")
        result = transcribe_with_progress(model, audio_output_path, language, f"Transcribing {media_file}")
        print("Transcription complete.")

        transcript_filename = os.path.splitext(media_file)[0] + ".txt"
        with open(os.path.join(TRANSCRIPTS_FOLDER, transcript_filename), "w", encoding='utf-8') as f:
            f.write(result['text'])

        print(f"Transcript for {media_file} saved to {transcript_filename}")

elif args.type == "audio":
    # Process each audio file in the audios folder
    for media_file in os.listdir(AUDIOS_PATH):
        media_path = os.path.join(AUDIOS_PATH, media_file)
        if not os.path.isfile(media_path):
            continue

        extension = os.path.splitext(media_file)[1].lower()
        if extension not in AUDIO_EXTENSIONS:
            continue

        print(f"Processing audio {media_file}...")
        print(f"Transcribing audio for {media_file}...")
        result = transcribe_with_progress(model, media_path, language, f"Transcribing {media_file}")
        print("Transcription complete.")

        transcript_filename = os.path.splitext(media_file)[0] + ".txt"
        with open(os.path.join(TRANSCRIPTS_FOLDER, transcript_filename), "w", encoding='utf-8') as f:
            f.write(result['text'])

        print(f"Transcript for {media_file} saved to {transcript_filename}")

print("All processing completed.")
