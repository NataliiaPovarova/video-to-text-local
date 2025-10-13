import whisper
from moviepy import VideoFileClip
import os
import yaml
import torch

VIDEOS_PATH = "videos"
AUDIOS_PATH = "audios"
TRANSCRIPTS_FOLDER = "transcripts"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
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
device = "cpu"
if torch.cuda.is_available():
    print("CUDA is available, checking for compatibility...")
    try:
        # This will raise a warning if not compatible, and might error on some setups.
        torch.cuda.get_device_name(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    except Exception as e:
        print(f"Could not use GPU due to an error: {e}")
        print("Falling back to CPU.")
else:
    print("CUDA not available, using CPU.")

print(f"Using device: {device}")
# ---------------------

print("Loading whisper model...")
model = whisper.load_model(model_name, device=device)
print("Model loaded.")

# Process each media file in the videos folder
for media_file in os.listdir(VIDEOS_PATH):
    media_path = os.path.join(VIDEOS_PATH, media_file)
    if not os.path.isfile(media_path):
        continue

    extension = os.path.splitext(media_file)[1].lower()
    audio_source_path = None

    if extension in VIDEO_EXTENSIONS:
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
        audio_source_path = audio_output_path
        print("Audio extracted.")
    elif extension in AUDIO_EXTENSIONS:
        print(f"Processing audio {media_file}...")
        audio_source_path = media_path
    else:
        continue

    print(f"Transcribing audio for {media_file}...")
    # NOTE: The warning about GPU incompatibility comes from PyTorch, not directly from this call.
    # If device is 'cpu', this will be slow.
    result = model.transcribe(audio_source_path, language=language, fp16=False)
    print("Transcription complete.")

    transcript_filename = os.path.splitext(media_file)[0] + ".txt"
    with open(os.path.join(TRANSCRIPTS_FOLDER, transcript_filename), "w", encoding='utf-8') as f:
        f.write(result['text'])

    print(f"Transcript for {media_file} saved to {transcript_filename}")

print("All media files have been transcribed.")
