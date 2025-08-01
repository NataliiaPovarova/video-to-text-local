import whisper
from moviepy import VideoFileClip
import os
import yaml
import torch

VIDEOS_PATH = "videos"
AUDIOS_PATH = "audios"
TRANSCRIPTS_FOLDER = "transcripts"

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

# for each video in the videos folder
for video_file in os.listdir(VIDEOS_PATH):
    if not video_file.endswith((".mp4", ".mov", ".avi", ".mkv")):
        continue

    print(f"Processing {video_file}...")
    video_path = os.path.join(VIDEOS_PATH, video_file)
    audio_path = os.path.join(AUDIOS_PATH, os.path.splitext(video_file)[0] + ".mp3")
    
    # extract audio from video
    print(f"Extracting audio from {video_file}...")
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    print("Audio extracted.")
    
    #transcribe audio
    print(f"Transcribing audio for {video_file}...")
    # NOTE: The warning about GPU incompatibility comes from PyTorch, not directly from this call.
    # If device is 'cpu', this will be slow.
    result = model.transcribe(audio_path, language=language, fp16=False)
    print("Transcription complete.")

    transcript_filename = os.path.splitext(video_file)[0] + ".txt"
    with open(os.path.join(TRANSCRIPTS_FOLDER, transcript_filename), "w", encoding='utf-8') as f:
        f.write(result['text'])

    print(f"Transcript for {video_file} saved to {transcript_filename}")

print("All videos have been transcribed.")
