# Video / Audio Transcription with Local Whisper

This project transcribes media using a locally hosted Whisper model. Videos have their audio track extracted first, while audio files (`.mp3`, `.m4a`) are sent straight to transcription. Outputs are saved to `transcripts/`.

## Features

- CLI flag `--type` selects whether to process items from `videos/` or `audios/`.
- Automatic audio extraction for supported video formats (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`).
- Progress indication during transcription (duration-based estimate).
- Language and model configuration via `params.yaml`.

## Project Layout

```
.
├── audios/         # Input audio files or extracted audio
├── videos/         # Input videos when running with --type video
├── transcripts/    # Transcription outputs (created automatically)
├── params.yaml     # Whisper language/model settings
├── requirements.txt
├── main.py         # Entry point
├── Dockerfile      # Container image definition
└── .dockerignore
```

## Setup and Installation (Local)

### 1. Prerequisites: `ffmpeg`

The script relies on `ffmpeg` to decode audio. Ensure it is on your PATH.

- **Windows (Recommended)**:
  Run in PowerShell as Administrator:
  ```powershell
  choco install ffmpeg
  ```
- **macOS**:
  ```bash
  brew install ffmpeg
  ```

### 2. Python Environment

Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

**Important for NVIDIA GPU Users:**
To use CUDA acceleration, you must install a PyTorch version compatible with your GPU **before** installing other requirements.

**For RTX 50 Series (Blackwell) or newer:**
You need CUDA 12.8+ support.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**For older GPUs (RTX 30/40 series):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only (or after installing PyTorch):**
Install the rest of the dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `params.yaml` to choose the Whisper language and model size:

```yaml
language: ru    # e.g. 'en', 'ru'
model: base     # tiny | base | small | medium | large-v3
```

## Usage

### Process Videos
1. Place video files in `videos/`.
2. Run:
   ```bash
   python main.py --type video
   ```

### Process Audio
1. Place audio files in `audios/`.
2. Run:
   ```bash
   python main.py --type audio
   ```

**Supported extensions:**
- Videos: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`
- Audio: `.mp3`, `.m4a`

## Run with Docker

The provided `Dockerfile` bundles all dependencies. Note that default Docker setup uses CPU.

```bash
# Build
docker build -t whisper-transcriber .

# Run (example for video)
docker run --rm \
  -v "$(pwd)/videos:/app/videos" \
  -v "$(pwd)/audios:/app/audios" \
  -v "$(pwd)/transcripts:/app/transcripts" \
  -v "$(pwd)/params.yaml:/app/params.yaml:ro" \
  whisper-transcriber --type video
```
