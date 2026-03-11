# Video / Audio Transcription with Local Whisper

This project transcribes media using a locally hosted Whisper model, then optionally cleans the transcript with a local Ollama model. Videos have their audio track extracted first, while audio files (`.mp3`, `.m4a`) are sent straight to transcription. Outputs are saved to `transcripts/`, and execution logs are written to `logs/`.

## Features

- CLI flag `--type` selects whether to process items from `videos/` or `audios/`.
- Automatic audio extraction for supported video formats (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`).
- Optional cleanup step via local Ollama using `cleanup_model` and `cleanup_prompt`.
- Progress indication during transcription (duration-based estimate).
- Language and model configuration via `configurations/params.yaml`.
- Runtime paths and processing options via `configurations/general_config.yaml`.
- Structured codebase with a root orchestrator (`main.py`) and domain modules in `src/`.

## Project Layout

```
.
├── audios/                     # Input audio files or extracted audio
├── videos/                     # Input videos when running with --type video
├── transcripts/                # Transcription outputs (created automatically)
├── logs/                       # Runtime logs
├── configurations/
│   ├── general_config.yaml     # Runtime paths, extensions, logging, and service settings
│   ├── params.yaml             # Whisper language/model settings
│   └── prompts.yaml            # Cleanup prompt for Ollama
├── src/
│   ├── cleanup.py              # Ollama cleanup logic
│   ├── file_preprocessing.py   # Media discovery/extraction/transcript file operations
│   └── utils.py                # Reusable utilities and shared error handling
├── requirements.txt
├── main.py                     # Root orchestrator entrypoint
├── Dockerfile                  # Container image definition
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

### 2. Prerequisites: Ollama

The cleanup step runs locally via Ollama. Install it and pull your cleanup model:

```bash
ollama pull gpt-oss:latest
```

When CUDA is available, the script asks Ollama to use GPU for cleanup.
Please note, that some models, while being local, require authentication.

### 3. Python Environment

Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

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

Edit `configurations/params.yaml` to choose the Whisper transcription model and cleanup model:

```yaml
language: ru    # e.g. 'en', 'ru'
transcription_model: base     # tiny | base | small | medium | large-v3
cleanup_model: llama3.1:8b    # Ollama model name
```

Edit `configurations/prompts.yaml` to customize cleanup instructions:

```yaml
cleanup_prompt: |
  The text you receive is a transcription of a video / audio file. Please clean it up while preserving the meaningful information:
  - remove meaningless phrases like "oh, wait", "um", "got it", etc.;
  - split the text into paragraphs for easier reading;
  - if a part of the text contains a lot of meaningless artifacts, which might appear due to the poor quality of the sound, mark it as "[unclear]" and leave as is.
```

Edit `configurations/general_config.yaml` to control:
- input/output paths (`videos`, `audios`, `transcripts`, `logs`);
- supported media extensions;
- output naming (`.txt`, `_clean`, extracted audio extension);
- Ollama endpoint and timeout;
- logging level, log filename, and log format.

By default, logs are written to `logs/transcriber.log`.

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

### Optional Cleanup

Add `--cleanup` to generate cleaned transcripts via Ollama:

```bash
python main.py --type video --cleanup
```

**Supported extensions:**
- Videos: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`
- Audio: `.mp3`, `.m4a`

**Outputs:**
- Raw transcript: `<name>.txt`
- Cleaned transcript: `<name>_clean.txt` (via Ollama, only with `--cleanup`)
- Logs: `logs/transcriber.log`

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
  -v "$(pwd)/configurations:/app/configurations:ro" \
  whisper-transcriber --type video
```
