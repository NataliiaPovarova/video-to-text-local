# Video / Audio Transcription with Local Whisper

This project transcribes media using a locally hosted Whisper model. Videos have their audio track extracted first, while audio files (`.mp3`, `.m4a`) are sent straight to transcription. Outputs are saved to `transcripts/`.

## Features

- CLI flag `--type` selects whether to process items from `videos/` or `audios/`.
- Automatic audio extraction for supported video formats (`.mp4`, `.mov`, `.avi`, `.mkv`).
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

## Configuration

Edit `params.yaml` to choose the Whisper language and model size:

```
language: ru
model: base
```
## Run with Docker (recommended)

The provided `Dockerfile` bundles all dependencies (including ffmpeg). The image uses CPU-only PyTorch wheels.

### Build the image

```
docker build -t whisper-transcriber .
```

### Run for videos

```
docker run --rm \
  -v "$(pwd)/videos:/app/videos" \
  -v "$(pwd)/audios:/app/audios" \
  -v "$(pwd)/transcripts:/app/transcripts" \
  -v "$(pwd)/params.yaml:/app/params.yaml:ro" \
  whisper-transcriber --type video
```

### Run for audios

```
docker run --rm \
  -v "$(pwd)/audios:/app/audios" \
  -v "$(pwd)/transcripts:/app/transcripts" \
  -v "$(pwd)/params.yaml:/app/params.yaml:ro" \
  whisper-transcriber --type audio
```

> Tip: mount specific files/folders as needed; transcripts directory will be created inside the container if it does not exist.

## Run without Docker (make sure ffmpeg is in PATH!)

1. **Install ffmpeg** and ensure it is on your `PATH`.
   - macOS: `brew install ffmpeg`
   - Windows (PowerShell admin): `choco install ffmpeg`

2. **Create a virtual environment** (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```
pip install -r requirements.txt
```

4. **Run the script**:
   - Process videos: `python main.py --type video`
   - Process audios: `python main.py --type audio`

Media files must be placed in their respective folders before running the command. Transcripts are written to `transcripts/`.

## Notes

- GPU acceleration is not configured inside the container; transcription will run on CPU.
- Ensure mounted directories exist locally before running the container to avoid permission issues.
- Adjust `params.yaml` (e.g., model size) based on accuracy vs. performance trade-offs. Larger models require more CPU time and memory.
# Video / Audio Transcription with Local Whisper

This project transcribes audio from video files into text using a locally run instance of OpenAI's Whisper model.

## Features

- Extracts audio from video files (`.mp4`, `.mov`, `.avi`, `.mkv`).
- Transcribes audio files directly (`.mp3`, `.m4a`).
- Select input source with `--type` flag: `video` or `audio`.
- Saves transcriptions as `.txt` files in `transcripts/`.
- Configuration is managed via `params.yaml`.

## Project Structure

```
.
├── audios/         # Stores the extracted audio files
├── main.py         # The main script to run
├── params.yaml     # Configuration for the model
├── requirements.txt# Python dependencies
├── transcripts/    # Stores the final text transcripts
└── videos/         # Place your input video files here
```

## Setup and Installation

### 1. Prerequisites: `ffmpeg`

The script relies on `ffmpeg` to process audio and video files. You must install it on your system and ensure it's available in your system's PATH.

**On Windows (recommended):**
Use the [Chocolatey](https://chocolatey.org/) package manager. Open PowerShell as an administrator and run:
```powershell
choco install ffmpeg
```

Alternatively, you can download `ffmpeg` from [their official website](https://ffmpeg.org/download.html), extract it, and manually add its `bin` directory to your system's PATH environment variable.

**On MacOS**
```bash
brew install ffmpeg
```
If you don't have Homebrew, install from [here](https://brew.sh/)

### 2. Python Environment

It is recommended to use a Python virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Python Dependencies

Install the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. GPU and PyTorch (Important)

For high-performance transcription, this script is intended to run on a CUDA-enabled NVIDIA GPU. However, this requires a compatible version of PyTorch.

- **Check Compatibility**: The script will automatically detect if a compatible GPU is available. If not, it will fall back to using the CPU, which will be significantly slower.
- **Install Correct PyTorch Version**: If you have an NVIDIA GPU, it is highly recommended to install a version of PyTorch that supports it. Go to the [official PyTorch website](https://pytorch.org/get-started/locally/) and use the tool to find the correct installation command for your system (select the latest CUDA version).

You may need to uninstall existing versions first to avoid conflicts:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 # <-- Example command, use one from the website
```

## Configuration

Configure transcription settings in `params.yaml`:

```yaml
language: ru        # Language code (e.g., 'en', 'es', 'fr', 'ru')
model: base     # Whisper model: tiny | base | small | medium | large-v3
```
Pick the model based on accuracy vs. speed trade-offs and your hardware capacity.

## Usage

### Process videos (reads from `videos/`, extracts audio to `audios/`, then transcribes)
1. Put your video inside folder `videos/`
2.
  ```bash
  python main.py --type video
  ```

### Process audios (reads `.mp3`/`.m4a` from `audios/`):
1. Put your audio inside `audios/`
2.
  ```bash
  python main.py --type audio
  ```

Supported extensions:
- Videos: `.mp4`, `.mov`, `.avi`, `.mkv`
- Audios: `.mp3`, `.m4a`

Outputs are written to `transcripts/`.
