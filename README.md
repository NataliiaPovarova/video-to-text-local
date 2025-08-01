# Video Transcription with Local Whisper

This project transcribes audio from video files into text using a locally run instance of OpenAI's Whisper model.

## Features

- Extracts audio from video files (`.mp4`, `.mov`, `.avi`, etc.).
- Transcribes the audio using a specified Whisper model.
- Saves the resulting transcriptions as `.txt` files.
- Configuration is managed via a `params.yaml` file.

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

Before running the script, you can configure the transcription settings in the `params.yaml` file:

```yaml
language: ru       # Language of the audio (e.g., 'en', 'es', 'fr', 'ru')
model: large-v3  # Whisper model to use (e.g., 'tiny', 'base', 'small', 'medium', 'large-v3')
```

## Usage

1.  Place your video files inside the `videos/` directory.
2.  Run the main script from your terminal:

    ```bash
    python main.py
    ```

The script will process each video, creating an audio file in `audios/` and a final text file in `transcripts/`. You will see progress updates printed in the console.
