# Quick Start — Local Video/Audio Transcription

A fast path to your first transcript. For full setup (GPU/CUDA, Ollama cleanup,
diarization, Docker), see [README.md](README.md).

## One-time setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

No system `ffmpeg` is required: at startup the app auto-provisions the `ffmpeg`
bundled with its `imageio-ffmpeg` dependency (`ensure_ffmpeg_on_path` in
`src/utils/system.py`), exposing it on PATH from `~/.cache/video-to-text-local/bin`.
Install your own ffmpeg only if you prefer a specific build.

## Transcribe a file

### Easiest: the `transcribe` wrapper (macOS/Linux)

```bash
./transcribe /path/to/your/video.mp4
```

- Copies the file into `videos/` (or `audios/` for `.mp3`/`.m4a`), runs Whisper,
  and writes the transcript to `transcripts/<name>.txt`.
- The wrapper defaults to **English**. For another language:
  `./transcribe file.mp4 --language ru`
- To transcribe everything already in `videos/`: just run `./transcribe`
- Extra flags pass straight through, e.g. `./transcribe clip.mp4 --cleanup`
- It expects the virtualenv at `./venv` and an `ffmpeg` resolvable on PATH.

### Manual (cross-platform, equivalent)

```bash
source venv/bin/activate
# drop files into videos/ then:
python main.py --type video --language en
# or, for audio files placed in audios/:
python main.py --type audio --language en
```

Note: `main.py`'s own default is `--language ru` (omit `--language` for Russian).
The `transcribe` wrapper overrides that default to `en`.

## Where things are

- **Transcripts:** `transcripts/<name>.txt`
- **Logs:** `logs/transcriber.log`
- **Model:** `configurations/params.yaml` → `transcription_model`
  (`tiny | base | small | medium | large-v3` — bigger = more accurate but slower on CPU).

## Web GUI (optional)

Prefer a browser? `pip install -r requirements-gui.txt` then `python app.py`
and open http://127.0.0.1:7860 — upload, pick a model, transcribe, download.
See the [Web GUI section in README.md](README.md#web-gui-local) for the full feature list.

## Notes

- Device is auto-detected; with no NVIDIA GPU, transcription runs on **CPU**. The
  first run of a given model downloads its weights once (`large-v3` ≈ 3 GB), cached
  afterward under `~/.cache/whisper`.
- Optional `--cleanup` (needs a local **Ollama** server) and `--diarize` (needs a
  HuggingFace token in `.env`) are supported — see [README.md](README.md).
