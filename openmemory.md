# OpenMemory Guide

## Overview
- Entry point: `main.py`
- Main workflow: transcribe media (`video` or `audio`) with local Whisper, optional cleanup via Ollama.
- Architecture: modular pipeline with step-based orchestration.

## Architecture
- **Pipeline pattern**: each file is processed through a sequence of `PipelineStep` instances managed by `PipelineOrchestrator`.
- **Data model**: `PipelineContext` carries state between steps; `TranscriptDocument` holds structured transcript with segments, speakers, timestamps.
- Runtime settings are centralized in `configurations/general_config.yaml`.
- Model settings are loaded from `configurations/params.yaml`; language is a CLI argument (`--language`, default `ru`).
- Cleanup prompt is loaded from `configurations/prompts.yaml`.
- Secrets (e.g. `HF_TOKEN`) live in a local `.env` at the project root, loaded by `python-dotenv` at the top of `main.py` via `load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")`. `.env` is gitignored; `.env.example` is the committed template. Existing OS env vars override the `.env`.
- Documentation is bilingual and split across two files at the project root: `README.md` (English, default — what GitHub renders) and `README.ru.md` (Russian). Each has a language switcher at the top linking to the other. When editing one, update the other to keep them in sync.

## Project Structure
```
main.py                      # CLI entry point, builds and runs pipeline
src/
├── __init__.py              # Package docstring
├── models/                  # Data models (dataclasses)
│   ├── __init__.py
│   └── document.py          # TranscriptSegment, Speaker, TranscriptDocument, PipelineContext
├── pipeline/                # Pipeline orchestration
│   ├── __init__.py
│   ├── orchestrator.py      # PipelineOrchestrator
│   ├── steps.py             # PipelineStep ABC
│   ├── steps_ingestion.py   # VideoIngestionStep, AudioIngestionStep
│   ├── steps_transcription.py # TranscriptionStep
│   ├── steps_cleanup.py     # CleanupStep
│   └── steps_output.py      # OutputStep
├── ingestion/               # File discovery and media extraction
│   ├── __init__.py
│   ├── file_loader.py       # discover_media_files()
│   └── video_extractor.py   # extract_audio_from_video()
├── transcription/           # ASR engines
│   ├── __init__.py
│   └── asr_engine.py        # transcribe_audio() -- Whisper wrapper
├── processing/              # Post-transcription processing
│   ├── __init__.py
│   └── cleanup.py           # cleanup_with_ollama()
├── output/                  # Formatting and file writing
│   ├── __init__.py
│   └── formatter.py         # write_transcript(), write_text_file()
└── utils/                   # Shared utilities
    ├── __init__.py           # Re-exports for convenience
    ├── errors.py             # ProcessingError
    ├── config.py             # load_yaml_file()
    ├── cli.py                # parse_cli_args()
    ├── device.py             # select_device()
    ├── logging_setup.py      # setup_logging()
    └── system.py             # ensure_directories(), ensure_ffmpeg_available()
tests/                       # pytest test suite
configurations/              # YAML configuration files
```

## Components
- `main.py`: orchestrates configuration loading, builds pipeline steps, discovers files, and runs each through the pipeline.
- `src/models/document.py`: core data model -- `TranscriptSegment` (with text, timestamps, speaker, confidence), `TranscriptDocument` (collection of segments with metadata), `PipelineContext` (carries state between steps).
- `src/pipeline/orchestrator.py`: iterates through steps, handles errors, logs progress.
- `src/pipeline/steps.py`: abstract base class defining the step interface (`execute`, `should_skip`).
- `src/pipeline/steps_*.py`: concrete pipeline steps for each stage.
- `src/ingestion/`: file discovery (sorted, extension-filtered) and video-to-audio extraction.
- `src/transcription/asr_engine.py`: Whisper transcription with progress bar, returns structured `TranscriptDocument` preserving per-segment timestamps.
- `src/processing/cleanup.py`: Ollama HTTP cleanup with timeout/error handling.
- `src/output/formatter.py`: writes transcript documents and raw text to disk.
- `src/utils/`: configuration loading, CLI parsing, device selection, logging, system checks.

## Patterns
- **Configuration-first**: move environment-specific constants out of Python code and read from YAML.
- **Secrets via `.env`**: per-developer secrets (tokens, keys) loaded from `.env` at startup; never committed. Template lives in `.env.example`. Code reads them via `os.environ.get(...)` so existing OS env vars and CI/CD still take precedence.
- **Modular pipeline**: each processing stage is a self-contained `PipelineStep` that can be added/removed/reordered.
- **Structured data flow**: `PipelineContext` → steps enrich it → final output. No loose variables passed between stages.
- **Local-first**: all models run locally by default (Whisper + Ollama). Cloud APIs to be added as alternative backends in future phases.
- **Extensibility**: new steps (diarization, summarization, integrations) plug in by implementing `PipelineStep` and adding to the step list.
- **In-memory audio for pyannote**: `PyannoteBackend.diarize` hands pyannote a `{"waveform": Tensor, "sample_rate": int, "uri": stem}` mapping instead of a file path. This makes pyannote's `Audio.validate_file` take its in-memory branch and never call `torchcodec`. The decode helpers are `PyannoteBackend._load_in_memory` (logging + tensor packaging) and `PyannoteBackend._decode_audio` (format-aware reader). Fast path: when the input is the cached 16 kHz mono WAV produced by `ensure_wav_16k_mono`, scipy.io.wavfile reads it in <1 s even for multi-hour recordings — no extra ffmpeg subprocess. Fallback for other formats: `whisper.audio.load_audio` (ffmpeg subprocess). `DiarizationBackend.diarize` accepts an optional `logger` so the in-memory load can emit a "Loaded audio for diarization (...)" heartbeat with format and timing.
- **Real prep-bar for diarization**: `diarize_document` in `src/transcription/diarizer.py` wraps `ensure_wav_16k_mono` with the project's `duration_progress_bar` (driven by the source audio duration computed up-front via `_get_audio_duration_seconds`). This replaces the placeholder `tqdm(total=1, unit="step")` that used to just flash on completion and gives visible movement during the 30–60 s ffmpeg conversion on first run (cached WAV runs still finish instantly).

## Toolchain constraints
- `pyannote.audio >= 4.0` pulls `torchcodec` as a transitive dependency. On Windows `torchcodec`'s native extension frequently fails to load (FFmpeg shared-library ABI mismatches across minor versions — e.g. `torchcodec 0.14.0`'s `libtorchcodec_core8.dll` against FFmpeg 8.1.1 fails with `WinError 127: ERROR_PROC_NOT_FOUND`; also PyTorch C++ ABI drift, e.g. `torch_parse_device_string` missing when `torchcodec 0.14.0` is paired with `torch 2.9.1+cu128`). pyannote swallows the underlying `ImportError` into a warning and then crashes later with `NameError: name 'AudioDecoder' is not defined`. We sidestep this entirely by passing pyannote pre-decoded audio (see Patterns / `PyannoteBackend._load_in_memory`). The `torchcodec is not installed correctly` warning at startup is expected and harmless.
- The same broken native load on Windows can also surface as a modal "the procedure entry point ... could not be located in the DLL ..." dialog from the Windows DLL loader, which would block batch runs until the user clicks OK. `main.py` calls `ctypes.windll.kernel32.SetErrorMode(0x0001)` (SEM_FAILCRITICALERRORS) at the very top of the file — before any import that could pull in `torch` / `pyannote.audio` / `torchcodec` — to suppress that popup for the process. The text warning on stderr is preserved.
- `choco install ffmpeg` installs the **essentials** (static) build, not the shared build. The `ffmpeg-shared` choco package is separate (and unofficial). Since the project no longer requires the shared FFmpeg DLLs, the essentials build is fine.

## User Defined Namespaces
- [Leave blank - user populates]
