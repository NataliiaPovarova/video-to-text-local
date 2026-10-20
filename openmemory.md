# OpenMemory Guide

## Overview
Local Whisper transcription (CLI + Gradio GUI). Shared seam: `src/service.py::transcribe_file`. Config: `configurations/general_config.yaml`. Full operating contract: `CLAUDE.md` + `docs/SYSTEM-SPEC.md`.

## Architecture
- Input gate (C-006, D-012): the seam (`src/service.py::transcribe_file`) probes each file with `src/utils/ffmpeg_tools.py::probe_media` and accepts it when ffmpeg can read it and finds an audio stream; video vs audio comes from the probe (cover art is not video). No hardcoded extension lists in code.
- One extension list — `extensions.audio` / `extensions.video` in `general_config.yaml` — is used for CLI folder discovery, the GUI `file_types` picker, and nothing else. The `transcribe` wrapper classifies via `python -m src.utils.ffmpeg_tools classify <file>`.
- ffmpeg is the only media tool (moviepy removed); `imageio-ffmpeg` supplies a fallback binary.

## Components
- `configurations/general_config.yaml` — the single extension list (CLI discovery + GUI picker).
- `src/utils/ffmpeg_tools.py` — all ffmpeg calls: `probe_media`/`MediaInfo`, `classify_media`, `extract_audio`, `decode_to_array`.
- `app.py` — Gradio `file_types` from the config list; inline preview limited to `.mp3`/`.m4a` (browser-playable).
- `transcribe` — bash wrapper; classifies via `ffmpeg_tools classify`.

## Patterns
- **Add a format for discovery:** add the extension to `general_config.yaml` `extensions` and the bilingual READMEs. Nothing else — acceptance is by probe. Preview allowlist in `app.py` only if browsers can play the container.

## User Defined Namespaces
- [Leave blank - user populates]
