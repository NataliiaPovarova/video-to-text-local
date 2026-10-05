# OpenMemory Guide

## Overview
Local Whisper transcription (CLI + Gradio GUI). Shared seam: `src/service.py::transcribe_file`. Config: `configurations/general_config.yaml`. Full operating contract: `CLAUDE.md` + `docs/SYSTEM-SPEC.md`.

## Architecture
- CLI discovers files via `extensions.audio` / `extensions.video` in `general_config.yaml`.
- GUI + seam classify uploads via hardcoded `_AUDIO_EXTS` / `_VIDEO_EXTS` in `src/service.py` and matching `file_types` in `app.py`. The `transcribe` wrapper has its own extension case. Keep all four lists in sync when adding formats.
- Decode path is ffmpeg (Whisper `load_audio`); any format ffmpeg can decode can be added as an audio extension without extra codec code.

## Components
- `configurations/general_config.yaml` — CLI discovery extensions (audio currently: `.mp3`, `.m4a`, `.aac`).
- `src/service.py` — `_AUDIO_EXTS` / `_VIDEO_EXTS` for input classification at the CLI+GUI seam.
- `app.py` — Gradio `file_types` allowlist; inline preview limited to `.mp3`/`.m4a` (browser-playable). Raw `.aac` is accepted for transcription but not previewed.
- `transcribe` — bash wrapper extension case (`mp3|m4a|aac` → `audios/`).

## Patterns
- **Add an audio format:** update (1) `general_config.yaml` `extensions.audio`, (2) `src/service.py` `_AUDIO_EXTS`, (3) `app.py` `file_types`, (4) the `transcribe` wrapper case, (5) bilingual READMEs + `QUICKSTART.md`, (6) test fixture + a seam test. Preview allowlist only if browsers can play the container.

## User Defined Namespaces
- [Leave blank - user populates]
