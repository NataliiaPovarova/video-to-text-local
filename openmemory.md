# OpenMemory Guide

## Overview
- Entry point: `main.py`
- Main workflow: transcribe media (`video` or `audio`) with local Whisper, optional cleanup via Ollama.

## Architecture
- Runtime settings are centralized in `configurations/general_config.yaml`.
- Model settings are loaded from `configurations/params.yaml`; language is a CLI argument (`--language`, default `ru`).
- Cleanup prompt is loaded from `configurations/prompts.yaml`.
- Code is split into root orchestrator `main.py` and implementation modules inside `src/`.

## Components
- `main.py`: orchestrates configuration loading and dispatches audio/video processing.
- `src/utils.py`: reusable utilities (config loading, CLI parsing, logging setup, device/dependency checks, progress transcription wrapper, error handling).
- `src/cleanup.py`: Ollama cleanup request/response handling and cleanup-specific errors.
- `src/file_preprocessing.py`: media discovery, video audio extraction, transcript writing, and per-file processing loops.
- `configurations/general_config.yaml`: runtime constants, logging destination/options, and processing/dependency settings.

## Patterns
- Configuration-first: move environment-specific constants out of Python code and read from YAML.
- Modular processing: keep orchestration in root entrypoint and move domain logic to focused modules under `src/`.

## User Defined Namespaces
- [Leave blank - user populates]
