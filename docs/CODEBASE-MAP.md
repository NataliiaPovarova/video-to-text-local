# Local Whisper Web GUI — CODEBASE-MAP
> Where things live. Keep it short and current — a navigation aid, not a spec. Update when structure changes.

> **verified-against** `cd888bf` · 2026-06-21 — current-state doc; restamp via `/curate` when it drifts.

## Layout

- `main.py` — CLI entry point. `orchestrate()` discovers files and calls the seam per file (`main.py:135`).
- `app.py` — Gradio Blocks GUI (run with `python app.py`). Single-file UI; includes a gradio_client bool-schema compatibility shim (top of file) and the batch runner.
- `src/service.py` — **the seam.** `transcribe_file(...)`, `get_whisper_model` (1-slot cache), diarization-backend cache, `reset_caches()` (test helper).
- `src/history.py` — `HistoryEntry` + `HistoryStore` (atomic, lock-guarded JSON index at `transcripts/.history.json`).
- `src/utils/naming.py` — `sanitize_stem()` + `build_output_basename()` (Unicode/Arabic + collision safety). Pure functions.
- `src/models/document.py` — `TranscriptSegment`, `TranscriptDocument`, `PipelineContext` (incl. `.exception`).
- `src/pipeline/` — the reused step pipeline:
  - `orchestrator.py` — runs steps, **captures the aborting exception onto `context.exception`** (`:35-36`).
  - `steps_ingestion.py` (raises `MediaDecodeError`), `steps_transcription.py` (threads `progress_callback`), `steps_diarization.py`, `steps_cleanup.py`, `steps_output.py` (optional `output_basename` override).
- `src/transcription/` — `asr_engine.py` (Whisper wrapper; captures detected language at `:137`), `diarizer.py`, `alignment.py`, `audio_prep.py`, `diarization_backends/pyannote_backend.py` (imports torch + pyannote).
- `src/ingestion/` — `file_loader.py` (discovery), `video_extractor.py` (moviepy audio extract).
- `src/processing/cleanup.py` — optional Ollama cleanup.
- `src/output/formatter.py` — txt writer; srt/vtt formatters (Phase 2).
- `src/utils/` — `errors.py` (`ProcessingError`, `MediaDecodeError`), `config.py`, `cli.py`, `device.py` (imports torch), `logging_setup.py`, `system.py`, `progress.py`.
- `configurations/` — `general_config.yaml`, `params.yaml`, `prompts.yaml`, `diarization.yaml`.
- `tests/` — 20 files, ~115 test fns. `conftest.py` carries a speechbrain lazy-import guard for full-suite stability.
- `docs/` — the groundwork doc set (this file et al.); `docs/archive/` holds superseded specs/plans + the pre-groundwork CLAUDE.md + openmemory.md.
- `.github/workflows/release.yml` — semantic-release on `master` only (no test CI).

## Hot paths (the code most changes touch)

- The seam `src/service.py::transcribe_file` — every transcription (CLI or GUI) flows through here.
- The GUI handlers in `app.py` (`_run_transcription`, `_run_batch`, history list/open/download/delete).
- The output naming/writing path (`naming.py` → `steps_output.py`).

## The chokepoints (shared accessors — fix defects HERE, not at call sites)

- **`src/service.py::transcribe_file`** — the one place per-job params, model caching, and output naming are wired. CLI + GUI parity lives or dies here.
- **`src/service.py::get_whisper_model`** — the single-slot model cache (evict-before-load).
- **`src/history.py::HistoryStore`** — the only writer/reader of the history index (atomic + locked).
- **`src/pipeline/orchestrator.py`** — the single place a step exception is captured onto the context (enables `MediaDecodeError` surfacing in the GUI).
- **`src/utils/naming.py`** — the only place filenames are sanitized + de-collided.

## Where the invariants are enforced

- INV-1 → `main.py:48-135`  ·  INV-2 → `src/service.py:125-141`  ·  INV-3 → `src/service.py:40-74`
- INV-4 → `app.py:280`  ·  INV-5 → `app.py:281`  ·  INV-6 → `src/utils/naming.py` (+ `src/service.py:191-192`)
- INV-7 → `src/history.py:33,46-48`  ·  INV-8 → `src/transcription/asr_engine.py:137`
- INV-9 → `requirements-gui.txt`  ·  INV-10 → `app.py:165-179`, `main.py`
