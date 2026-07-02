# Local Whisper Web GUI — SYSTEM-SPEC
> The SINGLE canonical "what the system IS right now." One living spec only — never a second. Refresh against HEAD when it drifts. This describes current reality, not plans (plans live in archived spec/plan docs) and not history (that's BUILD-STATE/DECISIONS).

> **verified-against** `cd888bf` · 2026-06-21 — a current-state doc; reconfirm against code and restamp via `/curate` when it drifts.

## What it is

A local, single-user Gradio web GUI over an existing Whisper transcription pipeline. Upload a media file → pick model + language (with auto-detect) → transcribe with live progress → read/download the transcript → browse a searchable history (open/download/delete). Binds `127.0.0.1` only; no auth, no cloud, no paid deps. Phase 1 (core) has landed; Phase 2 (srt/vtt export, batch upload, search, media preview) is in progress.

## Stack & surfaces

- **Stack:** Python 3.9.6 · openai-whisper 20250625 · Gradio 4.44.1 (`gradio>=4.44,<5`, `huggingface_hub<1.0`) · moviepy · pyannote.audio (optional diarization) · Ollama (optional cleanup) · pytest.
- **Entry points / surfaces:**
  - `python main.py --type {video,audio} [--language X] [--cleanup] [--diarize|--no-diarize] [--num-speakers N]` — the CLI (unchanged behavior).
  - `python app.py` — the Gradio GUI on `http://127.0.0.1:7860` (`app.py:281`).
  - `src/service.py::transcribe_file(...)` — **the seam**: the single in-process function both surfaces call to transcribe one file with per-job parameters.
- **Data model (no DB):**
  - `src/models/document.py` — `TranscriptSegment`, `TranscriptDocument` (segments + metadata), `PipelineContext` (carries state between steps, incl. `exception`).
  - History persistence — `transcripts/.history.json`, a JSON index owned by `src/history.py::HistoryStore` (atomic writes, lock-guarded). Not SQLite (single-user, zero new deps).
- **External integrations (all optional / local):** Whisper model weights (local); pyannote.audio diarization (needs an HF token, env-driven); Ollama cleanup (local LLM). Missing any of these degrades gracefully with a warning — never a crash.
- **Config:** `configurations/general_config.yaml` (paths, logging, file refs), `configurations/params.yaml` (default model — **never mutated at runtime**), `configurations/prompts.yaml`, `configurations/diarization.yaml`. Secrets in a gitignored `.env` (`.env.example` is the template).

## Invariants (authoritative list — mirrors CLAUDE.md §5; see AUDIT.md for test status)

- INV-1 — CLI parity (`main.py:48-135`).
- INV-2 — `params.yaml` never mutated at runtime (`src/service.py:125-141`).
- INV-3 — single resident Whisper model, evict-before-load (`src/service.py:40-74`).
- INV-4 — one transcription at a time, `queue(default_concurrency_limit=1)` (`app.py:280`). *(PENDING test.)*
- INV-5 — local-only `127.0.0.1` bind, no auth/cloud (`app.py:281`). *(PENDING test.)*
- INV-6 — Unicode/Arabic-safe, collision-safe filenames (`src/utils/naming.py`; applied `src/service.py:191-192`).
- INV-7 — crash-safe, non-ASCII-preserving history index (`src/history.py:33,46-48`).
- INV-8 — auto-detect records the real language, never blank (`src/transcription/asr_engine.py:137`).
- INV-9 — GUI install never downgrades numpy/torch/whisper (`requirements-gui.txt`). *(PENDING automated test.)*
- INV-10 — a failing file never aborts a batch (`app.py:165-179`, `main.py`).

## Known open issues at this HEAD (`cd888bf`)

1. ~~**Non-portable Windows test (finding #1).**~~ **RESOLVED — C-001** (branch `fix/windows-ffmpeg-test-guard`). `tests/test_utils.py::TestEnsureFfmpegOnPath::test_windows_uses_exe_name_and_copy_fallback` is now guarded with `@pytest.mark.skipif(sys.platform != "win32", ...)`. The predicate is evaluated at collection time against the real host platform (before the in-body `sys.platform` monkeypatch), so the test still RUNS on the win32 target and skips cleanly off-Windows — no more `_winapi` crash under Python ≥3.12. Off-Windows suite is now 0 failed. `skipif` was chosen over a `_winapi` mock because `sys.platform` is version-stable whereas the private `_winapi` surface varies by Python version. See DECISIONS D-002 (resolution) + BUILD-STATE `[C-001]`.
2. **No automated test gate (CI).** `.github/workflows/release.yml` runs only semantic-release on `master`; `package.json`'s `npm test` is a no-op (`echo ... exit 1`). The chunk loop assumes a green baseline, but nothing runs `pytest` automatically. Recommend a minimal pytest CI workflow. (D-002)
3. **INV-4 / INV-5 unprotected.** Queue concurrency=1 and the `127.0.0.1` bind have no named test (`launch()`/`queue()` aren't exercised). Tracked PENDING in AUDIT — usually a first post-adoption chunk.
4. **INV-9 has no automated guard.** The no-downgrade rule is enforced only by a manual `pip install --dry-run`. A CI step asserting numpy/torch/whisper versions after a GUI install would close it.
5. **No linter/typecheck configured.** No ruff/flake8/mypy/pre-commit. Style/type drift is uncaught. Optional to add; out of scope until decided. (D-002)
6. **`.gitignore` hygiene.** It ignored `docs/` and listed `CLAUDE.md` three times plus dead Windows-backslash paths; the groundwork adoption un-ignores the contract docs and removes the dead dupes (D-003). Verify nothing relied on `docs/` being ignored.

## Test baseline (executed during adoption)

Full stack installed in the adoption sandbox (Linux, Python 3.12): **114 passed, 1 failed** (the failure = known issue #1 above), 0 collection errors. On the project's Windows + Python 3.9 target the suite is green (the repo's own verified `requirements-gui.txt` note + the green-gated curate at HEAD). Command: `python -m pytest -q`. *(Issue #1 resolved by C-001 on branch `fix/windows-ffmpeg-test-guard`: the test is now platform-guarded, so an off-Windows re-run yields 0 failed — 109 passed, 2 skipped on a Python-3.9 executor without `gradio`.)*
