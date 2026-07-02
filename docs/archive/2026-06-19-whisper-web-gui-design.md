> **Archived 2026-06-21 by `/curate`.** The web GUI this spec designs has shipped:
> `app.py`, `src/service.py`, `src/history.py`, and the `src/pipeline/` +
> `src/transcription/` modules are now authoritative. Kept for its design rationale
> (rejected alternatives, dependency constraints). Historical — do not implement from
> it; the Status, branch/commit, and test counts reflect the original spec, not the
> current state.

# Design Spec — Local Web GUI for Whisper Transcription

- **Date:** 2026-06-19
- **Status:** Approved — review changes folded in (verified numpy/pillow resolution, stem
  sanitization, fix-branch stability)
- **Branch:** `feat/web-gui`, based on `fix/whisper-trailing-hallucination` (commit
  `c15937f`) — **confirmed stable and ready to merge**: full suite (41 tests) green, working
  tree clean. The GUI relies on that branch's `condition_on_previous_text=False` fix for
  clean output.
- **Author:** brainstorming session (Claude + user)

## 1. Context & goal

The repo transcribes local video/audio to text with a locally-hosted Whisper model,
plus optional Ollama cleanup and optional pyannote diarization. Today it is **CLI +
folder-batch only**: drop files in `videos/`/`audios/`, run `python main.py --type video
--language ru`, read `transcripts/<name>.txt`.

**Goal:** add a local, single-user, zero-cost **web GUI** so a user can upload a file,
pick the model + language, click Transcribe, watch live progress, read/download the
result (txt + srt/vtt), and browse/search a history of past transcriptions they can
re-download or delete from — **without breaking the existing CLI**.

**Stack:** Gradio. Reason: built for exactly this (upload, dropdowns, long-running jobs
with queue-based live progress, file download) in minimal local code.

## 2. Constraints & corrections to the original brief

The original brief's "Key files" list predates the **"Refactor/restructure for scaling"**
commit and is stale. Confirmed facts from reading the current code:

- **Whisper library: `openai-whisper` 20250625** (not faster-whisper). Progress is a
  **duration-based wall-clock estimate** already implemented in
  `src/transcription/asr_engine.py` (a ticker thread comparing elapsed vs. audio
  duration). We reuse it; we do not get true decoder callbacks from openai-whisper.
- **Per-segment timestamps already exist.** `TranscriptSegment` (`src/models/document.py`)
  carries `start_time`, `end_time`, `speaker`, `confidence`. SRT/VTT need **no** pipeline
  change to "capture" timestamps — contrary to the brief's guess.
- **Actual module layout (reused, not reimplemented):**
  - `main.py` — orchestrator entrypoint
  - `src/ingestion/` — `discover_media_files`, `extract_audio_from_video`
  - `src/transcription/asr_engine.py` — `transcribe_audio`
  - `src/processing/cleanup.py` — `cleanup_with_ollama` (raises `ProcessingError` on failure)
  - `src/output/formatter.py` — `write_transcript`, `format_document_with_speakers`
  - `src/pipeline/` — `PipelineOrchestrator` + step classes
  - `src/models/document.py` — `PipelineContext`, `TranscriptDocument`, `TranscriptSegment`
  - `src/transcription/diarization_config.py` — `resolve_hf_token` (raises if `HF_TOKEN` absent)
- **Python 3.9.6** in the venv ⇒ **Gradio 4.x** (`gradio>=4.44,<5`); Gradio 5 needs Python
  ≥3.10. Decision: stay on 3.9 + Gradio 4.x (zero disruption to the working venv).
  **Verified** (`pip install --dry-run 'gradio>=4.44,<5'`, no install): resolves to
  **gradio 4.44.1** and leaves **numpy 2.0.2, torch, and openai-whisper untouched** (none
  appear in the resolver's install set). See §11 for the hardened dependency rule + the one
  pillow caveat.
- **History store: JSON index file** (not SQLite) — single-user, zero new deps.
- **Scope: full, phased core-first** (Phase 1 shippable, Phase 2 adds the rest).

### Non-goals

- No cloud/paid services; no React+separate-backend SPA.
- No multi-user/auth; binds `127.0.0.1` only.
- No persistence of uploaded **inputs** (history tracks outputs + metadata; media playback
  applies to the current upload).

## 3. Architecture

```
                    ┌─────────────────────────────────────┐
                    │  src/service.py  (NEW — the seam)    │
  app.py (Gradio) ─▶│  transcribe_file(path, *, model,     │◀─ main.py (CLI)
  src/history.py ◀──│    language, cleanup, diarize, ...,  │   loops over discovered
  (JSON index)      │    progress_callback) -> Result      │   files, calls it per file
                    │  + get_whisper_model(name)  (1 slot) │
                    └──────────────┬──────────────────────┘
                                   │ builds & runs the existing pipeline
                    ┌──────────────▼──────────────────────┐
                    │  EXISTING (reused):                  │
                    │  ingestion → TranscriptionStep →     │
                    │  [Diarization] → [Cleanup] → Output  │
                    └──────────────────────────────────────┘
```

**The seam:** a single in-process function `transcribe_file(...)` builds and runs the
existing step-pipeline for **one** file. Both the CLI (`main.py`) and GUI (`app.py`) call
it. Per-job `model` / `language` / `cleanup` / `diarize` / `num_speakers` are **function
arguments — `configurations/params.yaml` is NEVER mutated at runtime.** This removes the
global-config footgun for repeated/queued jobs.

`transcribe_file` returns a `TranscribeResult` dataclass: output paths (`txt`, optional
`clean`, `srt`, `vtt`), document metadata (id, language, duration, word count, segment
count, model used, options), and an `errors`/`warnings` list (e.g. cleanup-skipped).

### 3.1 Single-slot model cache (folded-in refinement #1)

`get_whisper_model(name, device)` holds **at most one** resident Whisper model:

```
_resident: (name, model) | None     # module-level, lock-guarded
get_whisper_model(name, device):
    if resident and resident.name == name: return resident.model   # reuse, no reload
    evict resident: drop reference; gc.collect(); torch.cuda.empty_cache() if cuda
    load model; set resident = (name, model); return model
```

Rationale: jobs run **one at a time** (queue `concurrency_limit=1`), so only one model is
ever needed concurrently. Evicting the previous model on an **actual model change** means
comparing e.g. `base` vs `large-v3` of the same clip never stacks ~5 GB and OOMs. A reload
(~tens of seconds for large-v3) is paid **only on a real model change**; same-model jobs
and CLI batches (one model for all files) reuse the resident model with zero reload.

### 3.2 Progress callback

`transcribe_audio` gains an optional `progress_callback: Callable[[float], None]`. The
existing duration ticker thread calls it with `min(elapsed/total, 1.0)`. The CLI keeps its
tqdm bar; the GUI passes a callback that updates `gr.Progress`. No behavior change for the CLI.

## 4. Modules

| File | Change |
|------|--------|
| `src/service.py` | **NEW.** `get_whisper_model` (1-slot cache), cached diarization backend, `transcribe_file(...) -> TranscribeResult`. |
| `src/output/formatter.py` | **ADD** `format_srt/format_vtt/write_srt/write_vtt` from segment timestamps (+ speaker labels when diarized). |
| `src/history.py` | **NEW.** `HistoryStore(transcripts/.history.json)`: `add/list/get/delete/search`; atomic writes (temp+rename); lock-guarded. |
| `app.py` | **NEW.** Gradio UI; `python app.py` → `127.0.0.1`, opens browser. |
| `src/transcription/asr_engine.py` | **ADD** optional `progress_callback`. |
| `src/pipeline/steps_output.py` | **ADD** optional output-basename override (CLI passes none → unchanged `<stem>`; GUI passes collision-safe basename). |
| `main.py` | Refactor `orchestrate()` to call `transcribe_file` per discovered file. **CLI behavior identical.** |
| `requirements-gui.txt` | **NEW.** `gradio>=4.44,<5`. Core install stays lean. |

## 5. The GUI (`app.py`)

- **Transcribe tab:** upload (validates `mp4/mov/avi/mkv/webm`, `mp3/m4a`) · model dropdown
  (default = `params.yaml`, speed/accuracy hint, "downloads ~3 GB on first use" notice when
  weights aren't cached under `~/.cache/whisper`) · language dropdown/input incl.
  **Auto-detect** (passes no language) · cleanup + diarize checkboxes (diarize disabled with
  a note when `HF_TOKEN` is absent) · **Transcribe** button · live progress bar · raw
  transcript box · cleaned transcript in a separate tab · download buttons (txt + srt/vtt;
  clean when present) · media player + copy-to-clipboard.
- **History tab:** searchable table (by filename/date) with columns model · language ·
  duration · word count · date · options · status; row select → view / re-download / delete.
  Read from the JSON index — never inferred from filenames.
- **Responsiveness:** `demo.queue(default_concurrency_limit=1)` → one transcription at a
  time; extra jobs queue; the server/UI never freezes. Progress driven by §3.2.

## 6. Output naming & coexistence (folded-in refinement #3)

**Stem sanitization (folded-in).** Before building the output name, the uploaded file's stem
is sanitized: Unicode-normalize (NFC); keep readable Unicode/Arabic where filesystem-safe;
replace path separators and OS-reserved/illegal characters (`/ \ : * ? " < > |`, ASCII
control chars) and collapse runs of whitespace to `_`; strip leading/trailing dots and
spaces; truncate the stem to a safe length (≤80 chars, preserving uniqueness via the id
suffix). If the result is empty (e.g. a name that was entirely illegal characters), fall back
to the job id. This runs for the GUI path only; the CLI keeps the OS-provided stem.

GUI output basename: **`<sanitized-stem>__<model>`** (e.g. `walkthrough__base.srt`,
`walkthrough__large-v3.srt`). If that basename already exists (same source + same model run
again), append a short unique suffix from the job id: `<sanitized-stem>__<model>__<id8>`.

**Confirmed:** this lets **multiple transcriptions of the same source file coexist without
overwriting** — `base` vs `large-v3` of one video produce distinct files
(`walkthrough__base.txt` and `walkthrough__large-v3.txt`), and even two same-model runs
coexist via the id suffix. Each `HistoryStore` entry records the exact final paths, so
re-download/delete always target the right files. **The CLI keeps plain `<stem>.txt`**
(unchanged) by passing no override to `OutputStep`.

## 7. Error handling

| Failure | Behavior |
|---------|----------|
| Unsupported upload extension | Validated up front; clear UI error; no job created |
| **Unreadable / corrupt / unsupported-codec / no-audio-track** (refinement #2) | Ingestion/decode raises a **distinct `MediaDecodeError`** (subclass of `ProcessingError`); UI shows a specific message — *"Could not read/decode '\<file\>': it may be corrupt, truncated, missing an audio track, or use an unsupported codec."* — **NOT** the generic "unexpected error" bucket; job recorded `failed`. |
| Model download fails (network) | Caught; UI error; job `failed` in history |
| **Ollama down** (cleanup) | `ProcessingError` caught; **raw transcript still saved/shown**; cleaned tab shows "cleanup unavailable: \<msg\>"; job succeeds with a warning |
| **No HF token** (diarize) | Toggle greyed with explanation; if enabled-but-fails, caught, raw transcript kept |
| OOM / other unexpected | Caught; UI shows error + **tail of `logs/transcriber.log`**; job `failed` |

Distinct error taxonomy: `MediaDecodeError` (read/decode), Ollama-unavailable (warning, not
fatal), diarization-unavailable (warning/greyed), model-download failure, generic. The GUI
maps each to its own message; only truly unclassified exceptions hit the generic bucket.

## 8. History store

`transcripts/.history.json` (already gitignored via `transcripts/`). One record per job:

```
id, source_filename, model, language, options{cleanup, diarize, num_speakers},
duration_seconds, word_count, created_at(UTC ISO), status(success|failed|warning),
outputs{txt, clean?, srt, vtt}, message?(error/warning text)
```

Writes are atomic (temp file + `os.replace`) and lock-guarded; safe because jobs are
serialized. `search(query)` filters in Python over filename/date (instant at single-user
scale). `delete(id)` removes the record and its output files.

## 9. Testing

- `tests/test_service.py` — `transcribe_file` with a **mocked** Whisper model: asserts
  txt/srt/vtt written, returned metadata correct, **`params.yaml` untouched**, single-slot
  cache evicts on model change & reuses on same model, cleanup/diarize toggles wire through.
- `tests/test_sanitize_stem.py` — stem sanitization unit tests: **an Arabic/Unicode filename**
  (e.g. `محاضرة الفيزياء.mp4`) sanitizes to a filesystem-safe, non-empty, readable stem that
  round-trips through the output-naming path; plus spaces, OS-reserved/illegal chars
  (`/ \ : * ? " < > |`), an over-long name (truncated to ≤80 chars), and an all-illegal name
  (falls back to the job id). Asserts the final `<stem>__<model>` path is writable and unique.
- `tests/test_history.py` — add/list/get/delete/search round-trip on a temp JSON; delete
  removes files.
- `tests/test_formatter_srt_vtt.py` — SRT/VTT timestamp formatting from known segments
  (incl. speaker labels).
- **Existing 41 tests stay green.** CLI smoke path unchanged.

## 10. Phasing

**Phase 1 — core, shippable:**
`transcribe_file` + single-slot model cache; **stem sanitization + collision-safe output
naming** (§6); `main.py` refactor (CLI unchanged); `app.py` with upload+validation,
model/language(+auto-detect), Transcribe, live progress, raw result, txt download; JSON
`HistoryStore` (list/open/download/delete); one-at-a-time queue; graceful Ollama handling;
`MediaDecodeError`; service + history + sanitization tests; `requirements-gui.txt`; docs
(README EN/RU + CLAUDE.md + `python app.py` one-liner).

**Phase 2 — added (top-down priority):**
srt/vtt export & downloads; cleanup + diarize toggles with graceful degradation; batch
upload (queue multiple); history search + richer metadata; media playback + copy-to-clipboard;
error + log-tail surfacing in the UI; srt/vtt tests.

## 11. Risks / open notes

- **Dependency rule (hardened, verified).** The GUI install MUST NOT downgrade `numpy`,
  `torch`, or `openai-whisper` — the ASR stack runs on `numpy 2.0.2` and a downgrade can
  break it. Verified via `pip install --dry-run 'gradio>=4.44,<5'`: it resolves to
  **gradio 4.44.1** with **numpy/torch/whisper unchanged** (none appear in the resolver's
  install set). If any future gradio pin would force `numpy<2`, pick a different gradio 4.x
  that accepts numpy 2 — never move numpy. **Do not assume; re-run the dry-run whenever the
  pin changes.**
  - **Caveat — pillow:** gradio 4.x pins `pillow<11`, so installing it downgrades
    `pillow 11.3.0 → 10.4.0`. Pillow is used only by `moviepy` (frame I/O), NOT by the ASR
    stack, and 10.4.0 is within the repo's existing `pillow<12.0` allowance. No gradio 4.x
    avoids this (4.44.1 is the final 4.x). **Install-time gate:** after installing the GUI
    deps, confirm the ASR stack still imports and the existing 41 tests pass; if the pillow
    downgrade breaks anything, isolate gradio in a dedicated venv rather than moving it back.
- openai-whisper progress is an **estimate** (no true decoder callback); acceptable and
  matches current CLI behavior.
- Past-entry **media playback** is out of scope (inputs not persisted); transcript
  view/download/delete from history is in scope.

## 12. Done means

One command (`python app.py`) opens localhost; user uploads a file, picks model + language,
clicks Transcribe, watches progress without the page freezing, reads & downloads the result
(txt + srt/vtt), and sees it in a searchable history they can re-download or delete from —
**with the existing CLI working unchanged** and **all existing tests green**.
