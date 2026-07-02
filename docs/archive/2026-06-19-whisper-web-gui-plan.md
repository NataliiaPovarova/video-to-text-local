> **Archived 2026-06-21 by `/curate`.** This implementation plan is complete and
> shipped; the code in `app.py`, `src/service.py`, `src/history.py`,
> `src/utils/naming.py`, and `src/pipeline/`/`src/transcription/` is authoritative.
> Kept for historical context. Do not implement from it — the unchecked checkboxes,
> embedded source copies, and test counts reflect the original plan, not current state.

# Local Whisper Web GUI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local, single-user, zero-cost Gradio web GUI for uploading a media file, picking model + language, transcribing with live progress, viewing/downloading the result (txt + srt/vtt), and browsing a searchable history — without breaking the existing CLI.

**Architecture:** Introduce one in-process seam, `src/service.py::transcribe_file(...)`, that builds and runs the **existing** step-pipeline for a single file with per-job parameters (model/language/cleanup/diarize passed as arguments — `params.yaml` is never mutated). Both the CLI (`main.py`) and the GUI (`app.py`) call it. A single-slot Whisper-model cache and a cached diarization backend live in the service so batch/queued jobs never reload. History is a JSON index (`transcripts/.history.json`) owned by the GUI.

**Tech Stack:** Python 3.9.6, openai-whisper 20250625, Gradio 4.x (`gradio>=4.44,<5`), moviepy, pyannote.audio (optional), Ollama (optional), pytest.

## Global Constraints

Every task implicitly includes these (verbatim from the spec):

- **Python 3.9.6 / Gradio 4.x only.** `gradio>=4.44,<5` (Gradio 5 needs Python ≥3.10). Use 3.9-compatible syntax: keep the existing `from __future__ import annotations` idiom in modules that use `X | Y` annotations at runtime.
- **Dependency rule (hardened):** the GUI install MUST NOT downgrade `numpy`, `torch`, or `openai-whisper`. Verified: `pip install --dry-run 'gradio>=4.44,<5'` resolves to **gradio 4.44.1** leaving numpy 2.0.2 / torch / whisper untouched (only pillow 11.3.0→10.4.0, which is fine — pillow is moviepy-only and within the repo's `pillow<12.0` allowance). Re-run the dry-run if the pin ever changes; if a future gradio needs `numpy<2`, switch gradio, not numpy.
- **`configurations/params.yaml` is NEVER mutated at runtime.** Per-job model/language/cleanup/diarize/num_speakers are function arguments.
- **CLI behavior must stay identical.** `python main.py --type {video,audio} [--language X] [--cleanup] [--diarize|--no-diarize] [--num-speakers N]` produces the same `transcripts/<stem>.txt` (and `<stem>_clean.txt`) as today.
- **All existing 41 tests stay green.** New tests are additive.
- **Local only:** GUI binds `127.0.0.1`; no cloud, no auth, zero new paid deps.
- **One transcription at a time:** `demo.queue(default_concurrency_limit=1)`.
- **Bilingual docs:** any user-facing doc change updates `README.md` AND `README.ru.md`.

---

## File Structure

**New files:**
- `src/utils/naming.py` — stem sanitization + collision-safe output basename. Pure functions, no I/O except an injected `exists` callback.
- `src/service.py` — the seam: `get_whisper_model` (1-slot cache), `get_diarization_backend` (cache), `transcribe_file(...) -> TranscribeResult`, `reset_caches()` (test helper).
- `src/history.py` — `HistoryEntry` dataclass + `HistoryStore` (atomic, lock-guarded JSON index).
- `app.py` — Gradio Blocks UI (top-level, run with `python app.py`).
- `requirements-gui.txt` — `gradio>=4.44,<5`.
- Tests: `tests/test_naming.py`, `tests/test_service.py`, `tests/test_history.py`, `tests/test_media_decode_error.py`, `tests/test_formatter_srt_vtt.py`.

**Modified files:**
- `src/utils/errors.py` — add `MediaDecodeError(ProcessingError)`.
- `src/utils/__init__.py` — export `MediaDecodeError`.
- `src/models/document.py` — add `PipelineContext.exception` field.
- `src/pipeline/orchestrator.py` — capture the raising exception onto `context.exception`.
- `src/pipeline/steps_ingestion.py` — raise `MediaDecodeError` on undecodable/no-audio media.
- `src/transcription/asr_engine.py` — optional `progress_callback`; capture detected language when `language is None`.
- `src/pipeline/steps_transcription.py` — thread `progress_callback`.
- `src/pipeline/steps_output.py` — optional `output_basename` override + records written paths; (Phase 2) optional srt/vtt.
- `src/output/formatter.py` — (Phase 2) `format_srt/format_vtt/write_srt/write_vtt`.
- `main.py` — `orchestrate()` calls `transcribe_file` per discovered file.
- `README.md`, `README.ru.md`, `CLAUDE.md`, `QUICKSTART.md` — GUI usage docs.

---

# PHASE 1 — core, shippable

## Task 1: `MediaDecodeError` taxonomy + orchestrator exception capture

**Files:**
- Modify: `src/utils/errors.py`
- Modify: `src/utils/__init__.py`
- Modify: `src/models/document.py:66-82` (PipelineContext)
- Modify: `src/pipeline/orchestrator.py:33-38`
- Test: `tests/test_media_decode_error.py` (create)

**Interfaces:**
- Produces: `MediaDecodeError(ProcessingError)`; `PipelineContext.exception: Exception | None` (set by the orchestrator to the exception that aborted the run, or `None`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_media_decode_error.py`:

```python
import logging
from pathlib import Path

from src.models import PipelineContext
from src.pipeline import PipelineOrchestrator, PipelineStep
from src.utils import MediaDecodeError, ProcessingError


class RaisingStep(PipelineStep):
    def __init__(self, exc: Exception):
        self._exc = exc

    def execute(self, context, logger):
        raise self._exc


def _ctx():
    return PipelineContext(source_path=Path("/tmp/x.mp3"), input_type="audio")


def test_media_decode_error_is_processing_error():
    assert issubclass(MediaDecodeError, ProcessingError)


def test_orchestrator_records_exception_on_context():
    logger = logging.getLogger("test")
    exc = MediaDecodeError("bad file")
    pipeline = PipelineOrchestrator(steps=[RaisingStep(exc)], logger=logger)
    ctx = pipeline.run(_ctx())
    assert ctx.exception is exc
    assert ctx.errors  # string error still recorded


def test_orchestrator_exception_none_on_success():
    logger = logging.getLogger("test")
    pipeline = PipelineOrchestrator(steps=[], logger=logger)
    ctx = pipeline.run(_ctx())
    assert ctx.exception is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_media_decode_error.py -v`
Expected: FAIL — `ImportError: cannot import name 'MediaDecodeError'` (and `PipelineContext` has no `exception`).

- [ ] **Step 3: Implement**

`src/utils/errors.py` — replace entire file:

```python
class ProcessingError(RuntimeError):
    """Raised when processing cannot continue safely."""


class MediaDecodeError(ProcessingError):
    """Raised when an input media file cannot be read or decoded.

    Covers corrupt/truncated files, a missing audio track, and unsupported
    codecs. Distinct from the generic error bucket so the UI can show a
    specific, actionable message.
    """
```

`src/utils/__init__.py` — add the import and export:

```python
from .cli import parse_cli_args
from .config import load_yaml_file
from .device import select_device
from .errors import MediaDecodeError, ProcessingError
from .logging_setup import setup_logging
from .system import ensure_directories, ensure_ffmpeg_available

__all__ = [
    "MediaDecodeError",
    "ProcessingError",
    "ensure_directories",
    "ensure_ffmpeg_available",
    "load_yaml_file",
    "parse_cli_args",
    "select_device",
    "setup_logging",
]
```

`src/models/document.py` — add the field to `PipelineContext` (after `errors`):

```python
@dataclass
class PipelineContext:
    """Carries state through the processing pipeline."""

    source_path: Path
    input_type: str  # "video" | "audio"
    language: str = "ru"
    audio_path: Path | None = None
    document: TranscriptDocument | None = None
    cleaned_text: str | None = None
    errors: list[str] = field(default_factory=list)
    exception: Exception | None = None

    def fail(self, message: str) -> None:
        self.errors.append(message)
        if self.document:
            self.document.pipeline_state = PipelineState.FAILED
```

`src/pipeline/orchestrator.py` — set `context.exception` in the except block (lines 33-38):

```python
            self._logger.info("Executing step: %s", step.name)
            try:
                context = step.execute(context, self._logger)
            except Exception as exc:
                context.exception = exc
                context.fail(f"Step '{step.name}' failed: {exc}")
                self._logger.exception("Step '%s' failed: %s", step.name, exc)
                break
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_media_decode_error.py tests/test_pipeline.py tests/test_models.py -v`
Expected: PASS (new file green; existing pipeline/model tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/utils/errors.py src/utils/__init__.py src/models/document.py src/pipeline/orchestrator.py tests/test_media_decode_error.py
git commit -m "feat: add MediaDecodeError and capture aborting exception on context"
```

---

## Task 2: Stem sanitization + collision-safe naming (`src/utils/naming.py`)

**Files:**
- Create: `src/utils/naming.py`
- Test: `tests/test_naming.py` (create)

**Interfaces:**
- Produces:
  - `sanitize_stem(name: str, *, fallback: str = "transcript", max_len: int = 80) -> str`
  - `build_output_basename(stem: str, model: str, *, exists: Callable[[str], bool], job_id: str) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_naming.py`:

```python
from src.utils.naming import build_output_basename, sanitize_stem


class TestSanitizeStem:
    def test_plain_name_unchanged(self):
        assert sanitize_stem("walkthrough") == "walkthrough"

    def test_spaces_become_underscores(self):
        assert sanitize_stem("my lecture file") == "my_lecture_file"

    def test_arabic_unicode_preserved_and_safe(self):
        out = sanitize_stem("محاضرة الفيزياء")
        assert out  # non-empty
        assert " " not in out
        # readable Arabic letters survive (the space between words is collapsed)
        assert "محاضرة" in out

    def test_illegal_characters_replaced(self):
        out = sanitize_stem('a/b\\c:d*e?f"g<h>i|j')
        for ch in '/\\:*?"<>|':
            assert ch not in out

    def test_overlong_truncated(self):
        out = sanitize_stem("x" * 500, max_len=80)
        assert len(out) <= 80

    def test_all_illegal_falls_back_to_id(self):
        assert sanitize_stem("///:::", fallback="job123") == "job123"

    def test_strips_leading_trailing_dots_and_spaces(self):
        assert sanitize_stem("  ..name..  ") == "name"

    def test_windows_reserved_device_name_prefixed(self):
        # Windows treats CON (with OR without an extension) as the CON device,
        # so the bare root name must be escaped with a leading "_".
        assert sanitize_stem("CON.mp4") == "_CON.mp4"
        assert sanitize_stem("con") == "_con"  # case-insensitive
        assert sanitize_stem("LPT9") == "_LPT9"

    def test_all_dots_with_extension_stays_non_empty(self):
        # A name that is only dots/extension must never sanitize to "" --
        # it falls back to the document id (here the default "transcript").
        out = sanitize_stem("....mp4")
        assert out  # non-empty


class TestBuildOutputBasename:
    def test_no_collision_returns_plain(self):
        out = build_output_basename("clip", "base", exists=lambda b: False, job_id="abcd1234ef")
        assert out == "clip__base"

    def test_collision_appends_job_suffix(self):
        out = build_output_basename("clip", "base", exists=lambda b: b == "clip__base", job_id="abcd1234ef")
        assert out == "clip__base__abcd1234"

    def test_different_models_distinct(self):
        a = build_output_basename("clip", "base", exists=lambda b: False, job_id="x")
        b = build_output_basename("clip", "large-v3", exists=lambda b: False, job_id="x")
        assert a != b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_naming.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.utils.naming'`.

- [ ] **Step 3: Implement**

Create `src/utils/naming.py`:

```python
from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

# Path separators, OS-reserved/illegal filename characters, and ASCII control
# chars. Covers the Windows-reserved set plus POSIX separators.
_ILLEGAL = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")
_UNDERSCORES = re.compile(r"_+")

# Windows reserved device names. A file whose root name (the part before the
# first ".") matches one of these is unusable on Windows REGARDLESS of any
# extension -- "CON.mp4" is still the CON device. The user's platform is
# Windows, so guard these. Compared case-insensitively.
_RESERVED_NAMES = (
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def sanitize_stem(name: str, *, fallback: str = "transcript", max_len: int = 80) -> str:
    """Normalize an uploaded file's stem into a filesystem-safe, readable stem.

    - NFC-normalizes Unicode (keeps readable Arabic/CJK where filesystem-safe)
    - replaces path separators / OS-reserved / control chars with "_"
    - collapses whitespace runs to "_" and repeated "_" to a single "_"
    - strips leading/trailing dots, spaces, and underscores
    - truncates to ``max_len`` characters
    - prefixes "_" when the root name is a Windows reserved device name
      (CON, PRN, AUX, NUL, COM1-9, LPT1-9), so the file is writable on Windows
    - returns ``fallback`` when nothing safe remains. Callers pass the document
      id as ``fallback`` (it defaults to "transcript" only when no id is given),
      so an all-dots / all-illegal name still yields a usable, unique stem.
    """
    stem = unicodedata.normalize("NFC", name)
    stem = _ILLEGAL.sub("_", stem)
    stem = _WHITESPACE.sub("_", stem)
    stem = _UNDERSCORES.sub("_", stem)
    stem = stem.strip(" ._")
    stem = stem[:max_len].strip(" ._")
    if not stem:
        return fallback
    # Reserved-name check runs on the sanitized stem so it can't be bypassed by
    # illegal chars; the "_" prefix makes "CON" -> "_CON", "CON.mp4" -> "_CON.mp4".
    root = stem.split(".", 1)[0]
    if root.upper() in _RESERVED_NAMES:
        stem = "_" + stem
    return stem


def build_output_basename(
    stem: str,
    model: str,
    *,
    exists: Callable[[str], bool],
    job_id: str,
) -> str:
    """Return a collision-safe output basename ``<stem>__<model>``.

    ``exists(basename)`` reports whether an output with that basename already
    exists on disk. On collision (same source + same model run again), append a
    short job-id suffix so multiple transcriptions coexist without overwriting.
    """
    base = f"{stem}__{model}"
    if not exists(base):
        return base
    return f"{stem}__{model}__{job_id[:8]}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_naming.py -v`
Expected: PASS (12 tests).

- [ ] **Step 5: Commit**

```bash
git add src/utils/naming.py tests/test_naming.py
git commit -m "feat: stem sanitization and collision-safe output naming"
```

---

## Task 3: Progress callback + detected-language capture in transcription

**Files:**
- Modify: `src/transcription/asr_engine.py:31-113`
- Modify: `src/pipeline/steps_transcription.py:11-32`
- Test: `tests/test_asr_engine.py` (extend existing)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `transcribe_audio(model, audio_path, language, progress_update_interval_seconds, logger, *, progress_callback: Callable[[float], None] | None = None) -> TranscriptDocument` where `language: str | None` (falsy `language` — `None` **or** `""` — ⇒ auto-detect, i.e. no `language` kwarg reaches Whisper; the returned `document.language` is the Whisper-detected language).
  - `TranscriptionStep(whisper_model, progress_update_interval, *, progress_callback=None)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_asr_engine.py`:

```python
class TestProgressAndLanguage:
    def _model(self, result):
        from unittest.mock import MagicMock

        model = MagicMock()
        model.transcribe.return_value = result
        return model

    def test_progress_callback_called_with_completion(self):
        import logging
        from pathlib import Path

        from src.transcription.asr_engine import transcribe_audio

        calls = []
        model = self._model({"text": "hi", "segments": [], "language": "en"})
        transcribe_audio(
            model=model,
            audio_path=Path("/tmp/nope.mp3"),
            language="en",
            progress_update_interval_seconds=0.01,
            logger=logging.getLogger("t"),
            progress_callback=lambda f: calls.append(f),
        )
        assert calls and calls[-1] == 1.0  # always signals completion

    def test_detected_language_captured_when_auto(self):
        import logging
        from pathlib import Path

        from src.transcription.asr_engine import transcribe_audio

        model = self._model({"text": "hola", "segments": [], "language": "es"})
        doc = transcribe_audio(
            model=model,
            audio_path=Path("/tmp/nope.mp3"),
            language=None,
            progress_update_interval_seconds=0.01,
            logger=logging.getLogger("t"),
        )
        assert doc.language == "es"
        # language=None means we pass no language kwarg to Whisper (auto-detect)
        _, kwargs = model.transcribe.call_args
        assert "language" not in kwargs

    def test_empty_string_language_routes_to_auto(self):
        import logging
        from pathlib import Path

        from src.transcription.asr_engine import transcribe_audio

        model = self._model({"text": "hi", "segments": [], "language": "fr"})
        transcribe_audio(
            model=model,
            audio_path=Path("/tmp/nope.mp3"),
            language="",  # empty string must also auto-detect (regression guard)
            progress_update_interval_seconds=0.01,
            logger=logging.getLogger("t"),
        )
        _, kwargs = model.transcribe.call_args
        assert "language" not in kwargs  # no language kwarg passed at all
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_asr_engine.py::TestProgressAndLanguage -v`
Expected: FAIL — `transcribe_audio() got an unexpected keyword argument 'progress_callback'`.

- [ ] **Step 3: Implement**

`src/transcription/asr_engine.py` — replace the function body from the signature through the return. Key changes: new `progress_callback` kwarg, `language: str | None`, ticker invokes the callback, completion always signalled, detected language captured.

```python
def transcribe_audio(
    model,
    audio_path: Path,
    language: str | None,
    progress_update_interval_seconds: float,
    logger: logging.Logger,
    *,
    progress_callback: Callable[[float], None] | None = None,
) -> TranscriptDocument:
    """Run Whisper transcription and return a structured TranscriptDocument.

    Preserves segment-level data (text, timestamps) from Whisper output. When
    ``language`` is falsy (None or ""), Whisper auto-detects and the detected
    language is stored on the returned document. ``progress_callback`` (optional)
    receives a 0.0-1.0 fraction during the run and 1.0 on completion.
    """
    audio_str = str(audio_path)
    total_seconds = _get_audio_duration_seconds(audio_str)

    stop_event = threading.Event()
    pbar: tqdm | None = None

    def _run_progress_bar() -> None:
        nonlocal pbar
        if total_seconds is None or total_seconds <= 0:
            return
        pbar = tqdm(total=int(total_seconds), desc=f"Transcribing {audio_path.name}", unit="s")
        start_time = time.time()
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            current = int(min(elapsed, pbar.total))
            if current != pbar.n:
                pbar.n = current
                pbar.refresh()
            if progress_callback is not None:
                progress_callback(min(elapsed / total_seconds, 1.0))
            time.sleep(progress_update_interval_seconds)
        pbar.n = pbar.total
        pbar.refresh()
        pbar.close()

    thread: threading.Thread | None = None
    if total_seconds and total_seconds > 0:
        thread = threading.Thread(target=_run_progress_bar, daemon=True)
        thread.start()
    else:
        logger.debug("Could not estimate media duration for progress bar: %s", audio_path)

    transcribe_kwargs = {"fp16": False, "condition_on_previous_text": False}
    if language:  # None or "" => auto-detect (omit the language kwarg entirely)
        transcribe_kwargs["language"] = language

    try:
        # condition_on_previous_text=False stops Whisper's self-conditioning
        # decode loop from running away into repeated/garbled phantom segments
        # on low-confidence trailing audio (silence, outro, background noise).
        # With the default (True), large-v3 emitted ~18s of hallucinated
        # content past the true end of audio. See tests/test_asr_engine.py.
        result = model.transcribe(audio_str, **transcribe_kwargs)
    finally:
        stop_event.set()
        if thread:
            thread.join(timeout=0.5)
        if progress_callback is not None:
            progress_callback(1.0)

    segments = [
        TranscriptSegment(
            text=seg.get("text", "").strip(),
            start_time=seg.get("start"),
            end_time=seg.get("end"),
        )
        for seg in result.get("segments", [])
    ]

    if not segments and result.get("text"):
        segments = [TranscriptSegment(text=result["text"])]

    detected_language = result.get("language") or language or "unknown"

    doc = TranscriptDocument(
        source_file=str(audio_path),
        segments=segments,
        language=detected_language,
        pipeline_state=PipelineState.TRANSCRIBED,
    )

    logger.info(
        "Transcription complete: %d segments, %.1fs total",
        len(segments),
        doc.duration_seconds or 0.0,
    )
    return doc
```

Add the import at the top of the file (after the existing imports):

```python
from collections.abc import Callable
```

> **Note for the existing tests:** the prior test asserted `language` was passed positionally/keyword as `"en"`. With `language="en"` the new code still puts `language="en"` in `transcribe_kwargs`, so `kwargs["language"] == "en"` holds. `condition_on_previous_text is False` and `fp16 is False` are still asserted-true. If the existing test reads `model.transcribe.call_args` positionally, it remains valid because the first positional arg (`audio_str`) is unchanged.

`src/pipeline/steps_transcription.py` — thread the callback:

```python
class TranscriptionStep(PipelineStep):
    """Run Whisper ASR on the audio file."""

    def __init__(self, whisper_model, progress_update_interval: float, *, progress_callback=None) -> None:
        self._model = whisper_model
        self._interval = progress_update_interval
        self._progress_callback = progress_callback

    def execute(self, context: PipelineContext, logger: logging.Logger) -> PipelineContext:
        if context.audio_path is None:
            context.fail("No audio path set before transcription step")
            return context

        document = transcribe_audio(
            model=self._model,
            audio_path=context.audio_path,
            language=context.language,
            progress_update_interval_seconds=self._interval,
            logger=logger,
            progress_callback=self._progress_callback,
        )
        document.source_file = str(context.source_path)
        context.document = document
        return context
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_asr_engine.py -v`
Expected: PASS (existing 2 tests + new 2).

- [ ] **Step 5: Commit**

```bash
git add src/transcription/asr_engine.py src/pipeline/steps_transcription.py tests/test_asr_engine.py
git commit -m "feat: optional progress callback and auto-detect language in transcription"
```

---

## Task 4: Ingestion raises `MediaDecodeError` on undecodable media

**Files:**
- Modify: `src/pipeline/steps_ingestion.py`
- Test: `tests/test_media_decode_error.py` (extend)

**Interfaces:**
- Consumes: `MediaDecodeError` (Task 1), `_get_audio_duration_seconds` (existing private helper in `asr_engine`, already reused by `diarizer.py`).
- Produces: `VideoIngestionStep` / `AudioIngestionStep` raise `MediaDecodeError` (with a source-named message) instead of silently failing on undecodable input.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_media_decode_error.py`:

```python
import logging as _logging

import pytest

from src.models import PipelineContext as _Ctx
from src.pipeline import AudioIngestionStep, VideoIngestionStep
from src.utils import MediaDecodeError as _MDE


def test_audio_ingestion_raises_on_undecodable(monkeypatch, tmp_path):
    fake = tmp_path / "broken.mp3"
    fake.write_bytes(b"not really audio")
    # Force the duration probe to report "cannot decode".
    monkeypatch.setattr("src.pipeline.steps_ingestion._probe_duration", lambda p: None)
    step = AudioIngestionStep()
    ctx = _Ctx(source_path=fake, input_type="audio")
    with pytest.raises(_MDE):
        step.execute(ctx, _logging.getLogger("t"))


def test_video_ingestion_raises_on_no_audio_track(monkeypatch, tmp_path):
    fake = tmp_path / "silent.mp4"
    fake.write_bytes(b"x")
    monkeypatch.setattr(
        "src.pipeline.steps_ingestion.extract_audio_from_video",
        lambda src, out, logger: False,  # simulate no audio track
    )
    step = VideoIngestionStep(tmp_path, ".mp3")
    ctx = _Ctx(source_path=fake, input_type="video")
    with pytest.raises(_MDE):
        step.execute(ctx, _logging.getLogger("t"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_media_decode_error.py -k "undecodable or no_audio_track" -v`
Expected: FAIL — `AttributeError: ... has no attribute '_probe_duration'` / no exception raised.

- [ ] **Step 3: Implement**

`src/pipeline/steps_ingestion.py` — replace entire file:

```python
from __future__ import annotations

import logging
from pathlib import Path

from src.ingestion.video_extractor import extract_audio_from_video
from src.models import PipelineContext, PipelineState, TranscriptDocument
from src.transcription.asr_engine import _get_audio_duration_seconds
from src.utils.errors import MediaDecodeError

from .steps import PipelineStep


def _probe_duration(path: Path) -> float | None:
    """Return decodable duration in seconds, or None if the file can't be read."""
    return _get_audio_duration_seconds(str(path))


class VideoIngestionStep(PipelineStep):
    """Extract audio from a video file."""

    def __init__(self, audios_path: Path, extracted_audio_extension: str) -> None:
        self._audios_path = audios_path
        self._extension = extracted_audio_extension

    def execute(self, context: PipelineContext, logger: logging.Logger) -> PipelineContext:
        audio_output = self._audios_path / (context.source_path.stem + self._extension)
        try:
            success = extract_audio_from_video(context.source_path, audio_output, logger)
        except Exception as exc:  # moviepy/ffmpeg decode failure
            raise MediaDecodeError(
                f"Could not read/decode '{context.source_path.name}': it may be corrupt, "
                "truncated, or use an unsupported codec."
            ) from exc
        if not success:
            raise MediaDecodeError(
                f"Could not read/decode '{context.source_path.name}': it has no audio track."
            )

        context.audio_path = audio_output
        context.document = TranscriptDocument(
            source_file=str(context.source_path),
            language=context.language,
            pipeline_state=PipelineState.INGESTED,
        )
        return context


class AudioIngestionStep(PipelineStep):
    """Prepare an audio file for transcription (no extraction needed)."""

    def execute(self, context: PipelineContext, logger: logging.Logger) -> PipelineContext:
        # Probe decodability up front so corrupt/unsupported audio yields a clear
        # MediaDecodeError instead of an opaque ffmpeg failure mid-transcription.
        if _probe_duration(context.source_path) is None:
            raise MediaDecodeError(
                f"Could not read/decode '{context.source_path.name}': it may be corrupt, "
                "truncated, missing an audio track, or use an unsupported codec."
            )
        context.audio_path = context.source_path
        context.document = TranscriptDocument(
            source_file=str(context.source_path),
            language=context.language,
            pipeline_state=PipelineState.INGESTED,
        )
        logger.info("Audio file ready: %s", context.source_path.name)
        return context
```

> **CLI parity note:** previously a no-audio video logged a warning and the file was skipped via `context.fail`. Now it raises `MediaDecodeError`, which the orchestrator catches and records on `context.errors`/`context.exception`, then breaks — same net effect (file skipped, error logged, batch continues to the next file). Verify with the existing ingestion tests in Step 4.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_media_decode_error.py tests/test_ingestion.py -v`
Expected: PASS. If `tests/test_ingestion.py` asserted the old return-False/skip behavior of the *step* (not the `extract_audio_from_video` function), update that assertion to expect `MediaDecodeError`; the underlying `extract_audio_from_video` bool contract is unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/steps_ingestion.py tests/test_media_decode_error.py
git commit -m "feat: ingestion raises MediaDecodeError on undecodable or no-audio media"
```

---

## Task 5: `OutputStep` basename override + records written paths

**Files:**
- Modify: `src/pipeline/steps_output.py`
- Test: `tests/test_output.py` (extend)

**Interfaces:**
- Consumes: nothing new.
- Produces: `OutputStep(transcripts_folder, transcript_extension, cleaned_suffix, *, output_basename: str | None = None)`. After `execute`, `context.document.metadata["outputs"]` holds `{"txt": str, "clean": str | None}` with the exact paths written.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_output.py`:

```python
import logging

from src.models import PipelineContext, TranscriptDocument, TranscriptSegment
from src.pipeline import OutputStep


def test_output_step_uses_basename_override(tmp_path):
    doc = TranscriptDocument(segments=[TranscriptSegment(text="hello")])
    ctx = PipelineContext(source_path=tmp_path / "orig name.mp3", input_type="audio", document=doc)
    step = OutputStep(tmp_path, ".txt", "_clean", output_basename="orig_name__base")
    step.execute(ctx, logging.getLogger("t"))
    written = tmp_path / "orig_name__base.txt"
    assert written.exists()
    assert ctx.document.metadata["outputs"]["txt"] == str(written)
    assert ctx.document.metadata["outputs"]["clean"] is None


def test_output_step_defaults_to_source_stem(tmp_path):
    doc = TranscriptDocument(segments=[TranscriptSegment(text="hi")])
    ctx = PipelineContext(source_path=tmp_path / "clip.mp3", input_type="audio", document=doc)
    OutputStep(tmp_path, ".txt", "_clean").execute(ctx, logging.getLogger("t"))
    assert (tmp_path / "clip.txt").exists()


def test_output_step_writes_clean_when_present(tmp_path):
    doc = TranscriptDocument(segments=[TranscriptSegment(text="hi")])
    ctx = PipelineContext(source_path=tmp_path / "clip.mp3", input_type="audio", document=doc)
    ctx.cleaned_text = "cleaned"
    OutputStep(tmp_path, ".txt", "_clean", output_basename="clip__base").execute(ctx, logging.getLogger("t"))
    assert (tmp_path / "clip__base_clean.txt").read_text(encoding="utf-8") == "cleaned"
    assert ctx.document.metadata["outputs"]["clean"] == str(tmp_path / "clip__base_clean.txt")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_output.py -k "output_step" -v`
Expected: FAIL — `OutputStep.__init__() got an unexpected keyword argument 'output_basename'`.

- [ ] **Step 3: Implement**

`src/pipeline/steps_output.py` — replace entire file:

```python
from __future__ import annotations

import logging
from pathlib import Path

from src.models import PipelineContext, PipelineState
from src.output.formatter import write_text_file, write_transcript

from .steps import PipelineStep


class OutputStep(PipelineStep):
    """Write transcript and (optionally) cleaned text to disk."""

    def __init__(
        self,
        transcripts_folder: Path,
        transcript_extension: str,
        cleaned_suffix: str,
        *,
        output_basename: str | None = None,
    ) -> None:
        self._transcripts_folder = transcripts_folder
        self._extension = transcript_extension
        self._cleaned_suffix = cleaned_suffix
        self._output_basename = output_basename

    def execute(self, context: PipelineContext, logger: logging.Logger) -> PipelineContext:
        if context.document is None:
            context.fail("No document to write")
            return context

        stem = self._output_basename or context.source_path.stem
        transcript_path = self._transcripts_folder / (stem + self._extension)
        write_transcript(context.document, transcript_path)
        logger.info("Transcript saved: %s", transcript_path)

        clean_path: Path | None = None
        if context.cleaned_text:
            clean_path = self._transcripts_folder / (stem + self._cleaned_suffix + self._extension)
            write_text_file(clean_path, context.cleaned_text)
            logger.info("Cleaned transcript saved: %s", clean_path)

        outputs = context.document.metadata.setdefault("outputs", {})
        outputs["txt"] = str(transcript_path)
        outputs["clean"] = str(clean_path) if clean_path else None

        context.document.pipeline_state = PipelineState.EXPORTED
        return context
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_output.py -v`
Expected: PASS (existing + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/steps_output.py tests/test_output.py
git commit -m "feat: OutputStep basename override and recorded output paths"
```

---

## Task 6: Single-slot Whisper model cache (`get_whisper_model`)

**Files:**
- Create: `src/service.py` (this task adds only the cache + `reset_caches`)
- Test: `tests/test_service.py` (create)

**Interfaces:**
- Produces:
  - `get_whisper_model(name: str, device: str, logger=None) -> model` — holds at most one resident model; reloads only on a real name change; evicts (drop ref, `gc.collect`, `torch.cuda.empty_cache` if cuda) before loading a different model.
  - `reset_caches() -> None` — clears the resident model (and, later, the diarization backend); test/teardown helper.

- [ ] **Step 1: Write the failing test**

Create `tests/test_service.py`:

```python
import src.service as service


class TestModelCache:
    def setup_method(self):
        service.reset_caches()

    def teardown_method(self):
        service.reset_caches()

    def test_same_model_loaded_once(self, monkeypatch):
        loads = []

        def fake_load(name, device=None):
            loads.append(name)
            return f"model:{name}"

        monkeypatch.setattr(service.whisper, "load_model", fake_load)
        a = service.get_whisper_model("base", "cpu")
        b = service.get_whisper_model("base", "cpu")
        assert a is b
        assert loads == ["base"]  # loaded exactly once

    def test_model_change_evicts_and_reloads(self, monkeypatch):
        loads = []
        collected = []
        monkeypatch.setattr(service.whisper, "load_model", lambda name, device=None: (loads.append(name) or f"model:{name}"))
        monkeypatch.setattr(service.gc, "collect", lambda: collected.append(True))

        service.get_whisper_model("base", "cpu")
        service.get_whisper_model("large-v3", "cpu")
        assert loads == ["base", "large-v3"]
        assert collected  # eviction ran gc.collect at least once
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.service'`.

- [ ] **Step 3: Implement**

Create `src/service.py` (cache portion only for now):

```python
from __future__ import annotations

import gc
import logging
import threading

import whisper

_MODEL_LOCK = threading.Lock()
_resident_model: tuple[str, object] | None = None  # (name, model)
_diarization_backend: tuple[str, str, object] | None = None  # (backend, model_id, obj)


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def get_whisper_model(name: str, device: str, logger: logging.Logger | None = None):
    """Return a resident Whisper model, holding at most one in memory.

    Reuses the resident model when ``name`` matches; otherwise evicts the
    previous model (drop ref + gc + empty CUDA cache) before loading the new
    one. Jobs run one at a time, so only one model is ever needed at once.
    """
    global _resident_model
    log = logger or logging.getLogger("video_to_text")
    with _MODEL_LOCK:
        if _resident_model is not None and _resident_model[0] == name:
            return _resident_model[1]
        if _resident_model is not None:
            log.info("Evicting Whisper model '%s' to load '%s'", _resident_model[0], name)
            _resident_model = None
            gc.collect()
            _empty_cuda_cache()
        log.info("Loading Whisper model: %s", name)
        model = whisper.load_model(name, device=device)
        _resident_model = (name, model)
        log.info("Whisper model loaded: %s", name)
        return model


def reset_caches() -> None:
    """Drop the resident model and diarization backend (tests/teardown)."""
    global _resident_model, _diarization_backend
    with _MODEL_LOCK:
        _resident_model = None
        _diarization_backend = None
        gc.collect()
        _empty_cuda_cache()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_service.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/service.py tests/test_service.py
git commit -m "feat: single-slot Whisper model cache in service"
```

---

## Task 7: `transcribe_file` + `TranscribeResult` (the seam)

**Files:**
- Modify: `src/service.py`
- Test: `tests/test_service.py` (extend)

**Interfaces:**
- Consumes: `get_whisper_model` (Task 6); ingestion/transcription/cleanup/diarization/output steps; `sanitize_stem`/`build_output_basename` (Task 2); `MediaDecodeError` (Task 1); `load_yaml_file`, `select_device`.
- Produces:
  - `@dataclass TranscribeResult` with: `document`, `txt_path`, `clean_path`, `srt_path`, `vtt_path`, `model`, `language`, `options: dict`, `status: str`, `warnings: list[str]`, `message: str | None`.
  - `transcribe_file(source_path, *, model, language=None, cleanup=False, diarize=False, num_speakers=None, write_srt_vtt=False, sanitize_output=False, config_path="configurations/general_config.yaml", logger=None, progress_callback=None) -> TranscribeResult`.
  - `get_diarization_backend(config, device, logger)` — cached per (backend, model).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_service.py`:

```python
from pathlib import Path

import src.service as service
from src.models import PipelineState, TranscriptDocument, TranscriptSegment


def _stub_doc():
    return TranscriptDocument(
        segments=[
            TranscriptSegment(text="Hello world.", start_time=0.0, end_time=1.5),
            TranscriptSegment(text="Second line.", start_time=1.5, end_time=3.0),
        ],
        language="en",
        pipeline_state=PipelineState.TRANSCRIBED,
    )


class TestTranscribeFile:
    def setup_method(self):
        service.reset_caches()

    def teardown_method(self):
        service.reset_caches()

    def _patch(self, monkeypatch):
        # Model load is a no-op object; transcription returns a stub document.
        monkeypatch.setattr(service.whisper, "load_model", lambda name, device=None: object())
        monkeypatch.setattr(service, "_resolve_device", lambda logger: "cpu")

        def fake_transcribe(model, audio_path, language, progress_update_interval_seconds, logger, progress_callback=None):
            if progress_callback:
                progress_callback(1.0)
            doc = _stub_doc()
            doc.source_file = str(audio_path)
            return doc

        monkeypatch.setattr("src.pipeline.steps_transcription.transcribe_audio", fake_transcribe)

    def test_transcribes_audio_and_writes_txt(self, monkeypatch, tmp_path):
        self._patch(monkeypatch)
        # point outputs at tmp by overriding config
        cfg = _tmp_config(tmp_path)
        src_audio = tmp_path / "clip.mp3"
        src_audio.write_bytes(b"x")
        monkeypatch.setattr("src.pipeline.steps_ingestion._probe_duration", lambda p: 1.0)  # skip real decode probe

        result = service.transcribe_file(
            src_audio, model="base", language="en", config_path=cfg, sanitize_output=True
        )
        assert result.status == "success"
        assert Path(result.txt_path).exists()
        assert "Hello world." in Path(result.txt_path).read_text(encoding="utf-8")
        assert result.model == "base"

    def test_does_not_mutate_params_yaml(self, monkeypatch, tmp_path):
        self._patch(monkeypatch)
        cfg = _tmp_config(tmp_path)
        params_file = tmp_path / "params.yaml"
        before = params_file.read_text(encoding="utf-8")
        src_audio = tmp_path / "clip.mp3"
        src_audio.write_bytes(b"x")
        monkeypatch.setattr("src.pipeline.steps_ingestion._probe_duration", lambda p: 1.0)
        service.transcribe_file(src_audio, model="large-v3", language="en", config_path=cfg, sanitize_output=True)
        assert params_file.read_text(encoding="utf-8") == before

    def test_unsupported_extension_raises_media_decode_error(self, monkeypatch, tmp_path):
        self._patch(monkeypatch)
        cfg = _tmp_config(tmp_path)
        bad = tmp_path / "file.pdf"
        bad.write_bytes(b"x")
        import pytest

        from src.utils import MediaDecodeError

        with pytest.raises(MediaDecodeError):
            service.transcribe_file(bad, model="base", language="en", config_path=cfg)

    def test_auto_detect_passes_no_language_kwarg(self, monkeypatch, tmp_path):
        # Integration: real transcribe_audio + real pipeline; only the Whisper
        # model and the decode probe are mocked. Asserts language=None reaches
        # model.transcribe as "no language kwarg" and the detected language is
        # captured end-to-end. (Does NOT use self._patch — uses the real
        # transcription path so the Task 3 auto-detect fix is exercised here.)
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hola", "segments": [], "language": "es"}
        monkeypatch.setattr(service.whisper, "load_model", lambda name, device=None: mock_model)
        monkeypatch.setattr(service, "_resolve_device", lambda logger: "cpu")
        # AudioIngestionStep probes decodability (Task 4); force it to pass.
        monkeypatch.setattr("src.pipeline.steps_ingestion._probe_duration", lambda p: 1.0)

        cfg = _tmp_config(tmp_path)
        src_audio = tmp_path / "clip.mp3"
        src_audio.write_bytes(b"x")
        result = service.transcribe_file(
            src_audio, model="base", language=None, config_path=cfg, sanitize_output=True
        )
        _, kwargs = mock_model.transcribe.call_args
        assert "language" not in kwargs  # auto-detect: no language kwarg
        assert result.language == "es"  # detected language captured end-to-end


def _tmp_config(tmp_path) -> str:
    """Write a minimal general_config + params + prompts under tmp_path; return config path."""
    import textwrap

    (tmp_path / "params.yaml").write_text("transcription_model: base\ncleanup_model: x\n", encoding="utf-8")
    (tmp_path / "prompts.yaml").write_text("cleanup_prompt: clean this\n", encoding="utf-8")
    (tmp_path / "diarization.yaml").write_text("enabled: false\n", encoding="utf-8")
    cfg = tmp_path / "general_config.yaml"
    cfg.write_text(
        textwrap.dedent(
            f"""
            paths:
              videos: "{tmp_path}/videos"
              audios: "{tmp_path}/audios"
              transcripts: "{tmp_path}/transcripts"
              logs: "{tmp_path}/logs"
            files:
              params: "{tmp_path}/params.yaml"
              prompts: "{tmp_path}/prompts.yaml"
              diarization: "{tmp_path}/diarization.yaml"
            extensions:
              video: [".mp4", ".mov", ".avi", ".mkv", ".webm"]
              audio: [".mp3", ".m4a"]
            output:
              transcript_extension: ".txt"
              cleaned_suffix: "_clean"
              extracted_audio_extension: ".mp3"
            ollama:
              url: "http://localhost:11434/api/generate"
              timeout_seconds: 600
              request_content_type: "application/json"
            processing:
              progress_update_interval_seconds: 0.25
            dependencies:
              ffmpeg_executable: "ffmpeg"
            logging:
              level: "INFO"
              file_name: "transcriber.log"
              format: "%(message)s"
            """
        ),
        encoding="utf-8",
    )
    return str(cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_service.py::TestTranscribeFile -v`
Expected: FAIL — `AttributeError: module 'src.service' has no attribute 'transcribe_file'`.

- [ ] **Step 3: Implement**

Append to `src/service.py` (imports at top, code below the cache):

Add to the import block at the top of `src/service.py`:

```python
import functools
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src.models import PipelineContext, TranscriptDocument
from src.pipeline import (
    AudioIngestionStep,
    CleanupStep,
    DiarizationStep,
    OutputStep,
    PipelineOrchestrator,
    TranscriptionStep,
    VideoIngestionStep,
)
from src.processing import cleanup_with_ollama
from src.transcription.diarization_config import load_diarization_config
from src.transcription.diarizer import create_diarization_backend
from src.utils import MediaDecodeError, ProcessingError, load_yaml_file, select_device
from src.utils.naming import build_output_basename, sanitize_stem

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_AUDIO_EXTS = {".mp3", ".m4a"}
_device_cache: str | None = None
```

Add these functions:

```python
@dataclass
class TranscribeResult:
    document: TranscriptDocument | None
    txt_path: str | None
    clean_path: str | None
    srt_path: str | None
    vtt_path: str | None
    model: str
    language: str | None
    options: dict
    status: str  # "success" | "warning" | "failed"
    warnings: list[str] = field(default_factory=list)
    message: str | None = None


def _resolve_device(logger: logging.Logger) -> str:
    global _device_cache
    if _device_cache is None:
        _device_cache = select_device(logger)
    return _device_cache


def get_diarization_backend(config, device: str, logger: logging.Logger):
    """Return a cached diarization backend, keyed by (backend, model)."""
    global _diarization_backend
    with _MODEL_LOCK:
        key = (config.backend, config.model)
        if _diarization_backend is not None and _diarization_backend[:2] == key:
            return _diarization_backend[2]
        backend = create_diarization_backend(config, device, logger)
        _diarization_backend = (config.backend, config.model, backend)
        return backend


def transcribe_file(
    source_path,
    *,
    model: str,
    language: str | None = None,
    cleanup: bool = False,
    diarize: bool = False,
    num_speakers: int | None = None,
    write_srt_vtt: bool = False,
    sanitize_output: bool = False,
    config_path: str = "configurations/general_config.yaml",
    logger: logging.Logger | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> TranscribeResult:
    """Build and run the existing step-pipeline for ONE file with per-job params.

    `params.yaml` is never mutated: `model`/`language`/`cleanup`/`diarize` are
    arguments. The same function backs both the CLI and the GUI.
    """
    source_path = Path(source_path)
    logger = logger or logging.getLogger("video_to_text")
    cfg = load_yaml_file(config_path)
    paths = cfg["paths"]
    output = cfg["output"]
    ollama = cfg["ollama"]
    processing = cfg["processing"]
    dependencies = cfg["dependencies"]

    transcripts_dir = Path(paths["transcripts"])
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    audios_dir = Path(paths["audios"])
    audios_dir.mkdir(parents=True, exist_ok=True)
    ext = source_path.suffix.lower()
    transcript_extension = output["transcript_extension"]

    options = {"cleanup": cleanup, "diarize": diarize, "num_speakers": num_speakers}

    # --- classify input ---
    if ext in _VIDEO_EXTS:
        input_type = "video"
        ingestion_step = VideoIngestionStep(audios_dir, output["extracted_audio_extension"])
    elif ext in _AUDIO_EXTS:
        input_type = "audio"
        ingestion_step = AudioIngestionStep()
    else:
        raise MediaDecodeError(
            f"Unsupported file type '{ext or source_path.name}'. "
            f"Supported: {', '.join(sorted(_VIDEO_EXTS | _AUDIO_EXTS))}."
        )

    # --- output basename (GUI: sanitized + collision-safe; CLI: plain stem) ---
    job_id = uuid.uuid4().hex
    if sanitize_output:
        stem = sanitize_stem(source_path.stem, fallback=job_id[:8])
        basename = build_output_basename(
            stem,
            model,
            exists=lambda b: (transcripts_dir / (b + transcript_extension)).exists(),
            job_id=job_id,
        )
    else:
        basename = None

    device = _resolve_device(logger)
    whisper_model = get_whisper_model(model, device, logger)

    # --- assemble steps ---
    transcription_step = TranscriptionStep(
        whisper_model=whisper_model,
        progress_update_interval=processing["progress_update_interval_seconds"],
        progress_callback=progress_callback,
    )
    steps = [ingestion_step, transcription_step]

    if diarize:
        diar_cfg = load_diarization_config(cfg["files"]["diarization"])
        diar_cfg.enabled = True
        backend = get_diarization_backend(diar_cfg, device, logger)
        steps.append(
            DiarizationStep(
                backend=backend,
                config=diar_cfg,
                work_dir=audios_dir / ".diarization_cache",
                ffmpeg_executable=dependencies["ffmpeg_executable"],
                num_speakers_override=num_speakers,
            )
        )

    if cleanup:
        params = load_yaml_file(cfg["files"]["params"])  # read-only
        prompts = load_yaml_file(cfg["files"]["prompts"])
        cleanup_func = functools.partial(
            cleanup_with_ollama,
            cleanup_model_name=params["cleanup_model"],
            cleanup_prompt=prompts["cleanup_prompt"],
            device=device,
            ollama_url=ollama["url"],
            ollama_timeout_seconds=ollama["timeout_seconds"],
            ollama_request_content_type=ollama["request_content_type"],
            logger=logger,
        )
        steps.append(CleanupStep(cleanup_func))

    steps.append(
        OutputStep(
            transcripts_folder=transcripts_dir,
            transcript_extension=transcript_extension,
            cleaned_suffix=output["cleaned_suffix"],
            output_basename=basename,
        )
    )

    # --- run ---
    context = PipelineContext(source_path=source_path, input_type=input_type, language=language or "")
    pipeline = PipelineOrchestrator(steps=steps, logger=logger)
    context = pipeline.run(context)

    # --- classify outcome ---
    if context.exception is not None:
        if isinstance(context.exception, MediaDecodeError):
            raise context.exception
        return TranscribeResult(
            document=context.document, txt_path=None, clean_path=None, srt_path=None, vtt_path=None,
            model=model, language=language, options=options, status="failed",
            message=str(context.exception),
        )

    doc = context.document
    outputs = (doc.metadata.get("outputs") if doc else None) or {}
    warnings: list[str] = []
    status = "success"
    # Cleanup was requested but produced nothing (e.g. Ollama down) -> warning.
    if cleanup and not outputs.get("clean"):
        warnings.append("Cleanup unavailable; raw transcript saved.")
        status = "warning"

    return TranscribeResult(
        document=doc,
        txt_path=outputs.get("txt"),
        clean_path=outputs.get("clean"),
        srt_path=outputs.get("srt"),
        vtt_path=outputs.get("vtt"),
        model=model,
        language=(doc.language if doc else language),
        options=options,
        status=status,
        warnings=warnings,
        message="; ".join(warnings) or None,
    )
```

> **Cleanup graceful degradation:** `CleanupStep` calls `cleanup_with_ollama`, which raises `ProcessingError` when Ollama is unreachable. To keep the raw transcript, the cleanup must not abort the whole job. Implement this by wrapping the cleanup call so a failure is logged and skipped rather than raised — see Task 14 (Phase 2) where the GUI exposes cleanup. **For Phase 1, the GUI does not pass `cleanup=True`,** so no change to `CleanupStep` is needed yet; the `status == "warning"` branch above is the contract Task 14 fulfills.

> **Decode-probe seam (tests):** the authoritative decode check lives in the ingestion steps (Task 4, `steps_ingestion._probe_duration`). Service-level tests that use fake-bytes fixtures monkeypatch `src.pipeline.steps_ingestion._probe_duration` to bypass real decoding.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_service.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/service.py tests/test_service.py
git commit -m "feat: transcribe_file seam with per-job params and cached diarization backend"
```

---

## Task 8: Refactor `main.py` to call `transcribe_file`

**Files:**
- Modify: `main.py` (imports + `orchestrate()` body + new `_process_batch` helper)
- Test: `tests/test_main_batch.py` (create) + CLI smoke + full suite

**Interfaces:**
- Consumes: `transcribe_file` (Task 7), `MediaDecodeError` (Task 1).
- Produces: unchanged CLI contract; new module-level helper `_process_batch(media_files, *, model_name, language, cleanup, diarize, num_speakers, logger) -> None` that transcribes each file and **continues past any single-file failure** (a corrupt/no-audio file must not kill the batch — see Task 4's parity note).

> **Baseline first:** before changing anything, run `python -m pytest -q` and record the green count (e.g. "≈70 passed") so Step 4 can confirm no regressions.

- [ ] **Step 1: Write the failing test**

Create `tests/test_main_batch.py` — the batch must attempt every file even when one raises:

```python
import logging
from pathlib import Path

import main
from src.service import TranscribeResult
from src.utils import MediaDecodeError


def _ok():
    return TranscribeResult(
        document=None, txt_path=None, clean_path=None, srt_path=None, vtt_path=None,
        model="base", language="en", options={}, status="success",
    )


def test_batch_continues_after_one_file_raises(monkeypatch, tmp_path):
    attempted = []

    def fake_transcribe(file_path, **kwargs):
        attempted.append(Path(file_path).name)
        if len(attempted) == 2:  # second of three files blows up
            raise MediaDecodeError("corrupt")
        return _ok()

    monkeypatch.setattr(main, "transcribe_file", fake_transcribe)
    files = [
        ("a.mp3", tmp_path / "a.mp3"),
        ("b.mp3", tmp_path / "b.mp3"),
        ("c.mp3", tmp_path / "c.mp3"),
    ]
    main._process_batch(
        files, model_name="base", language="en", cleanup=False,
        diarize=False, num_speakers=None, logger=logging.getLogger("t"),
    )
    assert attempted == ["a.mp3", "b.mp3", "c.mp3"]  # all three attempted despite #2 raising


def test_batch_continues_after_unexpected_exception(monkeypatch, tmp_path):
    attempted = []

    def fake_transcribe(file_path, **kwargs):
        attempted.append(Path(file_path).name)
        if len(attempted) == 1:
            raise RuntimeError("boom")  # non-MediaDecodeError must also be caught
        return _ok()

    monkeypatch.setattr(main, "transcribe_file", fake_transcribe)
    files = [("a.mp3", tmp_path / "a.mp3"), ("b.mp3", tmp_path / "b.mp3")]
    main._process_batch(
        files, model_name="base", language="en", cleanup=False,
        diarize=False, num_speakers=None, logger=logging.getLogger("t"),
    )
    assert attempted == ["a.mp3", "b.mp3"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_main_batch.py -v`
Expected: FAIL — `AttributeError: module 'main' has no attribute '_process_batch'`.

- [ ] **Step 3: Implement**

`main.py` — replace the body of `orchestrate()` (lines 59-199) with a discover-then-delegate loop and add the module-level `_process_batch` helper. Keep the Windows `SetErrorMode` block, dotenv, and `if __name__` guard exactly as they are.

```python
def orchestrate() -> None:
    general_config = load_yaml_file(CONFIG_PATH)
    paths = general_config["paths"]
    files = general_config["files"]
    output = general_config["output"]
    processing_cfg = general_config["processing"]  # noqa: F841 (kept for parity/readability)
    dependencies = general_config["dependencies"]
    logging_config = general_config["logging"]

    logger = setup_logging(
        logs_dir=paths["logs"],
        level=logging_config["level"],
        file_name=logging_config["file_name"],
        log_format=logging_config["format"],
    )
    logger.info("Application started.")

    videos_path = paths["videos"]
    audios_path = paths["audios"]
    transcripts_folder = paths["transcripts"]
    cleaned_suffix = output["cleaned_suffix"]
    transcript_extension = output["transcript_extension"]

    ensure_directories([audios_path, transcripts_folder], logger)
    ensure_ffmpeg_available(dependencies["ffmpeg_executable"], logger)

    args = parse_cli_args(videos_path, audios_path, cleaned_suffix, transcript_extension)
    logger.info(
        "CLI arguments parsed: type=%s, language=%s, cleanup=%s, diarize=%s, num_speakers=%s",
        args.type, args.language, args.cleanup, args.diarize, args.num_speakers,
    )

    # Resolve diarization on/off from config + flags (CLI semantics preserved).
    diarization_config = load_diarization_config(files["diarization"])
    if args.no_diarize:
        diarize = False
    elif args.diarize:
        diarize = True
    else:
        diarize = diarization_config.enabled

    params = load_yaml_file(files["params"])
    model_name = params["transcription_model"]

    if args.type == "video":
        source_folder = videos_path
        extensions = tuple(general_config["extensions"]["video"])
    else:
        source_folder = audios_path
        extensions = tuple(general_config["extensions"]["audio"])

    media_files = discover_media_files(source_folder, extensions)
    logger.info("Found %d supported files in %s.", len(media_files), source_folder)

    _process_batch(
        media_files,
        model_name=model_name,
        language=args.language,
        cleanup=args.cleanup,
        diarize=diarize,
        num_speakers=args.num_speakers,
        logger=logger,
    )

    logger.info("All processing completed.")


def _process_batch(
    media_files,
    *,
    model_name: str,
    language: str,
    cleanup: bool,
    diarize: bool,
    num_speakers,
    logger,
) -> None:
    """Transcribe each discovered file, continuing past any single-file failure.

    A corrupt/no-audio file raises MediaDecodeError (re-raised by the service);
    defensively, any other exception is caught too. We log it and move on so one
    bad file never kills the batch (matches Task 4's CLI-parity note).
    """
    for _filename, file_path in media_files:
        try:
            result = transcribe_file(
                file_path,
                model=model_name,
                language=language,
                cleanup=cleanup,
                diarize=diarize,
                num_speakers=num_speakers,
                config_path=CONFIG_PATH,
                logger=logger,
            )
        except MediaDecodeError as exc:
            logger.error("Skipping %s: %s", file_path.name, exc)
            continue
        except Exception as exc:  # one bad file must not kill the batch
            logger.exception("Unexpected error on %s: %s", file_path.name, exc)
            continue
        if result.status == "failed":
            logger.error("Failed: %s (%s)", file_path.name, result.message)
```

Update the imports near the top of `main.py`: remove the now-unused step/pipeline/cleanup/diarizer/whisper imports that moved into the service, and add `transcribe_file`. The new import block (replacing lines 29-53) is:

```python
import whisper  # noqa: F401  (kept so model weights resolve identically; optional)

from src.ingestion import discover_media_files
from src.service import transcribe_file
from src.transcription.diarization_config import load_diarization_config
from src.utils import (
    MediaDecodeError,
    ProcessingError,
    ensure_directories,
    ensure_ffmpeg_available,
    load_yaml_file,
    parse_cli_args,
    setup_logging,
)
```

> Remove `import functools`, the `src.models.PipelineContext` import, the `src.pipeline` step imports, `create_diarization_backend`, `cleanup_with_ollama`, and `select_device` from `main.py` — they're now the service's responsibility. (`whisper` may be dropped entirely; it's harmless to keep.)

- [ ] **Step 4: Run the new test + full suite**

Run: `python -m pytest tests/test_main_batch.py -v && python -m pytest -q`
Expected: both batch tests PASS; full suite green (the recorded baseline count + 2 new batch tests).

- [ ] **Step 5: CLI smoke test (manual, real audio)**

```bash
source venv/bin/activate
export PATH="$(pwd)/venv/bin:/opt/homebrew/bin:$PATH"
# put a short real clip in audios/ (or reuse an existing one), then:
python main.py --type audio --language en
ls -la transcripts/
```
Expected: `transcripts/<stem>.txt` written exactly as before; logs show one model load reused across files.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main_batch.py
git commit -m "refactor: main.py delegates to transcribe_file; batch survives per-file failures"
```

---

## Task 9: History store (`src/history.py`)

**Files:**
- Create: `src/history.py`
- Test: `tests/test_history.py` (create)

**Interfaces:**
- Produces:
  - `@dataclass HistoryEntry` with fields: `id, source_filename, model, language, options, duration_seconds, word_count, created_at, status, outputs, message=None`.
  - `HistoryStore(path)` with `add(entry)`, `list() -> list[dict]`, `get(id) -> dict | None`, `delete(id) -> bool` (also unlinks output files), `search(query) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_history.py`:

```python
from src.history import HistoryEntry, HistoryStore


def _entry(eid, name, outputs=None):
    return HistoryEntry(
        id=eid, source_filename=name, model="base", language="en",
        options={"cleanup": False, "diarize": False, "num_speakers": None},
        duration_seconds=12.3, word_count=4, created_at="2026-06-19T10:00:00+00:00",
        status="success", outputs=outputs or {},
    )


def test_add_and_list_newest_first(tmp_path):
    store = HistoryStore(tmp_path / ".history.json")
    store.add(_entry("a", "first.mp3"))
    store.add(_entry("b", "second.mp3"))
    rows = store.list()
    assert [r["id"] for r in rows] == ["b", "a"]


def test_get_returns_record(tmp_path):
    store = HistoryStore(tmp_path / ".history.json")
    store.add(_entry("a", "x.mp3"))
    assert store.get("a")["source_filename"] == "x.mp3"
    assert store.get("missing") is None


def test_delete_removes_record_and_files(tmp_path):
    out = tmp_path / "x__base.txt"
    out.write_text("hi", encoding="utf-8")
    store = HistoryStore(tmp_path / ".history.json")
    store.add(_entry("a", "x.mp3", outputs={"txt": str(out)}))
    assert store.delete("a") is True
    assert store.get("a") is None
    assert not out.exists()
    assert store.delete("a") is False


def test_search_by_filename_and_date(tmp_path):
    store = HistoryStore(tmp_path / ".history.json")
    store.add(_entry("a", "lecture.mp3"))
    store.add(_entry("b", "interview.mp4"))
    assert [r["id"] for r in store.search("lecture")] == ["a"]
    assert {r["id"] for r in store.search("2026-06-19")} == {"a", "b"}
    assert len(store.search("")) == 2


def test_survives_corrupt_index(tmp_path):
    path = tmp_path / ".history.json"
    path.write_text("{not json", encoding="utf-8")
    store = HistoryStore(path)
    assert store.list() == []
    store.add(_entry("a", "x.mp3"))
    assert store.get("a") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_history.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.history'`.

- [ ] **Step 3: Implement**

Create `src/history.py`:

```python
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class HistoryEntry:
    id: str
    source_filename: str
    model: str
    language: str | None
    options: dict
    duration_seconds: float | None
    word_count: int
    created_at: str  # UTC ISO-8601
    status: str  # "success" | "failed" | "warning"
    outputs: dict  # {"txt":..., "clean":..., "srt":..., "vtt":...}
    message: str | None = None


class HistoryStore:
    """Atomic, lock-guarded JSON index of transcription jobs.

    Safe because jobs are serialized (queue concurrency 1). Newest first.
    """

    def __init__(self, path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def _read(self) -> list[dict]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _write(self, records: list[dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.parent / (self._path.name + ".tmp")
        tmp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    def add(self, entry: HistoryEntry) -> None:
        with self._lock:
            records = self._read()
            records.insert(0, asdict(entry))
            self._write(records)

    def list(self) -> list[dict]:
        with self._lock:
            return self._read()

    def get(self, entry_id: str) -> dict | None:
        return next((r for r in self.list() if r.get("id") == entry_id), None)

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            records = self._read()
            removed = [r for r in records if r.get("id") == entry_id]
            if not removed:
                return False
            for r in removed:
                for p in (r.get("outputs") or {}).values():
                    if p:
                        try:
                            Path(p).unlink(missing_ok=True)
                        except OSError:
                            pass
            self._write([r for r in records if r.get("id") != entry_id])
            return True

    def search(self, query: str) -> list[dict]:
        q = (query or "").strip().lower()
        rows = self.list()
        if not q:
            return rows
        return [
            r for r in rows
            if q in r.get("source_filename", "").lower() or q in r.get("created_at", "").lower()
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_history.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/history.py tests/test_history.py
git commit -m "feat: atomic JSON-index HistoryStore"
```

---

## Task 10: `requirements-gui.txt` + install-time gate

**Files:**
- Create: `requirements-gui.txt`

- [ ] **Step 1: Create the file**

`requirements-gui.txt`:

```
# GUI-only dependencies. Install AFTER requirements.txt, in the same venv.
# Verified: resolves to gradio 4.44.1 and leaves numpy 2.0.2 / torch /
# openai-whisper untouched (only pillow 11.3.0 -> 10.4.0, which is moviepy-only
# and within the repo's pillow<12.0 allowance). Do NOT let any gradio pin move
# numpy below 2 — switch gradio instead. Re-run the dry-run if you change this.
gradio>=4.44,<5
```

- [ ] **Step 2: Install-time gate (manual, run once)**

```bash
source venv/bin/activate
pip install --dry-run -r requirements-gui.txt 2>&1 | grep -Ei 'numpy|torch|openai-whisper|whisper|pillow' || true
# Expect: numpy/torch/whisper NOT listed as "Would install"; pillow downgrades only.
pip install -r requirements-gui.txt
python -c "import numpy, whisper, torch; print('numpy', numpy.__version__); print('ok')"
python -m pytest -q   # all existing tests still green after install
```
Expected: `numpy 2.0.2` unchanged; suite green. If numpy/torch/whisper would move, STOP and pick a different gradio 4.x (do not downgrade them).

- [ ] **Step 3: Commit**

```bash
git add requirements-gui.txt
git commit -m "build: add GUI requirements (gradio 4.x), numpy-safe"
```

---

## Task 11: Gradio GUI — Phase 1 (`app.py`)

**Files:**
- Create: `app.py`
- Test: `tests/test_app_smoke.py` (create) — import + build smoke (UI logic verified manually)

**Interfaces:**
- Consumes: `transcribe_file` (Task 7), `HistoryStore`/`HistoryEntry` (Task 9), `MediaDecodeError`/`ProcessingError`.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_app_smoke.py`:

```python
def test_app_builds_blocks():
    import app

    demo = app.build_ui()
    # gradio Blocks exposes .launch; assert we constructed something launchable
    assert hasattr(demo, "launch")


def test_history_row_mapping_handles_empty():
    import app

    assert app._history_rows([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_app_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app'` (or `gradio` import error before `requirements-gui.txt` is installed — install it via Task 10 first).

- [ ] **Step 3: Implement**

Create `app.py`:

```python
from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from src.history import HistoryEntry, HistoryStore
from src.service import transcribe_file
from src.utils import MediaDecodeError, ProcessingError, load_yaml_file, setup_logging

CONFIG_PATH = "configurations/general_config.yaml"
MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v3"]
LANGUAGE_CHOICES = ["Auto-detect", "en", "ru", "es", "fr", "de", "ar", "it", "pt", "zh", "ja"]

_cfg = load_yaml_file(CONFIG_PATH)
_transcripts_dir = Path(_cfg["paths"]["transcripts"])
_default_model = load_yaml_file(_cfg["files"]["params"])["transcription_model"]
_history = HistoryStore(_transcripts_dir / ".history.json")

logger = setup_logging(
    logs_dir=_cfg["paths"]["logs"],
    level=_cfg["logging"]["level"],
    file_name=_cfg["logging"]["file_name"],
    log_format=_cfg["logging"]["format"],
)

_HISTORY_COLUMNS = ["id", "file", "model", "language", "duration (s)", "words", "date", "status"]


def _history_rows(records=None) -> list[list]:
    records = _history.list() if records is None else records
    rows = []
    for r in records:
        rows.append([
            r.get("id", ""),
            r.get("source_filename", ""),
            r.get("model", ""),
            r.get("language", ""),
            round(r.get("duration_seconds") or 0.0, 1),
            r.get("word_count", 0),
            (r.get("created_at", "") or "")[:19].replace("T", " "),
            r.get("status", ""),
        ])
    return rows


def _run_transcription(file_path, model, language, progress=gr.Progress()):
    if not file_path:
        raise gr.Error("Please upload a media file first.")
    lang = None if language == "Auto-detect" else language
    progress(0.0, desc="Starting…")
    try:
        result = transcribe_file(
            file_path,
            model=model,
            language=lang,
            sanitize_output=True,
            config_path=CONFIG_PATH,
            logger=logger,
            progress_callback=lambda f: progress(f, desc="Transcribing…"),
        )
    except MediaDecodeError as exc:
        raise gr.Error(str(exc))
    except ProcessingError as exc:
        raise gr.Error(f"Processing failed: {exc}")

    if result.status == "failed":
        raise gr.Error(result.message or "Transcription failed.")

    doc = result.document
    _history.add(
        HistoryEntry(
            id=doc.id,
            source_filename=Path(file_path).name,
            model=result.model,
            language=result.language,
            options=result.options,
            duration_seconds=doc.duration_seconds,
            word_count=len((doc.full_text or "").split()),
            created_at=doc.created_at.isoformat(),
            status=result.status,
            outputs={"txt": result.txt_path, "clean": result.clean_path,
                     "srt": result.srt_path, "vtt": result.vtt_path},
            message=result.message,
        )
    )
    text = doc.full_text if doc else ""
    return text, result.txt_path, gr.update(value=_history_rows())


def _load_selected(history_table, evt: gr.SelectData):
    # evt.index = [row, col]; first column is the id
    if evt is None or not history_table:
        return "", None
    row = evt.index[0]
    entry_id = history_table[row][0]
    rec = _history.get(entry_id)
    if not rec:
        return "", None
    txt = (rec.get("outputs") or {}).get("txt")
    content = Path(txt).read_text(encoding="utf-8") if txt and Path(txt).exists() else "(file missing)"
    return content, txt


def _delete_selected(entry_id):
    if not entry_id:
        raise gr.Error("Select a row to delete (its id appears in the box).")
    _history.delete(entry_id)
    return gr.update(value=_history_rows())


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Local Whisper Transcription") as demo:
        gr.Markdown("# 🎙️ Local Whisper Transcription\nLocal, private, zero-cost.")

        with gr.Tab("Transcribe"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_in = gr.File(
                        label="Upload audio/video",
                        file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm", ".mp3", ".m4a"],
                        type="filepath",
                    )
                    model_in = gr.Dropdown(
                        MODEL_CHOICES, value=_default_model, label="Model",
                        info="Bigger = more accurate but slower on CPU. large-v3 downloads ~3 GB on first use.",
                    )
                    lang_in = gr.Dropdown(LANGUAGE_CHOICES, value="Auto-detect", label="Language")
                    run_btn = gr.Button("Transcribe", variant="primary")
                with gr.Column(scale=2):
                    out_text = gr.Textbox(label="Transcript", lines=18, show_copy_button=True)
                    out_file = gr.File(label="Download .txt")

        with gr.Tab("History"):
            with gr.Row():
                refresh_btn = gr.Button("Refresh")
                del_id = gr.Textbox(label="Delete by id", scale=2)
                del_btn = gr.Button("Delete", variant="stop")
            history_table = gr.Dataframe(
                headers=_HISTORY_COLUMNS, value=_history_rows(), interactive=False, wrap=True,
            )
            hist_text = gr.Textbox(label="Selected transcript", lines=14, show_copy_button=True)
            hist_file = gr.File(label="Download selected .txt")

        run_btn.click(
            _run_transcription,
            inputs=[file_in, model_in, lang_in],
            outputs=[out_text, out_file, history_table],
        )
        refresh_btn.click(lambda: gr.update(value=_history_rows()), outputs=history_table)
        history_table.select(_load_selected, inputs=[history_table], outputs=[hist_text, hist_file])
        del_btn.click(_delete_selected, inputs=[del_id], outputs=[history_table])

    return demo


def main() -> None:
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", inbrowser=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke test + manual launch**

```bash
python -m pytest tests/test_app_smoke.py -v        # PASS
python app.py                                       # opens http://127.0.0.1:7860
```
Manual checks: upload a short clip → Transcribe → progress advances → transcript shows → `.txt` downloads → History tab lists the row → select row loads transcript → delete removes row and file. Confirm an unsupported file (e.g. `.pdf`) shows the specific MediaDecodeError message, not a generic crash.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app_smoke.py
git commit -m "feat: Gradio web GUI (Phase 1) — upload, transcribe, history"
```

---

## Task 12: Phase 1 docs (README EN/RU + CLAUDE.md + QUICKSTART)

**Files:**
- Modify: `README.md`, `README.ru.md`, `CLAUDE.md`, `QUICKSTART.md`

- [ ] **Step 1: Add a "Web GUI" section to both READMEs**

In `README.md`, after the "## Usage" section, add (and mirror in `README.ru.md` in Russian):

````markdown
## Web GUI (local)

A local, single-user web interface (Gradio) for uploading a file, picking model
+ language, transcribing with live progress, and browsing history.

```bash
pip install -r requirements-gui.txt   # one-time, after requirements.txt
python app.py                          # opens http://127.0.0.1:7860
```

The GUI reuses the same engine as the CLI; transcripts and the history index
live in `transcripts/`. It binds to `127.0.0.1` only (no network exposure).
````

- [ ] **Step 2: Note the GUI in QUICKSTART.md**

Add under "## Where things are" or a new "## Web GUI" heading:

```markdown
## Web GUI (optional)

Prefer a browser? `pip install -r requirements-gui.txt` then `python app.py`
and open http://127.0.0.1:7860 — upload, pick a model, transcribe, download.
```

- [ ] **Step 3: Note the GUI seam in CLAUDE.md**

Add a short architecture note so future work uses the seam:

```markdown
## Architecture note (GUI + CLI)

Both the CLI (`main.py`) and the web GUI (`app.py`) call the single in-process
seam `src/service.py::transcribe_file(...)`, which builds and runs the existing
step-pipeline for one file with per-job parameters. `configurations/params.yaml`
is never mutated at runtime. Whisper models and the diarization backend are
cached (single resident model) in `src/service.py`.
```

- [ ] **Step 4: Commit**

```bash
git add README.md README.ru.md CLAUDE.md QUICKSTART.md
git commit -m "docs: document the local web GUI (EN/RU + QUICKSTART + CLAUDE)"
```

**Phase 1 is shippable here:** upload → model/language(+auto-detect) → transcribe with live progress → raw transcript + txt download → searchable-by-refresh history (list/open/download/delete), one-at-a-time queue, MediaDecodeError surfaced.

---

## ✅ Phase 1 Exit Gate (BLOCKING — do not start any Phase 2 task until all pass)

Phase 1 must be genuinely shippable on its own before Task 13 begins. Verify all
of the following and report the results to the user; if any fails, fix it within
Phase 1 (do not proceed to Phase 2):

- [ ] **Full test suite green:** `python -m pytest -q` — all tests pass (original 41 + every Phase 1 test added in Tasks 1–11). Zero failures, zero errors.
- [ ] **GUI end-to-end smoke** (`python app.py`, manual): upload a short real clip → pick model + language **and** confirm **Auto-detect** works (no language forced) → progress bar advances → transcript renders → `.txt` downloads → the job appears in the **History** tab → select the row to open it → re-download it → delete it (row disappears and the file is removed from `transcripts/`).
- [ ] **Graceful degradation visible:** an unsupported file (e.g. `.pdf`) shows the specific `MediaDecodeError` message (not a generic crash); with Ollama not running, Phase 1 (which does not request cleanup) is unaffected.
- [ ] **One-at-a-time queue:** starting a second transcription while one runs queues it (UI never freezes).
- [ ] **CLI parity confirmed:** `python main.py --type audio --language en` still writes `transcripts/<stem>.txt` exactly as before; a batch with one corrupt file still processes the rest.
- [ ] **Dependency safety:** `python -c "import numpy; print(numpy.__version__)"` reports `2.0.2` (numpy/torch/whisper not downgraded by the GUI install).

Only when every box above is checked may the team proceed to Phase 2.

---

# PHASE 2 — added features (top-down priority)

## Task 13: SRT/VTT formatting (`formatter.py`)

**Files:**
- Modify: `src/output/formatter.py`
- Test: `tests/test_formatter_srt_vtt.py` (create)

**Interfaces:**
- Produces: `format_srt(document) -> str`, `format_vtt(document) -> str`, `write_srt(document, path)`, `write_vtt(document, path)`. Segments without timestamps are skipped; speaker labels are prefixed when present.

- [ ] **Step 1: Write the failing test**

Create `tests/test_formatter_srt_vtt.py`:

```python
from src.models import Speaker, TranscriptDocument, TranscriptSegment
from src.output.formatter import format_srt, format_vtt, write_srt, write_vtt


def _doc():
    return TranscriptDocument(segments=[
        TranscriptSegment(text="Hello.", start_time=0.0, end_time=1.5),
        TranscriptSegment(text="World.", start_time=1.5, end_time=3.25),
    ])


def test_srt_structure():
    srt = format_srt(_doc())
    assert "1\n00:00:00,000 --> 00:00:01,500\nHello." in srt
    assert "2\n00:00:01,500 --> 00:00:03,250\nWorld." in srt


def test_vtt_header_and_timestamps():
    vtt = format_vtt(_doc())
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in vtt


def test_speaker_label_prefixed():
    doc = TranscriptDocument(segments=[
        TranscriptSegment(text="Hi.", start_time=0.0, end_time=1.0, speaker=Speaker(id="SPEAKER_00", label="Alice")),
    ])
    assert "Alice: Hi." in format_srt(doc)


def test_segments_without_timestamps_skipped():
    doc = TranscriptDocument(segments=[TranscriptSegment(text="no times")])
    assert format_srt(doc).strip() == ""


def test_write_helpers(tmp_path):
    write_srt(_doc(), tmp_path / "a.srt")
    write_vtt(_doc(), tmp_path / "a.vtt")
    assert (tmp_path / "a.srt").read_text(encoding="utf-8").startswith("1")
    assert (tmp_path / "a.vtt").read_text(encoding="utf-8").startswith("WEBVTT")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_formatter_srt_vtt.py -v`
Expected: FAIL — `ImportError: cannot import name 'format_srt'`.

- [ ] **Step 3: Implement**

Append to `src/output/formatter.py`:

```python
def _format_timestamp(seconds: float, *, sep: str) -> str:
    """Format seconds as HH:MM:SS<sep>mmm (sep ',' for SRT, '.' for VTT)."""
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def _timed_segments(document: TranscriptDocument):
    for seg in document.segments:
        if seg.start_time is None or seg.end_time is None:
            continue
        text = seg.text.strip()
        if not text:
            continue
        if seg.speaker is not None:
            label = seg.speaker.label or seg.speaker.id
            text = f"{label}: {text}"
        yield seg.start_time, seg.end_time, text


def format_srt(document: TranscriptDocument) -> str:
    blocks = []
    for index, (start, end, text) in enumerate(_timed_segments(document), start=1):
        blocks.append(
            f"{index}\n"
            f"{_format_timestamp(start, sep=',')} --> {_format_timestamp(end, sep=',')}\n"
            f"{text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def format_vtt(document: TranscriptDocument) -> str:
    lines = ["WEBVTT", ""]
    for start, end, text in _timed_segments(document):
        lines.append(f"{_format_timestamp(start, sep='.')} --> {_format_timestamp(end, sep='.')}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def write_srt(document: TranscriptDocument, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_srt(document), encoding="utf-8")


def write_vtt(document: TranscriptDocument, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_vtt(document), encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_formatter_srt_vtt.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/output/formatter.py tests/test_formatter_srt_vtt.py
git commit -m "feat: SRT/VTT formatting from segment timestamps"
```

---

## Task 14: Wire srt/vtt + graceful cleanup into `OutputStep`, `CleanupStep`, service

**Files:**
- Modify: `src/pipeline/steps_output.py`
- Modify: `src/pipeline/steps_cleanup.py`
- Modify: `src/service.py`
- Test: `tests/test_output.py`, `tests/test_service.py` (extend)

**Interfaces:**
- Consumes: `write_srt/write_vtt` (Task 13).
- Produces:
  - `OutputStep(..., *, output_basename=None, write_srt_vtt=False)` — when `write_srt_vtt` and the document has timestamped segments, also writes `<basename>.srt`/`.vtt` and records them in `metadata["outputs"]`.
  - `CleanupStep(cleanup_func)` degrades gracefully: a `ProcessingError` from the cleanup is logged and recorded as a warning; the raw transcript is preserved (no pipeline abort).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_output.py`:

```python
def test_output_step_writes_srt_vtt_when_enabled(tmp_path):
    import logging
    from src.models import PipelineContext, TranscriptDocument, TranscriptSegment
    from src.pipeline import OutputStep

    doc = TranscriptDocument(segments=[TranscriptSegment(text="hi", start_time=0.0, end_time=1.0)])
    ctx = PipelineContext(source_path=tmp_path / "c.mp3", input_type="audio", document=doc)
    OutputStep(tmp_path, ".txt", "_clean", output_basename="c__base", write_srt_vtt=True).execute(
        ctx, logging.getLogger("t")
    )
    assert (tmp_path / "c__base.srt").exists()
    assert (tmp_path / "c__base.vtt").exists()
    assert ctx.document.metadata["outputs"]["srt"] == str(tmp_path / "c__base.srt")
```

Append to `tests/test_service.py` (inside `TestTranscribeFile`):

```python
    def test_cleanup_failure_keeps_raw_and_warns(self, monkeypatch, tmp_path):
        self._patch(monkeypatch)
        cfg = _tmp_config(tmp_path)
        src_audio = tmp_path / "clip.mp3"
        src_audio.write_bytes(b"x")
        monkeypatch.setattr("src.pipeline.steps_ingestion._probe_duration", lambda p: 1.0)

        from src.utils import ProcessingError

        def boom(*a, **k):
            raise ProcessingError("Ollama down")

        monkeypatch.setattr("src.pipeline.steps_cleanup.format_document_with_speakers", lambda d: "text")
        monkeypatch.setattr(service, "cleanup_with_ollama", boom)
        result = service.transcribe_file(
            src_audio, model="base", language="en", cleanup=True, config_path=cfg, sanitize_output=True
        )
        assert result.status == "warning"
        assert Path(result.txt_path).exists()        # raw transcript saved
        assert result.clean_path is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_output.py -k srt_vtt tests/test_service.py -k cleanup_failure -v`
Expected: FAIL — `OutputStep.__init__() got an unexpected keyword argument 'write_srt_vtt'` / cleanup raises and aborts.

- [ ] **Step 3: Implement**

`src/pipeline/steps_output.py` — add the `write_srt_vtt` param and srt/vtt writing. Update the import and `__init__`/`execute`:

```python
from src.output.formatter import write_srt, write_text_file, write_transcript, write_vtt
```

```python
    def __init__(
        self,
        transcripts_folder: Path,
        transcript_extension: str,
        cleaned_suffix: str,
        *,
        output_basename: str | None = None,
        write_srt_vtt: bool = False,
    ) -> None:
        self._transcripts_folder = transcripts_folder
        self._extension = transcript_extension
        self._cleaned_suffix = cleaned_suffix
        self._output_basename = output_basename
        self._write_srt_vtt = write_srt_vtt
```

In `execute`, after writing txt/clean and before setting `pipeline_state`, add:

```python
        srt_path: Path | None = None
        vtt_path: Path | None = None
        if self._write_srt_vtt and any(
            s.start_time is not None and s.end_time is not None for s in context.document.segments
        ):
            srt_path = self._transcripts_folder / (stem + ".srt")
            vtt_path = self._transcripts_folder / (stem + ".vtt")
            write_srt(context.document, srt_path)
            write_vtt(context.document, vtt_path)
            logger.info("Subtitles saved: %s, %s", srt_path, vtt_path)

        outputs = context.document.metadata.setdefault("outputs", {})
        outputs["txt"] = str(transcript_path)
        outputs["clean"] = str(clean_path) if clean_path else None
        outputs["srt"] = str(srt_path) if srt_path else None
        outputs["vtt"] = str(vtt_path) if vtt_path else None
```

(Remove the older two-line `outputs[...]` assignment from Task 5 so it isn't duplicated.)

`src/pipeline/steps_cleanup.py` — make cleanup non-fatal. Replace the `execute` tail:

```python
        logger.info("Running cleanup for %s", context.source_path.name)
        try:
            context.cleaned_text = self._cleanup_func(full_text)
        except ProcessingError as exc:
            logger.warning("Cleanup unavailable for %s: %s", context.source_path.name, exc)
            context.document.metadata.setdefault("warnings", []).append(f"cleanup unavailable: {exc}")
        return context
```

Add the import at the top of `steps_cleanup.py`:

```python
from src.utils.errors import ProcessingError
```

`src/service.py` — pass `write_srt_vtt` into `OutputStep`:

```python
    steps.append(
        OutputStep(
            transcripts_folder=transcripts_dir,
            transcript_extension=transcript_extension,
            cleaned_suffix=output["cleaned_suffix"],
            output_basename=basename,
            write_srt_vtt=write_srt_vtt,
        )
    )
```

> The service's existing `status == "warning"` branch (Task 7) already covers "cleanup requested but no clean output." With `CleanupStep` now swallowing `ProcessingError`, the job completes, the raw txt is written, and `outputs["clean"]` is absent → warning. Test passes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_output.py tests/test_service.py tests/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/steps_output.py src/pipeline/steps_cleanup.py src/service.py tests/test_output.py tests/test_service.py
git commit -m "feat: srt/vtt outputs and graceful Ollama-cleanup degradation"
```

---

## Task 15: GUI — cleanup + diarize toggles, srt/vtt downloads, graceful gating

**Files:**
- Modify: `app.py`

**Interfaces:**
- Consumes: `transcribe_file(..., cleanup, diarize, num_speakers, write_srt_vtt)`; `DiarizationConfig.resolve_hf_token` (to detect token presence).

- [ ] **Step 1: Implement** (UI-driven; verified by smoke test + manual)

In `app.py`, detect HF token availability and Ollama presence to gate toggles:

```python
import os

from src.transcription.diarization_config import load_diarization_config

_diar_cfg = load_diarization_config(_cfg["files"]["diarization"])
_HF_TOKEN_PRESENT = bool(os.environ.get(_diar_cfg.hf_token_env, "").strip())
```

Add controls in the Transcribe tab (under `lang_in`):

```python
                    cleanup_in = gr.Checkbox(label="Clean up with Ollama", value=False,
                                             info="Requires a local Ollama server. Raw transcript is kept if unavailable.")
                    diarize_in = gr.Checkbox(
                        label="Speaker diarization", value=False,
                        interactive=_HF_TOKEN_PRESENT,
                        info=("Identifies speakers (needs HF_TOKEN)."
                              if _HF_TOKEN_PRESENT else
                              "Disabled: set HF_TOKEN in .env to enable (see README)."),
                    )
                    speakers_in = gr.Number(label="Exact #speakers (optional)", value=None, precision=0)
```

Add subtitle + clean outputs in the right column:

```python
                    out_clean = gr.Textbox(label="Cleaned transcript", lines=10, show_copy_button=True, visible=False)
                    out_srt = gr.File(label="Download .srt")
                    out_vtt = gr.File(label="Download .vtt")
```

Update `_run_transcription` to pass the toggles, write srt/vtt, and return the extra outputs:

```python
def _run_transcription(file_path, model, language, do_cleanup, do_diarize, num_speakers, progress=gr.Progress()):
    if not file_path:
        raise gr.Error("Please upload a media file first.")
    lang = None if language == "Auto-detect" else language
    n_speakers = int(num_speakers) if num_speakers else None
    progress(0.0, desc="Starting…")
    try:
        result = transcribe_file(
            file_path, model=model, language=lang,
            cleanup=bool(do_cleanup), diarize=bool(do_diarize), num_speakers=n_speakers,
            write_srt_vtt=True, sanitize_output=True, config_path=CONFIG_PATH, logger=logger,
            progress_callback=lambda f: progress(f, desc="Transcribing…"),
        )
    except MediaDecodeError as exc:
        raise gr.Error(str(exc))
    except ProcessingError as exc:
        raise gr.Error(f"Processing failed: {exc}")
    if result.status == "failed":
        raise gr.Error(result.message or "Transcription failed.")

    doc = result.document
    _history.add(HistoryEntry(
        id=doc.id, source_filename=Path(file_path).name, model=result.model, language=result.language,
        options=result.options, duration_seconds=doc.duration_seconds,
        word_count=len((doc.full_text or "").split()), created_at=doc.created_at.isoformat(),
        status=result.status,
        outputs={"txt": result.txt_path, "clean": result.clean_path, "srt": result.srt_path, "vtt": result.vtt_path},
        message=result.message,
    ))
    clean_text = ""
    if result.clean_path and Path(result.clean_path).exists():
        clean_text = Path(result.clean_path).read_text(encoding="utf-8")
    warn = gr.Warning(result.message) if result.status == "warning" and result.message else None  # noqa: F841
    return (
        doc.full_text if doc else "",
        result.txt_path, result.srt_path, result.vtt_path,
        gr.update(value=clean_text, visible=bool(clean_text)),
        gr.update(value=_history_rows()),
    )
```

Update the `run_btn.click` wiring:

```python
        run_btn.click(
            _run_transcription,
            inputs=[file_in, model_in, lang_in, cleanup_in, diarize_in, speakers_in],
            outputs=[out_text, out_file, out_srt, out_vtt, out_clean, history_table],
        )
```

- [ ] **Step 2: Smoke test + manual**

```bash
python -m pytest tests/test_app_smoke.py -v
python app.py
```
Manual: with no HF_TOKEN, the diarize checkbox is disabled with the explanatory note. With Ollama off and cleanup checked, the job still succeeds, shows a warning toast, and the raw transcript + txt are produced; `.srt`/`.vtt` download.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: GUI cleanup/diarize toggles, srt/vtt downloads, graceful gating"
```

---

## Task 16: GUI — batch upload (queue multiple)

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Implement**

Switch `file_in` to accept multiple files and iterate, queuing each (concurrency 1 serializes them):

```python
                    file_in = gr.File(
                        label="Upload audio/video (one or many)",
                        file_count="multiple",
                        file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm", ".mp3", ".m4a"],
                        type="filepath",
                    )
```

Add a batch handler that loops; show the last transcript and refresh history:

```python
def _run_batch(file_paths, model, language, do_cleanup, do_diarize, num_speakers, progress=gr.Progress()):
    if not file_paths:
        raise gr.Error("Please upload at least one media file.")
    paths = file_paths if isinstance(file_paths, list) else [file_paths]
    last = ("", None, None, None, gr.update())
    total = len(paths)
    for i, p in enumerate(paths):
        progress(i / total, desc=f"File {i + 1}/{total}: {Path(p).name}")
        last = _run_transcription(p, model, language, do_cleanup, do_diarize, num_speakers, progress=progress)
    progress(1.0, desc="Done")
    return last
```

Point `run_btn.click` at `_run_batch` instead of `_run_transcription` (same inputs/outputs).

- [ ] **Step 2: Smoke + manual**

Run `python app.py`, upload 2 short clips, confirm both transcribe sequentially and both appear in History.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: GUI batch upload (serialized via queue)"
```

---

## Task 17: GUI — history search box

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Implement**

Add a search box to the History tab and wire it to `HistoryStore.search`:

```python
            search_in = gr.Textbox(label="Search history (filename or date)", placeholder="e.g. lecture or 2026-06-19")
```

```python
        search_in.change(lambda q: gr.update(value=_history_rows(_history.search(q))), inputs=search_in, outputs=history_table)
```

- [ ] **Step 2: Smoke + manual**

Confirm typing filters the table live; clearing it restores all rows.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: GUI history search by filename/date"
```

---

## Task 18: GUI — media playback + error/log-tail surfacing

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Implement media playback for the current upload**

Add a player that mirrors the current upload (inputs aren't persisted, so this applies to the active upload only):

```python
                    media_preview = gr.Audio(label="Preview (current upload)", visible=False)
```

```python
        def _show_media(paths):
            p = paths[0] if isinstance(paths, list) and paths else paths
            if p and Path(p).suffix.lower() in {".mp3", ".m4a"}:
                return gr.update(value=p, visible=True)
            return gr.update(visible=False)

        file_in.change(_show_media, inputs=file_in, outputs=media_preview)
```

- [ ] **Step 2: Implement log-tail on unexpected failure**

Add a helper and surface the tail of `logs/transcriber.log` when a job fails for a non-classified reason:

```python
def _log_tail(n: int = 25) -> str:
    log_path = Path(_cfg["paths"]["logs"]) / _cfg["logging"]["file_name"]
    if not log_path.exists():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])
```

In `_run_transcription`, change the generic failure branch to include the tail:

```python
    if result.status == "failed":
        raise gr.Error((result.message or "Transcription failed.") + "\n\n--- recent log ---\n" + _log_tail())
```

- [ ] **Step 3: Smoke + manual**

Confirm an mp3 upload shows a playable preview; force a failure (e.g. a deliberately corrupt file that passes extension check) and confirm the error includes a log tail.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: GUI media preview and log-tail on failure"
```

---

## Task 19: Phase 2 docs refresh

**Files:**
- Modify: `README.md`, `README.ru.md`

- [ ] **Step 1: Extend the Web GUI section** (EN + RU) to mention: srt/vtt downloads, optional cleanup/diarize toggles (and that diarize needs HF_TOKEN, cleanup needs Ollama, both degrade gracefully), batch upload, and history search.

- [ ] **Step 2: Commit**

```bash
git add README.md README.ru.md
git commit -m "docs: document Phase 2 GUI features (srt/vtt, toggles, batch, search)"
```

---

## Final verification

- [ ] **Run the full suite:** `python -m pytest -q` — all green (≈ original 41 + new ~30).
- [ ] **CLI parity:** `python main.py --type audio --language en` produces `transcripts/<stem>.txt` identical in shape to pre-change output.
- [ ] **GUI end-to-end:** `python app.py` — upload (single + batch), model/language(+auto), cleanup/diarize gating, live progress, txt+srt+vtt download, history list/search/open/download/delete, one-at-a-time queue, MediaDecodeError + Ollama-down + no-HF-token all degrade with clear messages.
- [ ] **Dependency safety:** `python -c "import numpy; print(numpy.__version__)"` still reports `2.0.2`.

---

## Self-review (completed against the spec)

**Spec coverage:**
- §3 seam `transcribe_file` + single-slot cache → Tasks 6, 7. ✓
- §3.1 evict-on-model-change → Task 6 (`test_model_change_evicts_and_reloads`). ✓
- §3.2 progress callback → Task 3. ✓
- §4 modules table (service, formatter srt/vtt, history, app, asr_engine callback, steps_output override, main refactor, requirements-gui) → Tasks 2–14. ✓ Plus cached diarization backend (§4) → Task 7. ✓
- §5 GUI (upload/validation, model+download notice, language+auto-detect, cleanup/diarize checkboxes with HF gating, progress, raw + clean tabs, downloads, media player, copy, history table) → Tasks 11, 15, 16, 17, 18. ✓
- §6 stem sanitization + collision-safe naming + CLI keeps plain stem → Tasks 2, 5, 7. ✓
- §7 error taxonomy (MediaDecodeError distinct; Ollama warning-not-fatal; diarize greyed; log-tail) → Tasks 1, 4, 14, 15, 18. ✓
- §8 history schema + atomic writes + delete-removes-files + search → Task 9. ✓
- §9 tests (service incl. params-untouched + cache; history; srt/vtt; Arabic sanitization) → Tasks 2, 6, 7, 9, 13. ✓
- §10 phasing → Phase 1 = Tasks 1–12; Phase 2 = Tasks 13–19. ✓
- §11 dependency rule + pillow gate → Task 10. ✓

**Placeholder scan:** no TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `transcribe_file`/`TranscribeResult` field names match between Task 7 and the `app.py` consumers (Tasks 11/15); `metadata["outputs"]` keys (`txt/clean/srt/vtt`) match between `OutputStep` (Tasks 5/14), `transcribe_file` (Task 7), and `HistoryEntry.outputs` (Tasks 9/11). `progress_callback` signature `Callable[[float], None]` consistent across asr_engine, TranscriptionStep, service, app. ✓
