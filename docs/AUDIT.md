# Local Whisper Web GUI — AUDIT
> The guardrail table: every invariant ↔ its named passing test ↔ status. An invariant with no named passing test is a **blocker** until the test exists. Updated in the same PR whenever an invariant is reinforced or added.

> **verified-against** `cd888bf` · 2026-06-21 — current-state doc; restamp via `/curate` when it drifts. Status reflects the executed adoption baseline (114 passed / 1 failed; the failure is INV-unrelated — see DECISIONS D-002).

| Invariant | Named test | Status |
|-----------|------------|--------|
| INV-1 — CLI parity / routes through the seam | `tests/test_main_batch.py::test_batch_continues_after_one_file_raises`, `::test_batch_continues_after_unexpected_exception` | GREEN |
| INV-2 — `params.yaml` never mutated at runtime | `tests/test_service.py::TestTranscribeFile::test_does_not_mutate_params_yaml` | GREEN |
| INV-3 — single resident model, evict-before-load | `tests/test_service.py::TestModelCache::test_same_model_loaded_once`, `::test_model_change_evicts_and_reloads` | GREEN |
| INV-4 — one transcription at a time (`queue(concurrency_limit=1)`) | *(none — `queue()` not exercised)* | **PENDING** |
| INV-5 — local-only `127.0.0.1` bind, no auth | *(none — `launch()` not exercised)* | **PENDING** |
| INV-6 — Unicode/Arabic-safe + collision-safe filenames | `tests/test_naming.py::test_arabic_unicode_preserved_and_safe`, `::test_illegal_characters_replaced`, `::test_windows_reserved_device_name_prefixed`, `::test_collision_appends_job_suffix`, `::test_different_models_distinct` | GREEN |
| INV-7 — crash-safe + non-ASCII-preserving history index | `tests/test_history.py::test_survives_corrupt_index`, `::test_add_and_list_newest_first`, `::test_delete_removes_record_and_files` | GREEN |
| INV-8 — auto-detect records real language, never blank | `tests/test_asr_engine.py::test_detected_language_captured_when_auto`, `::test_empty_string_language_routes_to_auto`; end-to-end `tests/test_service.py::TestTranscribeFile::test_auto_detect_passes_no_language_kwarg` | GREEN |
| INV-9 — GUI install never downgrades numpy/torch/whisper | *(none — install-time gate only: `pip install --dry-run -r requirements-gui.txt`)* | **PENDING** |
| INV-10 — a failing file never aborts a batch | `tests/test_app_smoke.py::test_run_batch_continues_after_file_failure` (GUI); `tests/test_main_batch.py::test_batch_continues_after_one_file_raises` (CLI) | GREEN |

**PENDING obligations (typical first post-adoption chunks):**
- INV-4 — add a test asserting the Blocks app sets `default_concurrency_limit=1` (inspect `demo`/queue config in a build-smoke test).
- INV-5 — add a test asserting `launch` is invoked with `server_name="127.0.0.1"` (patch `launch`, call the entry, assert kwargs).
- INV-9 — add a CI/test step that runs the GUI install (or `--dry-run`) and asserts numpy ≥2 + torch + openai-whisper versions are unchanged.
