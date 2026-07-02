# [ARCHIVED 2026-06-21] Pre-groundwork CLAUDE.md
> Superseded by the groundwork constitution at repo root (adopted at cd888bf).
> Its content (architecture seam, bilingual-docs rule, don't-implement-from-archive) was folded into the new CLAUDE.md §1/§7. Kept here for the audit trail.

# Project guidance for agents

## Architecture (GUI + CLI)

Both the CLI (`main.py`) and the web GUI (`app.py`) call the single in-process
seam `src/service.py::transcribe_file(...)`, which builds and runs the existing
step-pipeline for one file with per-job parameters. `configurations/params.yaml`
is never mutated at runtime. Whisper models and the diarization backend are
cached (single resident model) in `src/service.py`.

## Docs

User-facing docs are bilingual: `README.md` (English, the default GitHub renders)
and `README.ru.md` (Russian). Keep the two in sync when editing either. Superseded
plans, specs, and snapshots live in `docs/archive/` and are historical — don't
implement from them.
