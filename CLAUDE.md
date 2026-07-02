# Local Whisper Web GUI — CLAUDE.md
> The constitution. Read this first, every session. It governs how work is planned, executed, verified, and recorded. When in doubt, the rules here win over convenience.

**What this is:** a local, single-user Gradio web GUI layered over the existing Whisper CLI transcription pipeline — upload a media file, pick model + language, transcribe with live progress, view/download the transcript (txt; srt/vtt in Phase 2), and browse a searchable history. Both the GUI and the CLI call one in-process seam so the CLI stays unchanged.
**Stack:** Python 3.9.6 · openai-whisper 20250625 · Gradio 4.44.1 (`gradio>=4.44,<5`) · moviepy · pyannote.audio (optional) · Ollama (optional) · pytest
**Commands:** test `python -m pytest -q` · lint `(none configured)` · typecheck `(none configured)` · build/GUI-smoke `python -m pytest tests/test_app_smoke.py -q` (app is interpreted — no compile step)

> **Adopted at** `cd888bf` (branch `feat/web-gui`) · 2026-06-21 — this repo is mid-build (Phase 1 core landed; Phase 2 in progress). See `docs/BUILD-STATE.md` and `docs/DECISIONS.md` (D-002) for the adoption baseline.

---

## 1. Working model

- **Human (Ahmed)** — the decision-maker. Sets direction, makes severity/scope calls, merges PRs.
- **Planner (this Claude)** — strategy and judgment: writes the spec, picks root causes, designs the chunks, cross-validates claims against source. Produces docs, not commits.
- **Executor (Claude Code)** — implements one chunk at a time against the live repo under the loop below.
- **Sub-agents** — spawned for the adversarial gate (independent verify + adversarial review + a verdict-auditor). This mirrors the existing quality protocol: after a task's TDD passes, fresh-context skeptic sub-agents attack the code with new failing tests (edge cases, concurrency, malformed/Unicode input, runtime-value bugs) until a full round finds nothing and the suite is green, or it escalates after three rounds.

Docs are the contract between these roles — **not** chat memory. If it isn't written in the doc set, it didn't happen.

## 2. STEP 0 — every session, before any work

1. Read, in order: `CLAUDE.md`, `docs/LEARNINGS.md`, `docs/BUILD-STATE.md` (tail), `docs/DECISIONS.md` (tail).
2. Sync the repo (`git fetch`); report the current branch + HEAD short SHA + whether the tree is clean.
3. Run the full suite once for a baseline (`python -m pytest -q`); report the green counts.
4. Print a short situation report (SHA, baseline, the next chunk, any LEARNINGS rule relevant today). Then wait for the chunk.

**Never plan or fix on a stale tree or a red baseline.** A red baseline is fixed or surfaced before anything else.

## 3. Chunk discipline

- **One chunk = one PR off `feat/web-gui`** (the live integration branch; not `master`). **STOP after each** for explicit human review/merge. Do not batch chunks.
- Commit **only the fix + its test** (one logical change). A correction to a prior claim/doc is a **separate** commit (see §8).
- Branch per chunk; re-pull the integration branch before branching the next.

## 4. The chunk loop (full procedure in `docs/CHUNK-PROMPTS.md`)

Every chunk runs: **re-ground → RED → GREEN → independent verify → adversarial gate → docs → STOP.**

- **Re-ground** — confirm the diagnosis against live source *before* coding. A "confirmed" claim in a spec is a **lead, not a fact**. If the code disagrees, stop, say so, correct the spec (separate commit), and proceed from truth.
- **RED** — write the failing test first; prove it fails for the *right reason*. The test author does not edit production code during the fix.
- **GREEN** — the smallest change that fixes the root cause. The implementer does not edit the test. Full suite green.
- **Independent verify** — stash the fix only (keep the test) → the test must fail again → restore. Proves the test catches the bug.
- **Adversarial gate** — sub-agents try to break it (regressions, invariant violations, edge cases, over/under-claiming tests). A verdict-auditor gives an honest ship-ready call, no severity drift. Ship only on a unanimous verdict, 0 blockers.
- **Docs** — update the doc set (§7) in the same PR.

## 5. Invariants — "do no harm"  *(every chunk MUST preserve all of these)*

> A change that breaks any invariant is **not** ship-ready, no matter what it fixes. An invariant without a named passing test in `AUDIT.md` is a **blocker** until the test exists (INV-4, INV-5, INV-9 are currently PENDING — see `docs/AUDIT.md`).

- **INV-1 — CLI parity.** `python main.py --type {video,audio} [--language X] [--cleanup] [--diarize|--no-diarize] [--num-speakers N]` produces the same `transcripts/<stem>.txt` (and `<stem>_clean.txt`) as before the GUI existed. The CLI now routes through `src/service.py::transcribe_file` (`main.py:135`) — that refactor must not change CLI outputs. Enforced at: `main.py:48-135`.
- **INV-2 — `configurations/params.yaml` is never mutated at runtime.** Per-job `model`/`language`/`cleanup`/`diarize`/`num_speakers` are function arguments, not config writes. Enforced at: `src/service.py:125-141`.
- **INV-3 — single resident Whisper model (evict-before-load).** `get_whisper_model` holds at most one model; a model change evicts the old one (`gc.collect()`, `torch.cuda.empty_cache()` on CUDA) before loading. Jobs run one at a time, so one slot is sufficient and prevents VRAM blowups. Enforced at: `src/service.py:40-74`.
- **INV-4 — one transcription at a time.** `demo.queue(default_concurrency_limit=1)` — the single-slot cache and one-job assumption depend on this. Enforced at: `app.py:280`. *(PENDING test.)*
- **INV-5 — local-only.** The GUI binds `127.0.0.1` and adds no auth, no cloud, no paid deps. Enforced at: `app.py:281`. *(PENDING test.)*
- **INV-6 — filenames are Unicode/Arabic-safe and never overwrite.** Output stems are NFC-normalized, illegal/control chars scrubbed, Windows reserved device names guarded, and basenames are collision-safe (`<stem>__<model>` + short job-id on collision). Arabic/CJK is preserved where filesystem-safe. Enforced at: `src/utils/naming.py`, applied at the seam `src/service.py:191-192`.
- **INV-7 — history index is crash-safe and preserves non-ASCII.** Writes go to a `.tmp` file then `os.replace` (atomic); all reads/writes are lock-guarded; JSON is written `ensure_ascii=False` (Arabic stays readable); a corrupt index degrades gracefully, never crashes. Enforced at: `src/history.py:33,46-48`.
- **INV-8 — auto-detect records the real language, never blank.** When `language` is `None`/`""` (auto), the detected language is captured onto the document (`result.get("language") or language or "unknown"`) — never an empty string. Enforced at: `src/transcription/asr_engine.py:137`.
- **INV-9 — the GUI install never downgrades `numpy`, `torch`, or `openai-whisper`.** The ASR stack runs on numpy >=2; gradio is pinned `>=4.44,<5` (and `huggingface_hub<1.0`) precisely so the resolver leaves the ASR stack untouched. If a future gradio needs `numpy<2`, change gradio — never numpy. Enforced at: `requirements-gui.txt` (install-time gate). *(PENDING automated test — gate is manual: `pip install --dry-run -r requirements-gui.txt`.)*
- **INV-10 — a single failing file never aborts a batch.** Batch transcription (GUI and CLI) logs/warns on a file failure and continues to the next. Enforced at: `app.py:165-179`, `main.py` orchestrate loop.

Cross-cutting, always in force:
- **No regressions** — the full suite stays green throughout.
- **Fix shared chokepoints, not call sites** — the seam (`src/service.py`), the history store, the orchestrator's exception capture, and the naming helpers are chokepoints; fix defects there, not per call site.
- **Missing credentials/services never stall** — diarization (HF token / pyannote) and cleanup (Ollama) are optional; build against mocks, mark live steps PENDING with the exact verification to run later.

## 6. Re-grounding & honesty

- Re-confirm every finding against live code before acting on it. The deployed state may differ from any spec.
- Do **not** fabricate a fix for a defect that doesn't reproduce against HEAD. If it doesn't reproduce, record that and ship nothing for it.
- The verdict-auditor enforces honest classification — no inflating a finding to look thorough, no deflating one to look clean.

## 7. Doc-as-contract — the set and each doc's single job

- **`SYSTEM-SPEC.md`** — the *single* canonical "what the system IS right now." One living spec only. Refreshed against HEAD when it drifts.
- **`BUILD-STATE.md`** — append-only checkpoint log. One bracketed entry per finished chunk. An interrupted session resumes from here.
- **`DECISIONS.md`** — `D-NNN`: every ambiguity resolved or scope call made, with rationale and what was ruled out.
- **`LEARNINGS.md`** — `L-NNN`: lessons, written via the **`/learn`** skill. Don't write here off-schema.
- **`AUDIT.md`** — the guardrail table: each invariant ↔ its named passing test ↔ status.
- **`CODEBASE-MAP.md`** — where things live + the chokepoints.

### Project conventions (carried over from the prior CLAUDE.md, now archived)
- **Bilingual docs.** User-facing docs are `README.md` (English, the GitHub default) and `README.ru.md` (Russian). Keep the two in sync when editing either.
- **Don't implement from the archive.** Superseded plans/specs/snapshots live in `docs/archive/` and are historical only.
- **Python 3.9 syntax.** Keep `from __future__ import annotations` in modules using `X | Y` runtime annotations (gradio 5 / dropping 3.9 is out of scope).

### Doc-bloat control
Bloat is **docs that aren't read or aren't true** — not docs that are long. Ledgers (`BUILD-STATE`, `DECISIONS`, `LEARNINGS`) are append-only — read the tail; `/curate` rolls up closed entries to `docs/archive/` when long, never prunes. Current-state docs (`SYSTEM-SPEC`, `CODEBASE-MAP`, `AUDIT`) carry a `> verified-against <SHA> · <date>` stamp; a stamp older than HEAD with unre-checked claims is **stale**. Link, don't copy; lead with the conclusion; never restate code. `/curate` is the standing loop that enforces this.

## 8. Evidence coherence & archiving

- When a prior claim/doc was wrong, commit the correction **separately** — never amend to hide it. The trail is the audit.
- **Archive, don't delete.** Superseded docs move to `docs/archive/` with a dated "superseded by X" note.

## 9. The maintenance loops

- **`/curate`** — gated doc consolidation: reads docs *and* code together, dedupes, archives superseded docs (never deletes), requires explicit human GO. The project already uses it (HEAD is a curate commit). Run after a phase boundary or when docs drift.
- **`/learn`** — append-only lesson capture into `LEARNINGS.md`, safe mid-stream. A recurring lesson is promoted to an enforced guardrail (a new INV + its test). Keep `/learn` (per-task, granular) and `/curate` (phase-boundary consolidation) distinct — do not conflate them.

## 10. Freshness protocol

Every planning session **starts** with a sync and a reported SHA. Every executor session **ends** by writing the `BUILD-STATE` checkpoint with a `NEXT UP:` line. If a session is interrupted mid-gate, the next re-runs verify + adversarial review from the branch state before trusting any earlier "green."

## 11. Final-report format (end of a run)

Per-chunk summary · final test counts · the AUDIT table (invariants ↔ tests) · every PENDING item · the human's exact next actions.

---

*This file is the operating system. Change it deliberately (via a `D-NNN`), not casually.*
