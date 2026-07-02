# Local Whisper Web GUI — DECISIONS
> `D-NNN`: every ambiguity resolved or scope call made. Each entry: the finding/question, the decision, the rationale ("why this shape"), and what was ruled OUT of scope. Temptations and out-of-scope items are logged here and skipped — not built. Decisions are never silently reversed; superseding one gets a new D-NNN that references the old. Append-only ledger — never prune; `/curate` rolls up superseded decisions into a `docs/archive/` digest when the live file gets long.

---

## D-001 — adopt the groundwork operating system
**Decision:** this project runs the chunk-discipline loop. `CLAUDE.md` is the constitution; the doc set is the contract; `docs/CHUNK-PROMPTS.md` drives execution; `/curate` and `/learn` maintain docs and lessons.
**Why:** disciplined, self-correcting build loops with named-test proof and an adversarial gate produce auditable progress and accumulate judgment instead of drifting. The project already practices the compatible halves (a `/curate` commit is at HEAD; LEARNINGS/`/learn` are in use) — groundwork formalizes the full loop.
**Out of scope:** none.

## D-002 — adoption baseline at `cd888bf` (state + surfaced findings, NOT fixed here)
**Decision:** record the repo's state at adoption so future drift is measurable, and log scan findings as tracked items rather than fixing them during a read-only adoption. Fixes happen later as chunks under RED→GREEN.
**Baseline:** branch `feat/web-gui`, HEAD `cd888bf`, clean tree. Executed `python -m pytest -q` with the full ASR stack (torch/whisper/moviepy/pyannote/gradio) in a Linux/Python-3.12 sandbox → **114 passed, 1 failed, 0 collection errors**. numpy stayed ≥2 after the GUI install (INV-9 held).
**Findings (tracked, to become chunks):**
1. **Non-portable Windows test (finding #1, low severity).** `tests/test_utils.py::TestEnsureFfmpegOnPath::test_windows_uses_exe_name_and_copy_fallback` simulates `win32` on a non-Windows host; Python 3.12's `shutil.which` then dereferences `_winapi` (None on Linux) and crashes. Passes on the Windows/Py3.9 target. Fix: add `@pytest.mark.skipif(sys.platform != "win32")` or mock `_winapi`. This is the single non-green item in the baseline; it is an environment-portability gap, not a product defect.
2. **No automated test gate (med severity).** Only `.github/workflows/release.yml` (semantic-release on `master`) runs in CI; `npm test` is a no-op. Nothing runs `pytest` automatically, yet the loop assumes a green baseline. Fix: add a minimal pytest CI workflow.
3. **INV-4 / INV-5 unprotected (med).** Queue concurrency=1 (`app.py:280`) and the `127.0.0.1` bind (`app.py:281`) have no named test. Fix: add the two PENDING tests (AUDIT).
4. **INV-9 has no automated guard (med).** The numpy/torch/whisper no-downgrade rule is enforced only by a manual `pip install --dry-run`. Fix: a CI/test version-assert after the GUI install.
5. **No linter/typecheck (low).** No ruff/flake8/mypy/pre-commit configured. Adding one is optional and **out of scope** until the human decides.
**Resolutions:**
- **Finding #1 (non-portable Windows test) — RESOLVED by chunk C-001** (branch `fix/windows-ffmpeg-test-guard`, fix commit `f3ffc7c`). Added `@pytest.mark.skipif(sys.platform != "win32", ...)` to the test. Root cause confirmed against source: the test monkeypatches the singleton `sys.platform` (`src.utils.system.sys` *is* the global `sys`), so stdlib `shutil.which` sees `win32` too and dereferences `_winapi` (`None` off-Windows) on Python ≥3.12. The guard's predicate is evaluated at collection vs the real platform, so the test still runs on win32 and skips elsewhere; off-Windows suite is 0 failed (109 passed, 2 skipped). Chose `skipif` over a `_winapi` mock (version-stable vs version-fragile). No production code changed; no INV affected. RED (the Py3.12 crash) was reasoned, not reproduced — the executor host has no Python 3.12; the GREEN-equivalent (clean skip + 0 failed + collection-time predicate) was demonstrated. Adversarial gate: 3 fresh-context skeptics + verdict-auditor, unanimous ship-ready, 0 blockers, severity low. Findings #2–#5 remain open (separate chunks).

**Out of scope (now):** fixing 2–4 inside the adoption PR (they are separate chunks); adding lint/typecheck (5); any source change at all during adoption.

## D-003 — un-ignore the contract docs in `.gitignore`
**Decision:** as part of the docs-only integration, edit `.gitignore` to stop ignoring `docs/` and remove the duplicate/dead `CLAUDE.md` entries (it was listed three times, plus two dead Windows-backslash paths `.cursor\rules\openmemory.mdc` that git never matches). All functional ignores (`venv/`, `.env`, `transcripts/`, `logs/`, `__pycache__/`, `*.pyc`, editor-rule files, `node_modules/`, etc.) are preserved.
**Why:** the groundwork docs ARE the contract and must be committed and reviewable. With `docs/` ignored, the new `SYSTEM-SPEC`/`BUILD-STATE`/`AUDIT`/etc. would be untracked and the loop would silently lose its contract. The existing `docs/archive/*` were only present because they were force-added; relying on `git add -f` per file is error-prone. `CLAUDE.md` was already tracked (committed over the ignore), so the dead ignore lines were pure cruft and a latent footgun (a future `git rm --cached` would drop it).
**Rationale for doing it during adoption:** `.gitignore` is repo configuration, not source code, and this change is what makes the docs-only PR meaningful. It deletes nothing and is shown in the PR diff for review.
**Out of scope:** rewriting the rest of `.gitignore`; un-tracking anything; touching the `.env`/secrets rules.
