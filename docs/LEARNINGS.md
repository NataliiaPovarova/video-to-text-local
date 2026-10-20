# Local Whisper Web GUI — LEARNINGS
> `L-NNN`: lessons that change how we work. Written via the **`/learn`** skill (do not hand-write off-schema). A recurring lesson is promoted to an enforced guardrail (a new INV in CLAUDE.md §5 + its test in AUDIT.md). Append-only — `/curate` rolls up closed lessons to a `docs/archive/` digest when the live file gets long; nothing is deleted.

**Entry schema (what `/learn` writes):**
```
## L-NNN — <short rule, imperative>
- **Lesson:** <the rule, stated so a future agent can apply it>
- **Trigger:** <what happened that taught this>
- **Severity:** <low | med | high — blast radius if ignored>
- **Recurrence:** <first time | Nth recurrence of L-MMM>
- **Promotion:** <none | promote to INV-N + AUDIT row>
```

---

(no lessons yet — `/learn` appends here. Candidate seeds already captured as invariants from the pre-implementation review: numpy-no-downgrade → INV-9; filename sanitization → INV-6; auto-detect None→blank → INV-8; batch-abort regression → INV-10. If any recurs, `/learn` records it and may promote a stronger guard.)

## L-001 — after two PRs that touch the same code or tests merge, re-run the full suite on `main` before branching the next chunk
- **Lesson:** "Each PR green on its own branch" does not mean "green after both merge". When two PRs stacked or developed in parallel touch the same module, test file or test helper, run `python -m pytest -q` on the merged `main` as soon as both land, before branching the next chunk (CLAUDE.md §2 STEP 0). A textual rebase conflict is the visible risk. The silent risk is a semantic conflict: one PR removes or renames a symbol (function, helper, mock target) that the other PR's new tests still reference. Git merges that cleanly, and only the test run catches it. For new tests, use the shared fixtures/helpers already on `main` (e.g. `_stub_probe`) rather than patching private internals of modules another open PR is rewriting.
- **Trigger:** #42 (C-006) removed `src.pipeline.steps_ingestion._probe_duration` and moved seam tests to `_stub_probe`. #44 (GH-27) was written against pre-C-006 `main` and patched `_probe_duration` in 3 e2e tests. Both PRs were green alone. After both merged, `main` at `3354517` was red: 184 passed, 3 failed (AttributeError in monkeypatch). The GH-27 checkpoint had warned only about a textual conflict in `transcribe_file`. The breakage was found at the next session's STEP 0 and fixed test-only (FIX-BASELINE).
- **Severity:** med — no production defect, but red `main` blocks all chunk work (§2) and can hide a real regression behind "known failures".
- **Recurrence:** first time
- **Promotion:** none yet. Candidate if it recurs: a minimal pytest CI on push/PR to `main` (already listed as optional in the BOOTSTRAP NEXT UP). It would turn this into an automatic gate, but no new INV is needed.
