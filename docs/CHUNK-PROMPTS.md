# Local Whisper Web GUI — CHUNK PROMPTS
### The process as paste-able prompts. Run one chunk at a time in Claude Code.

Governed by `CLAUDE.md`. One chunk = one PR off `feat/web-gui` (the live integration branch), STOP after each. Commit only the fix + its test; corrections are separate commits.

**How to read this file.** Not every section is a step you run. **Runnable prompts:** §1 bootstrap, §3 filled per chunk. **Reference** (applied *inside* the prompts, never run alone): §2 model tiering, §4 sub-agent briefs, §5 merge gate, §6 feature variant. Order: §1 once per session, then §3 per chunk.

---

## §1 — STEP 0: session bootstrap (run every session, before anything)

```
STEP 0 — bootstrap. No code yet.
1. Read, in order: CLAUDE.md, docs/LEARNINGS.md, docs/BUILD-STATE.md (tail), docs/DECISIONS.md (tail).
2. git fetch && git status. Report: branch, HEAD short SHA, tree clean?
3. Baseline: run python -m pytest -q; report the green counts. (No separate lint/typecheck/build — the GUI build-smoke is `python -m pytest tests/test_app_smoke.py -q`.)
4. Print a 5-line situation report: SHA, baseline counts, the next chunk, any LEARNINGS rule relevant today.
Wait for the chunk prompt. If the baseline is RED, STOP and surface it — do not start work on a red tree.
   (Note: on a non-Windows host, expect the one known Windows-only failure from D-002 finding #1 until it is guarded.)
```

## §2 — Model tiering (reference, not a step)

> Configuration the chunk prompts assume whenever they say "spawn a sub-agent." Set once.

- **Lead orchestrator** — the strongest model. Owns the chunk, spawns sub-agents, makes the merge call.
- **Reasoning-heavy roles** (implementer, logic reviewer, verdict-auditor) — strongest model.
- **Mechanical roles** (re-run suite, diff-check, repeat ×3 for flakiness) — a faster model is fine.
- Ship only on a **unanimous** verdict, **0 blockers**.

## §3 — THE UNIVERSAL CHUNK PROMPT (fill the {{...}} and paste)

```
CHUNK: {{CHUNK_ID}} — {{TITLE}}
Spec: {{SPEC_REF}}. Branch: {{BRANCH}} (off feat/web-gui).
You are the lead orchestrator. Follow this exactly; STOP at the end.

STEP A — RE-GROUND (no code yet).
- Open the files in the spec. Confirm the root cause AGAINST SOURCE. A "confirmed" claim is a lead, not a fact —
  it may be stale or the deployed state may differ. If the code disagrees, STOP, tell me what's actually true,
  correct the spec (separate commit), and proceed from truth.
- Restate: exact file:line of the cause, the minimal fix, and which invariant (CLAUDE.md §5) it must preserve.
- If the defect does NOT reproduce at HEAD, record that and ship nothing for it. Do not invent a fix.

STEP B — RED (test first).
- Spawn a Test-Author sub-agent (§4): write the named failing test. Prove it fails for the RIGHT reason.
  The test author does NOT edit production code.

STEP C — GREEN (minimal fix).
- Spawn an Implementer sub-agent (§4): smallest change that fixes the root cause. Does NOT edit the test.
- Prove GREEN: named test passes; FULL suite green (python -m pytest -q).

STEP D — INDEPENDENT VERIFY (stash-revert).
- Spawn a Verifier sub-agent (§4): stash the FIX only (keep the test) → the test must FAIL → restore. Paste both.
  Re-run the full suite green after restore.

STEP E — ADVERSARIAL GATE.
- Spawn 2–3 Adversarial-Reviewer sub-agents (§4): try to BREAK the fix — regressions, invariant violations,
  edge cases, over/under-claiming tests, scope creep beyond the root cause. Each returns booleans + blockers.
  (Bias their attacks at: concurrency/queue, malformed/Unicode/Arabic input, crash-safety of the history index,
   auto-detect language edge cases, and CLI-parity regressions — the project's historical bug classes.)
- Spawn a Verdict-Auditor sub-agent (§4): honest ship-ready call, no severity drift, 0 blockers required.
  If any blocker is found, FIX and re-run the gate. Do not rationalize past a real blocker.

STEP F — DOCS (same PR).
- BUILD-STATE.md: one bracketed checkpoint (what was done · named test · final green counts · NEXT UP).
- DECISIONS.md: D-{{NEXT}} if the chunk resolved any ambiguity/scope question (rationale + what's out of scope).
- LEARNINGS.md: if a lesson emerged, capture it via /learn (L-{{NEXT}}, recurrence + severity). Off-schema = no.
- AUDIT.md: if an invariant was reinforced/added (e.g. an INV-4/5/9 PENDING flips to GREEN), update its row.
- SYSTEM-SPEC.md: if current reality changed, refresh + restamp verified-against.

STEP G — COMMIT + STOP.
- Commit ONLY the fix + its test on {{BRANCH}}. A STEP-A correction is a SEPARATE commit.
- Open the PR with: chunk ID, root cause, the named test, the stash-revert proof, the gate verdict, docs touched.
- STOP. Do not start the next chunk. Print the one-line resume pointer.
```

## §4 — Sub-agent role briefs (paste into each spawned agent)

**Test-Author**
```
Write ONLY the named test for this chunk. It must fail on current code for the RIGHT reason (assert the real
behavior, not an incidental). Use the existing pytest harness (tests/, conftest.py). Do not touch production code.
Return test + RED output.
```
**Implementer**
```
Make the named test pass with the smallest change that fixes the stated root cause. Preserve every invariant in
CLAUDE.md §5. Do NOT edit the test. No unrelated changes. Return the diff + GREEN output + full-suite green.
```
**Verifier (independent)**
```
Stash the FIX only (keep the test). Run the named test — it MUST fail (proves the test catches the bug). Restore.
Re-run the FULL suite — it MUST be green. Return both outputs. You wrote neither the test nor the fix.
```
**Adversarial-Reviewer (×2–3)**
```
Try to BREAK this fix. Check: regressions; violations of any CLAUDE.md §5 invariant; edge cases the test missed
(concurrency, malformed/Unicode/Arabic filenames, corrupt history index, auto-detect blank-language, CLI parity);
whether the test over/under-claims; whether the fix touches more than the root cause. Return JSON:
{ regression:bool, invariantBroken:bool, testHonest:bool, scopeMinimal:bool, blockers:[...] }.
```
**Verdict-Auditor**
```
Read the reviewers' outputs + the diff + the proofs. Give an HONEST ship-ready verdict, NO severity drift.
Ship-ready requires: full suite green, valid stash-revert proof, every reviewer clear, 0 blockers.
Return { shipReady:bool, reasons:[...], blockers:[...] }.
```

## §5 — Merge gate (Definition of Done)

- [ ] Root cause confirmed against source (STEP A); spec corrected if wrong (separate commit).
- [ ] Named RED test failed before the fix; passes after.
- [ ] Stash-revert proof: removing the fix re-breaks the test.
- [ ] Full suite green (`python -m pytest -q`). GUI changes: `tests/test_app_smoke.py` green.
- [ ] Every CLAUDE.md §5 invariant preserved (reviewers confirm).
- [ ] Adversarial gate unanimous, 0 blockers, no severity drift.
- [ ] BUILD-STATE updated; DECISIONS/LEARNINGS/AUDIT/SYSTEM-SPEC updated where applicable.
- [ ] PR contains only the fix + its test (+ separate correction commit if any).

## §6 — Feature-chunk variant (Phase 2: srt/vtt export, batch upload, search, media preview)

Net-new features insert a design step before RED:

```
STEP A0 — DESIGN (before RED). Short design doc: data model, API/UI surface, contracts, and how it respects every
CLAUDE.md §5 invariant (esp. INV-1 CLI parity, INV-4 one-at-a-time, INV-6 filename safety). Add the DECISIONS entry
(D-{{N}}) capturing the choice + what's out of scope. Get a GO, then proceed B→G, TDD-ing the new surface one
shippable slice at a time (one PR per slice).
```

## §7 — Resume protocol

Every session ends by writing the BUILD-STATE checkpoint with a `NEXT UP:` line. Every session starts with §1. The
docs are the contract — not chat memory. If interrupted mid-gate, the next session re-runs STEP D+E from the branch
state before trusting any earlier "green."
