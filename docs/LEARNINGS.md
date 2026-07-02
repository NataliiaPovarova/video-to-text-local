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
