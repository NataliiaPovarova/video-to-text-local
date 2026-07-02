# Local Whisper Web GUI — BUILD-STATE
> Append-only checkpoint log. One bracketed entry per finished chunk, written in the SAME PR. An interrupted session resumes from the latest `NEXT UP`. Never rewrite history here — append. Growth is not bloat; you read the tail. When the live tail gets long, `/curate` rolls up closed entries into a dated `docs/archive/` digest (never deletes them).

**Entry format:**
```
[<CHUNK-ID> | <one-line what + why> | STATUS: done — branch <name> off <sha>. <root cause>. <the fix>.
 NAMED TEST: <file::test>. PROOF: RED→GREEN + stash-revert. GATE: <verdict, 0 blockers>.
 DOCS: <D-NNN / L-NNN / AUDIT row>. GREEN: <suite counts>; lint+build clean. NEXT UP: <next chunk>.]
```

---

[BOOTSTRAP | groundwork operating system adopted onto a midway repo | STATUS: done — docs-only, no source modified. Adopted at HEAD cd888bf (branch feat/web-gui), clean tree.
 SCAN: stack/commands/chokepoints inferred from source (manifests + CI), not guessed. BASELINE (executed, full ASR stack, Linux/Py3.12): 114 passed, 1 failed, 0 collection errors — the 1 failure is the non-portable Windows test (finding #1, see SYSTEM-SPEC "Known open issues" #1 / D-002), not a code regression; green on the Windows/Py3.9 target. INVARIANTS: INV-1..INV-10 surfaced from source, confirmed, mirrored into CLAUDE.md §5 + AUDIT.md (7 GREEN, 3 PENDING: INV-4, INV-5, INV-9). DOCS GROUNDED: SYSTEM-SPEC + CODEBASE-MAP from actual code with file:line; prior CLAUDE.md folded + archived to docs/archive/CLAUDE.pre-groundwork.md. .gitignore un-ignores the contract docs (D-003).
 NEXT UP: human reviews/merges the docs-only adoption PR; then STEP 0, then the first chunks — (1) guard the non-portable Windows test (finding #1), (2) add the INV-4/INV-5 PENDING tests, (3) optional minimal pytest CI + INV-9 version-assert.]

[C-001 | guard the non-portable Windows ffmpeg test (D-002 finding #1) | STATUS: done — branch fix/windows-ffmpeg-test-guard off d477a50 (docs-only adoption commit). ROOT CAUSE (confirmed vs source): the test monkeypatches the singleton sys.platform="win32" (src.utils.system.sys IS the global sys), so stdlib shutil.which sees win32 too and on Python >=3.12 dereferences _winapi (None off-Windows) -> crash; green on the Windows/Py3.9 target and on this Py3.9 host. FIX: @pytest.mark.skipif(sys.platform != "win32", ...) on the test — predicate evaluated at collection vs the real platform, so it still RUNS on win32 and skips cleanly elsewhere. Test-infra only: no source touched, no INV affected (chose skipif over a _winapi mock — version-stable vs version-fragile).
 NAMED TEST: tests/test_utils.py::TestEnsureFfmpegOnPath::test_windows_uses_exe_name_and_copy_fallback (now platform-guarded). PROOF: off-Windows 110 passed/1 skipped -> 109 passed/2 skipped, 0 failed; the test SKIPS off-Windows with a visible reason and RUNS on win32. The Py3.12 crash (RED) was reasoned, not reproduced — no Python 3.12 on the executor host; the GREEN-equivalent was demonstrated (clean skip + 0 failed + predicate keyed to the real platform at collection). GATE: 3 fresh-context skeptics + verdict-auditor — unanimous ship-ready, 0 blockers, severity low (one cosmetic nit on the reason string was checked and found accurate: the monkeypatch is global).
 DOCS: D-002 finding #1 -> resolved (DECISIONS Resolutions); SYSTEM-SPEC "Known open issues" #1 -> resolved. AUDIT: unchanged (no INV affected). GREEN: 109 passed, 2 skipped, 0 failed; build (gradio smoke) not run here — gradio absent in this env. NEXT UP: chunk (2) — add the INV-4 (queue concurrency=1, app.py:280) and INV-5 (127.0.0.1 bind, app.py:281) PENDING tests; then (3) optional minimal pytest CI + INV-9 version-assert.]
