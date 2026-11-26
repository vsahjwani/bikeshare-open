---
name: seed-fault
description: Deliberately break the feature pipeline in one specific way, check that the test suite catches it, record the real failing output, then revert. Use to prove the tests can fail.
argument-hint: <the fault, e.g. "remove .shift(1) from cnt_roll_7">
---

Seed this fault and see whether the tests catch it: $ARGUMENTS

1. Run `git status --short`. If the tree is not clean, stop and say so. Never touch `data/raw/`.
2. Make the smallest possible edit to `src/features.py` or `src/config.py` that creates the fault described above. Show the diff.
3. Run `python -m pytest tests/ -q 2>&1 | tail -15` and show the output unedited.
4. Report which test failed and why. If every test still passes, say clearly that the fault was NOT caught; do not rewrite the tests to hide it.
5. Commit the broken state with the message "Seeded fault: $ARGUMENTS", then undo it with `git revert --no-edit HEAD`.
6. Append the fault, the test that caught it (or "not caught") and the pasted output to `docs/SEEDED_FAULTS.md`.
