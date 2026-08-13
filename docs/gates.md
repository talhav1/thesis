# Gate ritual

The procedure at every phase boundary. It exists because the two most valuable
documents in this project — `docs/discrepancies.md` and `MEETING_RESULTS.md` —
were both produced by *checking* rather than by writing, and checking only
happens if it is a scheduled step.

---

## Before the run

1. **Resolve the previous gate's open items.** `docs/claims.md` "Unresolved"
   rows and any `UNRESOLVED` block in a config.
2. **Pre-register the precision.** Decide the target Monte Carlo standard error
   and the replicate count *before* running, and put both in the config's
   `note`. A run whose replicate count was chosen after seeing the results is
   screening, whatever its size.
3. **Write or update the config** in `configs/experiments/`. No run is launched
   from ad-hoc flags alone.
4. **Commit. The tree must be clean.** `require_clean_tree` will refuse
   otherwise, and that refusal is the point.
5. **Tag it**: `git tag phaseN-gate && git push --tags`.

## The run

```bash
python experiments/run.py configs/experiments/<name>.json
```

Launch detached for anything over a few minutes. Do not hold an interactive
session open waiting for it.

## After the run

6. **Lint**: `python tools/manifest_lint.py --strict` must pass.
7. **Write the validation report from the summaries.** Open the CSVs. Do not
   write a number from memory or from a previous draft.
8. **Audit in a fresh session.** New context, instruction: *check every claim
   in this report against `results/`; where they disagree, the results file
   wins; list the disagreements rather than correcting them.* This is the step
   that catches overclaiming, and it works because the auditing session has no
   draft to defend.
9. **Append** to `docs/discrepancies.md` (new D-entries) and `docs/decisions.md`
   (new DEC-entries); **regenerate** `docs/claims.md`.
10. **Sync to the chat project** — upload the updated `manuscript/` and `docs/`
    files, and delete the superseded copies. Two versions of one report in the
    project is worse than one that is a week old.
11. **Then** take it to the advisor.

---

## Gate conditions

| Gate | Condition | Status |
|---|---|---|
| Phase 0 | Advisor agrees the dependence question is closed and the estimand is a physical quantile | passed |
| Phase 1 | Baseline matches published rankings within MC uncertainty, or differences are explained | passed, one documented exception (D13) |
| Phase 2 | Reference passes SBC; if only the particle method fails, computational approximation becomes a chapter | passed |
| Phase 3 | At least one reproducible interaction where adaptivity materially changes quantile reliability vs a non-adaptive design | passed (C5) |
| Phase 4 | Invariants 1–2 proved in full; 3–5 as propositions under explicit regularity assumptions | not started |
| Phase 5 | Reliability model evaluated on held-out **DGP families**, not held-out repetitions | not started |
| Phase 6 | Material reduction in worst-case false certainty or decision loss, efficiency trade-off reported, no serious degradation under correct probit | not started |
