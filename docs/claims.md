# Claims ledger

Every claim the project makes, its status, the file that backs it, and the
precision that produced it. **Regenerated at each gate**, superseding the
previous version — unlike `discrepancies.md` and `decisions.md`, which are
append-only.

Seeded from `MEETING_RESULTS.md` (Phases 0–3). Where a write-up and a results
file disagree, the results file wins and the disagreement is recorded rather
than silently corrected.

**Status vocabulary**

| Status | Meaning |
|---|---|
| **Established** | confirmatory tier, or an invariant enforced by a test |
| **Preliminary** | screening precision only (30 or 22 replicates; coverage SE 0.03–0.11) |
| **Negative** | the experiment ran and the effect is absent |
| **Unresolved** | the data do not settle it |

**Last regenerated:** Phase 3 gate. Test suite at that point: 104 passed,
1 deselected.

---

## Standing claims

| ID | Claim | Status | Tier | Evidence | Notes |
|---|---|---|---|---|---|
| C1 | Adaptive dependence does not invalidate the product likelihood under an ignorable policy | **Established** | invariant | `tests/test_factorization.py`, `src/factorization.py` | Constant to $<10^{-9}$ over a 72-pt grid; TV $<10^{-12}$. Non-ignorable oracle negative control breaks it, so the test is not vacuous. |
| C2 | Rotem's baseline reproduces for five of six designs | **Established** | 250 reps/cell | `results/summaries/phase1_vs_published.csv` | 122/160 within 2 SE; excluding `entropy_sigma`, 120/132, max \|z\| 3.39. |
| C2b | `entropy_sigma` does not reproduce in any setting | **Unresolved** | 250 reps/cell | `docs/discrepancies.md` D13 | Criterion verified exact against a closed form, so the discrepancy is in the comparison. Needs Rotem's code or a decision to report the degeneracy as a finding. |
| C3 | The reference posterior is calibrated with the adaptive design in the loop | **Established** | 600 SBC draws | `results/summaries/phase2_sbc_uniformity.csv` | Passes on every target under both designs. |
| C4 | The particle posterior is close to the reference but measurably worse | **Established** | 1,350 draws | `results/summaries/phase2_coverage_map.csv`, `results/raw/phase2_coverage_map_raw.parquet` | Makes computational approximation a chapter, not a defect. |
| C5 | Tail false certainty appears under adaptive design with a tail-perturbed curve | **Established** | confirmatory, CRN-paired | `results/summaries/phase3b_confirm_summary.csv`, `phase3b_confirm_h2.csv`, `results/raw/phase3b_confirm_raw.parquet` | Strongest result in the project. Demonstrated on **one** constructed curve family at **one** horizon — that is the main exposure. |
| C6 | `robit` shows the same signature at roughly half the size | **Preliminary** | screening | `results/summaries/phase3b_screen_summary.csv` (Block I and II) | Suggestive of generality, not confirmatory. |
| C7 | Prior misspecification is not the driver of the coverage loss | **Preliminary** | screening | `results/summaries/phase3b_screen_summary.csv` (Block II, prior sub-block) | See §Corrections item 4. |
| C8 | Turning rejuvenation off beats Rotem's rule despite 30× lower ESS | **Established** | — | `results/summaries/phase3a_attribution.csv`, `phase3a_attribution_raw.csv` | Corner case: ~1.6% unconditional frequency, zero for $\sigma \ge 1.2$. Concrete fix available (trigger on ESS, or drop the KL threshold to 0.02). |
| C9 | The reference posterior is adequate throughout Phase 3 | **Established** | 24-history audit | `results/summaries/phase3_reference_audit.csv`, `phase3_reference_audit_summary.csv` | Rests **entirely** on this audit: `ref_converged` is null in all 6,960 screening and 2,700 confirmatory rows. |
| C10 | The exploratory control does not restore coverage | **Negative** | screening | `results/summaries/phase3b_screen_summary.csv` | So a safeguard is a research contribution, not a write-up task. |

---

## Corrections carried forward

Recorded so a later reader does not rediscover them as new problems.

1. **`n_replicates_completed` is a row count in two manifests.** `phase1_horizon`
   reports 900 (= 300 × 3 slices), `phase3b_screen` reports 6,960 (= 2,320 × 3).
   Now prevented by `replicates_unit` (see `CLAUDE.md` §3).
2. **`seed_range` is fictional in every legacy manifest.** The code seeds with
   `default_rng([seed_base, replicate])`. The field is removed and the linter
   now rejects it.
3. **No `run_id` in any legacy manifest.** Provenance gap, now closed.
4. **Prior sub-block coverage range misattributed.** The calibrated cells span
   0.500–0.667; 0.455 is a *shifted*-prior cell, counted twice. The conclusion
   (prior misspecification is not the driver) is unaffected.
5. **D16 has an unrecorded prior analogue.** $w^\ast$ was frozen from the
   calibrated-prior, $\sigma_0=3$ cell, which makes false certainty
   incomparable across the **scale** sub-block — and, by the same argument, the
   **prior** sub-block, since narrow-prior intervals are narrower for prior
   reasons (6.90 vs 7.47 at $q_{0.99}$).
6. **Phase 3A and 3B interval widths are not comparable.** `_invert_cdf`
   bisection settings changed in commit `9f1d16f`, the commit containing the
   Phase 3A outputs, and every manifest records a dirty tree. Coverage, rank
   statistics and SBC are unaffected — `covered()` reads the rank statistic.
7. **$w^\ast$ carries its own Monte Carlo error.** Estimated as the 25th
   percentile of interval width from 30 replicates; applied to the
   300-replicate confirmatory probit control, 29.7% of runs fall below it
   rather than the nominal 25%.
8. **28 of 31 manifest-referenced artifacts were absent from the repository as
   published on GitHub.** Closed at D24: the artifacts existed all along in
   the local working copy the upload was made from (`~/Downloads/thesis-
   reliability`), which was never fully pushed. All 10 raw parquet files and
   all 30 summary CSVs are recovered from there, byte-identical to what the
   manifests describe.

---

## Reproducibility gap — closed (D21 opened it, D24 closed it)

Manifests reference 31 artifacts; **all 31 are now present** in
`results/raw/` and `results/summaries/`, recovered from the local working
copy at `~/Downloads/thesis-reliability` (a `git`-tracked checkout that
predates the GitHub upload and was never itself pushed). See
`docs/discrepancies.md` D24 for the recovery mechanics, including two raw
parquet files (`phase3a_histories.parquet`, `phase3b_designs.parquet`) that
were deleted-but-uncommitted even in that source and had to be restored with
`git restore` from its own history.

C2–C10 are now independently checkable against the actual original files, not
a reconstruction. `docs/discrepancies.md` D23's log-reconstruction was spot-
checked against these originals during the D24 recovery and matched to the
precision the logs printed at — the two didn't disagree on anything, they
just differed in how much precision survived. D23's `docs/recovered_from_logs/`
output is removed as redundant now that the originals are in `results/`; the
ledger entry itself stays, since discrepancies are append-only.

Legacy manifests still record absolute paths from a machine that no longer
exists (`/home/claude/thesis-reliability/...`), so `manifest_lint.py --strict`
still flags every legacy manifest's artifacts as "missing on disk" — that
check resolves the literal recorded path, not a repo-relative fallback. This
is a pre-existing, already-documented limitation (D19/D21), not something the
D24 recovery changes; the artifacts are confirmed present by direct
filesystem listing instead.

**Still open:** every legacy manifest is unpinned (D19) and none carries a
`run_id` (D20) — recovering the files does not retroactively fix the
provenance metadata around them. A clean re-run under the current contract
(option 3, previously) is no longer needed to *recover data*, but is still
the only way to get pinned, `run_id`-bearing manifests for these results.
