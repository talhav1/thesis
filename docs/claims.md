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
| C2b | `entropy_sigma` does not reproduce in any setting | **Unresolved** | 250 reps/cell | `docs/discrepancies.md` D13 | Criterion verified exact against a closed form (efficiency 1.000, corr 0.996), and the published table is provably inconsistent with the published criterion: her RMS error for $\mu$ is 1.34 from a prior 8 units off, which a design maximising $I(\sigma;Y\mid x)$ cannot achieve. Unresolved is now about *attribution* — text vs code — not about whether this implementation is right. Needs Rotem's code. |
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
8. **Manifest-referenced artifacts are all present.** All 10 raw parquet
   files and all 30 summary CSVs referenced by manifests exist in
   `results/raw/` and `results/summaries/`.

---

## Reproducibility gap — closed

Manifests reference 31 artifacts; **all 31 are present** in `results/raw/`
and `results/summaries/`. C2–C10 are independently checkable against the
original files.

`manifest_lint.py --strict` resolves each manifest's recorded artifact path
against the current checkout (repo-relative paths directly; legacy absolute
paths from a machine that no longer exists by re-rooting at the first
recognised top-level repo directory) and reports zero "missing on disk"
lines.

**Still open:** every legacy manifest is unpinned (D19) and none carries a
`run_id` (D20). This is intentional (`docs/decisions.md` DEC-10) rather than
a bug — backfilling it would fabricate provenance the original runs never
had. A clean re-run under the current contract is the only way to close
these two for good.
