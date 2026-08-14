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
| C3 | The reference posterior is calibrated with the adaptive design in the loop | **Established** | 600 SBC draws | `docs/recovered_from_logs/phase2_sbc_uniformity.csv` (recovered — D23) | Passes on every target under both designs. Original `results/summaries/phase2_sbc_uniformity.csv` is still absent from the repo; this is a full reconstruction from `results/phase2_sbc.log`. |
| C4 | The particle posterior is close to the reference but measurably worse | **Established** | 1,350 draws | `docs/recovered_from_logs/phase2_coverage_map.csv` (recovered — D23) | Makes computational approximation a chapter, not a defect. Reconstructed from `results/phase2_coverage.log`; original summary CSV still absent. |
| C5 | Tail false certainty appears under adaptive design with a tail-perturbed curve | **Established** | confirmatory, CRN-paired | `docs/recovered_from_logs/phase3b_confirm_summary.csv` + `..._h2_paired.csv` (recovered — D23) | Strongest result in the project. Demonstrated on **one** constructed curve family at **one** horizon — that is the main exposure. Full confirmatory table and paired H2 test reconstructed from `results/phase3b_confirm.log`; original summary CSV still absent. |
| C6 | `robit` shows the same signature at roughly half the size | **Preliminary** | screening | `docs/recovered_from_logs/phase3b_screen_summary_blockI_reference.csv` (recovered, **Block I only** — D23) | Suggestive of generality, not confirmatory. Block II, which this claim may also draw on, has no surviving table anywhere in the repo (not even the log) — unverifiable beyond Block I. |
| C7 | Prior misspecification is not the driver of the coverage loss | **Preliminary** | screening | Phase 3B screen — **not independently checkable** | See §Corrections item 4. The prior sub-block this claim rests on is Block II, which is not recoverable (D23) — original summary CSV and log table both absent. |
| C8 | Turning rejuvenation off beats Rotem's rule despite 30× lower ESS | **Established** | — | `docs/recovered_from_logs/phase3a_attribution.csv` (recovered — D23) | Corner case: ~1.6% unconditional frequency, zero for $\sigma \ge 1.2$. Concrete fix available (trigger on ESS, or drop the KL threshold to 0.02). Reconstructed from `results/phase3a_attribution.log`. |
| C9 | The reference posterior is adequate throughout Phase 3 | **Established** | 24-history audit | `docs/recovered_from_logs/phase3_reference_audit.csv` (recovered — D23) | Rests **entirely** on this audit: `ref_converged` is null in all 6,960 screening and 2,700 confirmatory rows. Reconstructed from `results/phase3_ref_audit.log`, including the `AUDIT PASSED` line. |
| C10 | The exploratory control does not restore coverage | **Negative** | screening | `docs/recovered_from_logs/phase3b_screen_summary_blockI_reference.csv` (recovered, **Block I only** — D23) | So a safeguard is a research contribution, not a write-up task. Same Block II caveat as C6. |

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
8. **28 of 31 manifest-referenced artifacts are absent from the repository.**
   See the section below — this is the largest single gap.
9. **Most of correction 8 turned out to be recoverable from committed log
   text, not lost.** `docs/discrepancies.md` D23. `docs/recovered_from_logs/`
   now holds a reconstruction of 10 of the 19 missing summary CSVs, spot-
   checked by hand against source. It is a reconstruction, not the original
   file: it carries whatever rounding the original `print()` call used, has
   no MC SEs beyond what was printed, and cannot supply anything the raw
   parquet alone held.

---

## Reproducibility gap (opened at the provenance audit; narrowed at D23)

Manifests reference 31 artifacts. **Three exist in the repository as
originally published.** Every raw parquet is missing. Of the 22 summary CSVs,
3 exist as originally published and **10 more are now reconstructed** in
`docs/recovered_from_logs/` (D23) from the committed `results/*.log` files —
leaving **9 summary CSVs genuinely unrecoverable** from anything currently in
the repository:

- `phase1_table1_and_3_reproduction.csv`
- `phase2_false_certainty.csv`, `phase2_coverage_map_stratified.csv`,
  `phase2_coverage_stratified.csv`
- `phase3a_paired_summary.csv`, `phase3a_stratified.csv` (a related but not
  identical table, `phase3a_nooverlap.csv`, was recovered instead)
- `phase3b_pseudo_truth.csv` (full per-policy table; the spread-across-
  policies summary `phase3b_pseudo_truth_spread.csv` was recovered)
- `phase3b_screen_summary.csv` (Block I only was recovered; Block II has no
  surviving table anywhere, including the log)

`MEETING_RESULTS.md` states that its ground truth is
`results/summaries/*.csv` and `results/raw/*.parquet`. **The raw parquet
cannot be consulted under any circumstances** — nothing recovers per-replicate
data from a printed pivot table. For the summary layer, **C3, C4, C5, C8, C9
are now independently checkable** via `docs/recovered_from_logs/`; **C6 and
C10 are checkable for Block I only**; **C7 remains unverifiable** (it rests on
the unrecovered Block II prior sub-block).

Legacy manifests also record absolute paths from a machine that no longer
exists (`/home/claude/thesis-reliability/...`), so even a recovered file would
not resolve against them. New manifests store repo-relative paths.

**This is still the highest-priority item in the project** for the raw layer.
Options, in order of preference:

1. Commit the summary CSVs (small, text, diffable) and track the raw parquet
   with Git LFS or a release asset. The 10 D23 reconstructions can seed this
   for the tables they cover; promoting them from `docs/recovered_from_logs/`
   into `results/summaries/` requires going through a runner (or otherwise
   documenting them as a manual, unmanifested exception) to keep the
   provenance contract meaningful — copying them over silently would misrepresent
   them as sanctioned runner output.
2. Commit summaries only, and record raw-file checksums in the manifests so a
   re-run can be verified against the original.
3. Re-run each phase from a clean tree under the new contract, and treat the
   current results as superseded.

Option 3 is the only one that also closes correction 6, and the only one that
recovers the raw parquet and the 9 tables D23 could not reach.
