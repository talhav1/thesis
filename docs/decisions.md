# Decision log

Statistical and methodological choices, with the reason and the date, so a
later reader can tell a deliberate choice from an accident — and so a choice
that later turns out to leak can be found before a report inherits it.

**Append-only.** A superseded decision is marked superseded and left in place.

Format: `DEC-n | date | decision | rationale | consequences | status`.

---

### DEC-1 | Phase 0 | The primary estimand is a physical quantile, not $(\mu,\sigma)$

**Rationale.** Under misspecification the fitted probit's parameters have no
stable meaning, while $q_p$ of the true curve does. Also makes the thesis
question decision-relevant.
**Consequences.** Under misspecification `mu` denotes the true curve's median
and `sigma` its standard deviation — Rotem's own choice for the Beta case.
**Status.** Active. Foundational; changing it invalidates Phases 1–3.

---

### DEC-2 | Phase 0 | The dependence-based approximation question is closed

**Rationale.** The product likelihood is exact under an ignorable policy
(C1), enforced as a numerical invariant with a non-ignorable negative control.
**Consequences.** The thesis may not describe Rotem's likelihood as an
independence approximation. Reframed as a short foundational proposition.
**Status.** Active.

---

### DEC-3 | Phase 1 | Coverage is evaluated on the rank statistic, not by interval containment

**Rationale.** Equivalent for a continuous posterior, but the rank form makes
the coverage layer and the SBC layer consistent by construction, so a
disagreement between them can never be an artefact of two conventions.
**Consequences.** `covered()` reads the rank statistic. This is why coverage
survived the `_invert_cdf` change that broke width comparability (DEC-7).
**Status.** Active. Vindicated.

---

### DEC-4 | Phase 1 | Rotem's posterior-median convention reproduced exactly

**Rationale.** `max{j : sum_{u<=j} w_(u) <= 0.5}` on ordered particle values,
so an MSE difference can never be blamed on a different median convention.
**Status.** Active.

---

### DEC-5 | Phase 1 | Main horizon $n=30$, slice at 20; $n=50$ on one setting only

**Rationale.** Every published table being reproduced is at $n=30$. Compute
allocation, not methodology.
**Consequences.** C5 is demonstrated at one horizon — a stated exposure.
**Status.** Active; revisit if C5 becomes the thesis core.

---

### DEC-6 | Phase 3 | $w^\ast$ frozen from the calibrated-prior, $\sigma_0=3$ cell

**Rationale.** A single threshold makes false certainty comparable across the
misspecification ladder.
**Consequences.** It is *not* scale-free. False certainty is therefore
incomparable across the **scale** sub-block (D16) **and** across the **prior**
sub-block (narrow-prior intervals are narrower for prior reasons: 6.90 vs 7.47
at $q_{0.99}$), which was not recorded at the time. $w^\ast$ also carries its
own MC error: 29.7% of confirmatory control runs fall below the nominal 25%.
**Status.** **Under review.** Candidates: scale-relative $w^\ast/\sigma^{true}$;
one threshold per $(p, n, \sigma_0)$; or replace the width threshold with a
decision-theoretic loss — the codebase already supports the asymmetric
threshold loss and the threshold-free dangerous-error rate, which gave
0.880 ± 0.019 where the width metric gave 0.250 ± 0.025.

---

### DEC-7 | Phase 3 | Reference posterior run at fixed resolution (`fixed_n=513`)

**Rationale.** Speed, with adequacy established by a separate 24-history audit.
**Consequences.** `ref_converged` and `ref_max_delta` are null in all 6,960
screening and 2,700 confirmatory rows, so per-run reference adequacy cannot be
checked; C9 rests entirely on the audit. What *is* recorded is
`ref_boundary_mass` (max $5.8\times10^{-10}$ screening, $4.5\times10^{-12}$
confirmatory).
**Status.** Active; acceptable given the audit, but state it wherever C9 is used.

---

### DEC-8 | Phase 3 | No safeguard implemented

**Rationale.** Establish the failure mechanism before mitigating it. The
exploratory control is a diagnostic, not a safeguard.
**Consequences.** Since it does not restore coverage (C10), a safeguard is a
genuine research contribution for Phase 6 rather than a write-up task.
**Status.** Active.

---

### DEC-9 | Provenance audit | Runs require a clean tree; `--allow-dirty` records a waiver

**Rationale.** All 13 legacy manifests record a dirty tree, which already cost
the project the Phase 3A/3B width comparison.
**Consequences.** `require_clean_tree` raises by default. Configs in
`configs/experiments/` reconstructed from legacy manifests are marked as
unverified pins and are superseded by the first clean re-run.
**Status.** Active.

---

### DEC-10 | Provenance audit | Legacy manifests are grandfathered, not rewritten

**Rationale.** Back-filling a `run_id` or a clean commit into a manifest whose
run was never pinned would fabricate provenance the run never had.
**Consequences.** `tools/manifest_lint.py` lists them under `LEGACY` and
reports their gaps without failing. They are superseded when the experiment is
next re-run under the contract.
**Status.** Active.
