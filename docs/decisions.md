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

---

### DEC-11 | Phase 4A | Detection tests are calibrated on a null cell, not on an asymptotic reference

**Rationale.** The oracle likelihood ratio has its shape parameter bounded
below at zero, so its asymptotic null is a 50:50 chi-squared mixture, and
n = 50 under a data-dependent sequential design is not an asymptotic regime in
any case. The correct-probit cell run under the *same policy* is an exact null
sample from the same design distribution.
**Consequences.** Critical values are the 1-alpha empirical quantile of the null
cell, per policy, and the realised level is reported alongside every one of them
(`results/summaries/phase4_undetectability_calibration.csv`). Detection rates
are readable as power rather than as miscalibration, at the cost of Monte Carlo
error in the critical value itself, which a confirmatory tier must re-estimate.
**Status.** Active.

---

### DEC-12 | Phase 4A | Predictive p-values use the mid-p convention

**Rationale.** The predictive discrepancies are functions of integer counts, and
in the upper stimuli of an adaptive design -- where the fitted probability is
near one and every replicate reproduces the observed successes exactly -- ties
are the typical case, not the exception. Counting them whole drove the
tail-directed p-value to exactly zero and rejected the *correct* model on 58-75%
of null replicates.
**Consequences.** Ties count half. The null level returns to nominal, which is
what a detection rate computed from the p-value requires.
`tests/test_model_check.py::test_ppc_p_values_are_not_degenerate_under_ties`
holds it there. Any later diagnostic that forms a predictive p-value on
saturated bins inherits the same hazard.
**Status.** Active.

---

### DEC-13 | Phase 4A | The oracle alternative is frozen at one shift for every cell

**Rationale.** Scoring each DGP against an alternative frozen at *its own* truth
would need a separate null per shift, and would leave the correct-probit cell --
which has no true shift -- without one at all.
**Consequences.** `log_bf_oracle` is the tail-perturbed family at shift95 = 1.0
throughout, so one column means the same thing in every cell and the null cell
calibrates it directly. For `tail_0.5` and `robit_1.0` it is therefore an
approximately-directed rather than an exact oracle; the exactly-directed test is
the free-shift likelihood ratio, which is reported for every cell.
**Status.** Active.

---

### DEC-14 | Phase 6 | The tail parameter is $r = 1/\nu$ on $[0, 0.5]$, and every result is reported under a fixed three-prior panel

**Rationale.** Two choices, and neither is a tuning knob.

*Parameterisation.* The safeguard replaces the fitted probit with a robit,
$p(x) = T_\nu((x-\mu)/\sigma)$. Written in $\nu$ or $\log\nu$, the correct-model
control sits at $\nu = \infty$ — a limit, reachable only asymptotically, so the
probit could never be *represented* in the fitted family, only approached. In
$r = 1/\nu$ it sits at $r = 0$, an endpoint of a bounded interval that a grid
can carry as a node. The correct model is then nested **exactly**, mirroring
`curve_families`, where every alternative reduces to the probit at
$\lambda = 0$. That is what makes the nesting statement in
`tests/test_robit_nesting.py` an identity in floating point rather than a
tolerance: with the $r$-prior degenerate at $0$ the three-parameter reference
posterior *is* the two-parameter one, to $<10^{-8}$ of a posterior standard
deviation in mean, sd and rank statistic for every target, on a shared box.
$r_{\max} = 0.5$ is $\nu = 2$, the heaviest tail with a finite variance; below
it the implied tolerance distribution has no scale and `sigma` would stop
denoting anything (DEC-1).

*Prior panel.* Phase 4A established that at $n = 50$ under the adaptive MI
design the data carry almost no information about the tail: in 65.5% of
`tail_1.0` runs the responses are bit-identical to what the correct probit
would have produced (`manuscript/phase4_undetectability_note.md` §3). A nearly
unidentified parameter is one whose posterior is close to its prior, so the
$r$-prior is not a nuisance detail — it is doing most of the work, and a
single choice of it would make the safeguard's effect a property of that
choice. That is precisely the defect this rung exists to cut: the probit's
failure is that $\nu = \infty$ is assumed with *certainty*, and replacing one
certainty with another, narrower one would not be progress.

**Consequences.** Three priors, fixed before any Phase 6 result was seen, and
**every** reported number appears under all three — never a "primary" prior with
the others as sensitivity:

| name | prior on $u = r/r_{\max}$ | prior mean $r$ | implied $\nu$ | reading |
|---|---|---|---|---|
| `reference_uniform` | Beta(1, 1) | 0.250 | 4.0 | spread out; asserts nothing beyond $\nu \ge 2$ |
| `near_probit` | Beta(1, 9) | 0.050 | 20.0 | sceptical: believes the probit, declines to assume it with certainty. 90% of mass below $r = 0.115$. Density at $r=0$ finite and positive |
| `heavy_tail` | Beta(3, 1) | 0.375 | 2.7 | favours heavy tails. Density **vanishes** at $r = 0$, so it does not merely doubt the probit, it excludes it |

Implemented as `src/robit.R_PRIOR_PANEL`; `R_DEGENERATE` is not a member and is
not a modelling option — it exists solely to drive the three-parameter
machinery back onto the two-parameter one for the nesting invariant.

Under the MI$(\mu,\sigma,r)$ design arm the prior is a property of the **arm**,
not of the report: an analyst who reports under a prior designs under it too,
so that arm is run once per prior and the cell is coherent. Under
MI$(\mu,\sigma)$ the design does not involve $r$, so one history serves all
three fits and the family contrast is exactly paired within a replicate.

A conclusion that holds under `reference_uniform` and fails under `near_probit`
is to be reported as a statement about the prior. The write-up may not select
the panel member that flatters the safeguard.

**Status.** Active. Frozen before the Phase 6 pilot ran.
