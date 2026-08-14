# Discrepancies from Rotem's thesis implementation

Every point where this reimplementation had to decide something the thesis
leaves open, or deliberately departs from it. Referenced by ID from the
source docstrings and from the validation report.

---

### D1 — Log-normal dispersion: variance or scale?

The thesis writes the prior for the proportionality constant as
$\alpha \sim \mathrm{LogNormal}(\alpha_0 = \log 0.08,\ \tau^2_\alpha = 0.75)$,
which reads as a **variance** of 0.75, i.e. scale $\tau_\alpha = 0.866$. But
section 4 then calls $\tau_\sigma = \tau_\alpha$ a *scale* parameter, which reads
as scale 0.75.

**Decision.** `PriorSpec.tau_is_variance` selects the reading; the default treats
0.75 as the variance ($\tau = 0.866$). The alternative is one config field away.

**Effect.** The prior for $\sigma$ has median 2.4 either way; the 90% prior interval
is roughly $[0.58, 9.9]$ under the default and $[0.70, 8.2]$ under the alternative.
This shifts absolute MSEs modestly but affects all five methods identically, so the
*rankings* — which is what the Phase 1 gate is about — are insensitive to it.

---

### D2 — Fedorov's algorithm in the Dror–Steinberg policy

Rotem calls `AlgDesign::optFederov` (R) for the $m$-run augmentation. This is an
independent exchange implementation over the same candidate grid: initialise from a
spread set plus one random restart, then sweep slots replacing each with the grid
point maximising $\log|I(\theta_{\text{med}},\cdot)|$ until no slot changes.

**Effect.** Exact numerical agreement with her DS column is not expected. Both search
the same objective over the same finite candidate set, and D-optimal designs for a
2-parameter probit are well separated, so the selected points usually coincide; where
they do not, the objective values differ negligibly. The DS arm should be read as
"a faithful DS-type policy", and agreement judged on operating characteristics.

---

### D3 — What the Beta-CDF misspecification's $[m, M]$ means

Rotem defines the false curve by $u = (x - m)/(M - m)$ where "$m \le x \le M$ represent
the limits of the grid of possible stress values", and states $[18, 48]$. But the
design grid is elsewhere tied to the prior calibration ($\mu_0 \pm 14$, i.e. $[16, 44]$
well-calibrated and $[8, 36]$ poorly calibrated). Those cannot both hold.

**Decision.** $[18, 48]$ *defines the true curve* and is held fixed; the design grid
stays $\mu_0 \pm 14$. Rationale: the physical response curve cannot depend on the
experimenter's prior beliefs. If the grid moved with $\mu_0$, the "same" misspecified
truth would be a different curve in the well- and poorly-calibrated arms and Table 3's
two blocks would not be comparable.

**Consequence.** With the poorly-calibrated grid $[8, 36]$, a large part of the design
region lies below 18 where the true response probability is exactly 0. Those stimuli
are legal and informative (a guaranteed non-response); the fitted probit likelihood
handles them without any floor, since it is evaluated with `log_ndtr`, not by taking
logs of the *true* curve.

**Resulting truths used as estimands** (computed, not quoted):
median 29.3559, sd 3.9007, $q_{0.01}=21.6440$, $q_{0.05}=23.4307$,
$q_{0.95}=36.2741$, $q_{0.99}=38.9279$. Rotem quotes median 29.3 and sd 3.9.

---

### D4 — Bruceton is not grid-snapped

Rotem's Bruceton starts at $x_0 \sim N(\mu_0, \sigma)$ and steps by $\pm d$, so its
realised stresses form a lattice offset by $x_0$ and generally do not lie on the
200-point candidate grid the Bayesian methods use. This is kept as written. It means
Bruceton is not restricted to the same design space as the other methods — a real
asymmetry in the original comparison, not introduced here.

---

### D5 — Weights carried in log space

The thesis normalises raw likelihoods, $w_j = L(\theta_j)/\sum_u L(\theta_u)$. In double
precision the product of $n$ probabilities underflows to zero for all particles around
$n \approx 40$, giving $0/0$. All weights here are accumulated as log-likelihoods and
normalised with `logsumexp`.

**Effect.** Purely numerical. Wherever the original does not underflow, the represented
posterior is identical to machine precision.

---

### D6 — KL baseline after rejuvenation

$D_{KL}(W^{(k)} \| W^{(k-1)})$ is only defined when both weight vectors live on the same
particle set. Rejuvenation replaces the particles, so the divergence across that step is
undefined. The thesis does not say what to do.

**Decision.** After a rejuvenation, the baseline is reset to the new set's weights, so
the next step's KL is measured within the new support. The alternative (comparing across
supports) is not well defined; the other alternative (suppressing the check for one step)
differs only in whether a second rejuvenation can fire immediately.

---

### D7 — Rejuvenation proposals with $\beta_1 \le 0$

The proposal $\beta^{new} \sim N(\beta_{med}, \hat\Sigma)$ is Gaussian on $\mathbb{R}^2$ and can
produce $\beta_1 = 1/\sigma \le 0$, i.e. a negative scale. The thesis does not mention this.

**Decision.** Such particles receive prior density 0 and hence weight 0 — the correct
behaviour, since they are outside the parameter space — and are **counted** in
`n_invalid_particles`, which is reported in every manifest. They are not resampled away
silently: a proposal that frequently lands outside the parameter space is evidence about
the rejuvenation step, which is exactly the kind of computational-fidelity finding Phase 2
is meant to surface.

---

### D8 — Quadrature nodes and the density integral

Section 5.3 asks for $R$ nodes "equally spaced between 0 and 1" in cumulative probability.
Two conventions are possible ($r/(R+1)$ or $(r-\tfrac12)/R$); the midpoint rule
$(r-\tfrac12)/R$ is used, with nodes taken as actual particle values under Rotem's
lower-inverse-CDF convention.

Because the nodes are particle values, a concentrated posterior can collapse several nodes
onto the same value, silently deflating the quadrature. The Riemann sum
$\sum_r \pi_r (\theta_{r+1} - \theta_r)$ estimates the total posterior density and should be
close to 1; it is computed on every call and reported as `quadrature_density_integral`,
along with the count of empty windows and distinct nodes. This diagnostic does not exist
in the thesis.

---

### D9 — An algebraic simplification of the section 5.3 formulae

The published formulae compute $\pi^{(1)}_r$, $\pi^{(0)}_r$ and $\bar\Phi$ separately and then
form
$P_{r,x} = \pi^{(1)}_r \bar\Phi / (\pi^{(1)}_r \bar\Phi + \pi^{(0)}_r (1 - \bar\Phi))$.
The normalising constants of $\pi^{(1)}$ and $\pi^{(0)}$ cancel exactly against $\bar\Phi$ and
$1 - \bar\Phi$, leaving

$$P_{r,x} = \frac{\sum_{j \in S_r} w_j\, p_j(x)}{\sum_{j \in S_r} w_j}.$$

Since each $S_r$ is a contiguous run of the $\theta$-sorted particles, the numerator is a
difference of prefix sums. This changes the cost from $O(RNL)$ to $O(NL)$ and is what makes
the scalar-target policies affordable at $N = 10{,}000$, $L = 200$.

**Verification.** `tests/test_numerics.py::test_scalar_mi_matches_literal_formulae` checks the
fast path against a literal transcription of the published equations, to $10^{-12}$, including
identical argmax.

---

### D10 — Horizon

Rotem runs $n = 50$ and slices at 20, 30, 50, reporting tables at $n = 30$. The main Phase 1
run here goes to $n = 30$ with a slice at 20, because every published table this reproduces is
at $n = 30$; the $n = 50$ arm is run separately on one setting (`phase1_horizon.py`) to check
the monotone-in-$n$ claim. This is a compute allocation choice, not a methodological one.

---

### D11 — Estimator and interval conventions

- **Point estimate.** Rotem's posterior median via `max{j : sum_{u<=j} w_(u) <= 0.5}` on the
  ordered particle values — reproduced exactly, so an MSE difference can never be blamed on
  a different median convention.
- **Credible intervals.** Central 95%. Coverage is evaluated as
  $\alpha/2 < P(Q < Q_{\text{true}} \mid \text{data}) < 1 - \alpha/2$ rather than by forming the
  interval and testing containment. The two are equivalent for a continuous posterior, but the
  rank form makes the coverage layer and the SBC layer consistent by construction, so a
  discrepancy between them can never be an artefact of two different conventions.
- **Targets under misspecification.** `mu` means the true curve's **median** and `sigma` its
  **standard deviation** — Rotem's own choice for the Beta case, and the plan's rule that
  physical quantities are the estimands.

---

### D12 — Not implemented in Phases 0–2

- **3pod** (Wu & Tian). Needed only for Rotem's Tables 6–7 non-parametric columns; a
  substantial separate implementation (three phases, Joseph's Robbins–Monro), and not required
  by any Phase 0–2 gate.
- **The two-stimulus model** (her section 6.2). Phase 7 in the plan, explicitly after the
  one-stimulus result is stable.
- **Neyer and Langlie designs.** Reviewed in the thesis but not simulated by it either.

---

### D13 — `entropy_sigma` does not reproduce, and the criterion is verified exact

**The observation.** Of the six designs compared against the published tables, five
reproduce (90.9% of 132 cells within two combined Monte Carlo standard errors). The
sixth, the entropy-of-$\sigma$ design, does not reproduce in *any* setting: 2/28 cells
agree, with ratios from 0.006 to 23.3.

**What this implementation does.** Maximising $I(\sigma; Y_{n+1} \mid x)$ drives the
design to the extreme ends of the candidate grid. Across 250 replicates in
`probit_ind_well` the realised stresses average $[16.00, 44.00]$ — exactly the grid
bounds — with 10.8 distinct levels, an overlapping pattern in **0%** of runs, and zero
rejuvenations. The posterior for $\mu$ therefore barely moves from the prior.

**This is the correct behaviour of the criterion, not a bug.** At the prior, with
$\mu \sim N(\mu_0, s_\mu^2)$ independent of $\sigma$,

$$P(Y=1 \mid \sigma, x) = \Phi\!\left(\frac{x-\mu_0}{\sqrt{\sigma^2+s_\mu^2}}\right),$$

so $I(\sigma;Y\mid x)$ reduces to a one-dimensional integral computable to machine
precision. That reference (`experiments/phase1_quadrature_audit.py`) shows the criterion
is **exactly zero at $x=\mu_0$** (computed $1.1\times10^{-6}$) — because
$\Phi(0)=1/2$ whatever $\sigma$ is — and rises **monotonically to both grid edges**,
where it is tied to a relative $8\times10^{-13}$. The implementation's estimate
correlates 0.996 with the exact criterion and selects a stimulus whose exact criterion
value is **1.000** of the optimum.

**Why the MSEs move the way they do.** Because the design learns almost nothing about
$\mu$, the reported $\mu$ error is set by the prior's calibration rather than by the
data. Under `probit_ind_well` the prior is centred at $\mu_0 = 30$, which *is* the truth,
so MSE($\mu$) = 0.013 — spuriously excellent. Under `probit_ind_poor` the prior is
centred at 22 against a truth of 30, so MSE($\mu$) = 42.0 $\approx 8^2$ — the squared
prior offset. Rotem reports 2.061 and 1.805 for these two cells: moderate values in both,
i.e. her entropy-of-$\sigma$ design *does* learn about $\mu$.

**Status: unresolved.** Either her implementation departs from the section 5.3 formulae
in some way the text does not record, or "entropy of $\sigma$" there means something
other than $I(\sigma;Y_{n+1}\mid x)$. Without her code this cannot be settled here. What
*can* be stated is that the criterion as written in the thesis has been implemented and
independently verified against an exact reference.

**Scope of the impact.** None of the Phase 2 calibration work uses this design
(`entropy_vector` and `uniform_grid` only), and the substantive reading is arguably
strengthened: a single-parameter information criterion with a nuisance parameter can
produce a design pinned to the arbitrary bounds of the candidate grid, which is precisely
the weak-identification failure mode the thesis sets out to characterise.

**Correction and sharpening (appended 2026-08-14).** Three additions. The entry above is
left as written.

*1. An arithmetic gloss overstated.* "MSE($\mu$) = 42.0 $\approx 8^2$ — the squared prior
offset" is loose: $8^2 = 64$, and $\sqrt{42.0} = 6.5$. The posterior does drift partway
toward the truth — under `probit_ind_poor` the posterior mean of $\mu$ reaches 24.45
against a prior centred at 22 and a truth of 30 — so the realised offset is about 5.5 in
the mean and 6.5 RMS, not the full 8. The mechanism is unchanged; only the arithmetic
shorthand was wrong.

*2. What is established about this implementation.* At the settings the published runs
used (`c_fraction=0.1, R=100`) the estimator correlates 0.996 with the closed-form
criterion and selects a stimulus whose exact criterion value is 1.000 of the optimum;
the criterion is zero at $x=\mu_0$ (computed $1.1\times10^{-6}$) and ties at both grid
edges to a relative $8\times10^{-13}$. In `results/raw/phase1_baseline_raw.parquet`
under `probit_ind_well`, `entropy_sigma` places a stimulus at a grid edge in **100%** of
runs, produces an overlapping pattern in **0%**, and leaves the posterior sd of $\mu$ at
4.79 against 1.05 for `entropy_vector`. So: this implementation maximises the criterion
as §5.3 states it, and that criterion provably cannot learn $\mu$.

*3. The published table is inconsistent with the published criterion.* Under
`probit_ind_poor` the prior is centred 8 units from the truth. Rotem reports
MSE($\mu$) = 1.805 for `entropy_sigma` — an RMS error of **1.34**, against 0.76 for
`entropy_mu` and 0.86 for `entropy_vector`. Her entropy-of-$\sigma$ design therefore
recovered $\mu$ about as well as the designs built to estimate $\mu$, starting from a
prior 8 units off. **No design maximising $I(\sigma;Y_{n+1}\mid x)$ can do that**, for the
reason established in point 2: the criterion is exactly zero at $\mu_0$ and maximal at the
grid edges, so it yields no overlapping pattern and leaves $\mu$ essentially at its prior.

*What this does and does not settle.* It locates the inconsistency: it is between the
thesis's stated criterion and the thesis's reported numbers, not in this implementation.
It does **not** identify which of the two is at fault, and nothing here should be read as
doing so. Her code may implement something other than what the text describes — a
posterior-variance-reduction objective, a $\mu$-marginalised one, or a restricted stimulus
range would each yield moderate $\mu$ error — or the text may describe the criterion
correctly while the tabulated column is not the one it is labelled as. Distinguishing
these needs her code.

Status therefore stays **Unresolved** on attribution. What changes is the precision of the
claim: "the discrepancy is in the comparison" can now be stated as *her published table
cannot have been produced by the criterion her own §5.3 specifies*.

---

### D14 — Bias in the section 5.3 quadrature

The same exact reference bounds the estimator's error. At Rotem's stated settings
($c = 0.1\,\mathrm{sd}$, $R = 100$):

| target | exact max $I$ | mean bias | bias / exact max | fraction of estimates negative |
|---|---|---|---|---|
| $\mu$ | 0.342 nats | $-0.0016$ | $-0.5\%$ | 0% |
| $\sigma$ | 0.017 nats | $-0.0047$ | $-27.7\%$ | **36.5%** |

The bias is downward, as the mechanism predicts: rectangular kernel smoothing of
$\pi(\theta \mid y_{n+1}, x)$ pulls $P_{r,x}$ towards $1/2$, inflating the conditional
entropy. It is roughly constant in absolute terms, so it matters in proportion to how
small the true criterion is — negligible for $\mu$, dominant for $\sigma$, where **more
than a third of the estimated values are negative**, which no mutual information can be.

Bandwidth is load-bearing. At $c = 0.4\,\mathrm{sd}$ the $\sigma$ criterion is so
over-smoothed that the selected stimulus retains only $6\times10^{-5}$ of the exact
optimum — a complete design failure. Rotem's $c = 0.1$ sits inside the safe range
(criterion efficiency 1.000 for both targets at $R \ge 100$), but nothing in the thesis
establishes that, and $R = 50$ is already marginal.

**Recommendation for the thesis.** Report the criterion's own numerical error alongside
the design it produces, and clip estimated mutual information at zero — a negative value
is a diagnostic, not a quantity to take an argmax over.

---

## Phase 3 additions

### D15 — Rotem's rejuvenation can be worse than no rejuvenation

Phase 3A's ablation, on identical histories with a correct probit model and a very
steep true curve ($\sigma=0.3$, about 2.4 prior standard deviations below the prior
median scale):

| inference arm | $q_{0.95}$ coverage | paired gap vs reference | mean ESS | rejuvenations |
|---|---|---|---|---|
| $N=10^4$, Rotem's rule ($D_{KL}>0.1$) | 0.647 | **−0.307 ± 0.039** | 845 | 5.4 |
| $N=10^4$, **no** rejuvenation | 0.873 | −0.080 ± 0.024 | **27** | 0 |
| $N=10^4$, $D_{KL}>0.02$ | 0.907 | −0.047 ± 0.020 | 2340 | 16.4 |
| $N=4\times10^4$, Rotem's rule | 0.680 | −0.273 ± 0.037 | 2772 | 5.4 |
| $N=1.6\times10^5$, Rotem's rule | 0.793 | −0.160 ± 0.030 | 7574 | 5.5 |

Three things follow.

1. **Rejuvenating at Rotem's threshold is worse than not rejuvenating at all** here,
   even though it raises the effective sample size thirty-fold. ESS is the wrong
   diagnostic: the proposal $N(\hat\beta_{med}, \hat\Sigma)$ is fitted to an already
   degenerate weighted sample, so it is centred and scaled wrongly, and the importance
   correction cannot repair a proposal with almost no mass where the posterior is.
2. **Firing it more often fixes most of it.** At $D_{KL}>0.02$ each rejuvenation is a
   small, well-supported move and the gap falls to −0.047.
3. **It is not pure Monte Carlo error.** A 16-fold increase in particles reduces the
   rank discrepancy by only 1.4–1.6×, where Monte Carlo scaling predicts 4×. There is
   a bias component, and it comes from the rejuvenation rule.

At $\sigma=3$ (prior-typical) every arm agrees with the reference exactly — paired
coverage difference 0.0000, rank difference $\le0.009$ — so this is a corner effect,
not a general indictment. Recorded because the thesis inherits the rule.

**Recommendation:** trigger on effective sample size rather than on a KL threshold, or
lower the threshold substantially; and report ESS *before* rejuvenation, since the
post-rejuvenation value hides the problem.

---

### D16 — The false-certainty threshold is not scale-free

Protocol §4 fixes one width threshold $w^\ast_{p,n}$ per (target, horizon) from the
correct-probit baseline at $\sigma_0=3$, and applies it unchanged everywhere. That is
right across *DGP shape* and *policy*, which is what it was written for. It is wrong
across the **curve-scale sub-block**: a true $\sigma_0=1.5$ makes every interval
narrow relative to a $\sigma_0=3$ threshold, and $\sigma_0=6$ makes none of them
narrow. The false-certainty column is therefore not comparable across that sub-block,
and the apparent H3 hits at $\sigma_0=1.5$ are an artefact of the mismatch.

The threshold was **not** redefined after the fact — the protocol forbids that — so the
affected numbers are reported with the caveat attached. The repair for Phase 6 is a
scale-relative threshold $w^\ast/\sigma^{\text{true}}$, or one threshold per
$(p, n, \sigma_0)$.

---

### D17 — H3 as pre-registered vs the false-certainty rate

H3 required a cell with coverage $<0.80$ **and mean interval width $\le w^\ast$**. In
the confirmatory cell (`tail_1.0` × adaptive MI, $n=50$, 300 replicates) coverage for
$q_{0.99}$ is 0.433 ± 0.029 but the *mean* width is 7.99 against $w^\ast=6.51$, so the
literal condition fails. The *per-replicate* false-certainty rate — the event the
metric was really meant to capture — is 0.250 ± 0.025 against a correct-model control
of 0.013 ± 0.007.

The mean-width clause was a drafting error: it mixes a cell-level average with a
run-level event, and a cell can contain many narrow-and-wrong runs while its mean
width sits above the threshold. The pre-registered statement is reported as failing,
and the run-level rate is reported alongside it. **The definition was not changed.**
For Phase 6, H3 should be stated on the run-level rate directly.

---

### D18 — Achieved confirmatory precision

Protocol §5 targets a Monte Carlo standard error of 0.008 on the primary metric, with
a fallback CPU budget. Achieved: cell C (control) reached 0.0066 and stopped on the SE
criterion; cells A and B stopped at the 300-replicate cap with false-certainty
standard errors of 0.025 and 0.018. The effects reported are 4–9 standard errors, so
the shortfall does not affect any conclusion, but the pre-registered precision was not
met for two of three confirmatory cells and no claim should be quoted at 0.008.

---

### D19 — Provenance: every legacy manifest is unpinned

All 13 manifests written before the provenance contract record a git commit ending in
`-dirty`, so no result in the project is tied to a committed state of the source. This
is not hypothetical: the bisection settings in `GridPosterior._invert_cdf` (`n_iter`
20→14, `rtol` $10^{-4}$→$10^{-3}$ of a posterior SD, bracket 0.05→0.25 SD) changed in
commit `9f1d16f`, which is the commit *containing* the Phase 3A outputs. Phase 3A and
3B interval widths are therefore not comparable and cannot be made so retrospectively.
Coverage, rank statistics and SBC are unaffected, because `covered()` is read off the
rank statistic rather than off the inverted interval (DEC-3).

**Fixed forward.** `src/provenance.require_clean_tree` refuses to start a run from a
dirty tree; `--allow-dirty` records `provenance.allow_dirty: true` in the manifest.
Legacy manifests are grandfathered by `tools/manifest_lint.py` rather than rewritten
(DEC-10) — back-filling a clean commit would fabricate provenance the run never had.

---

### D20 — Provenance: no run identifier, and `seed_range` was fictional

No legacy manifest carries a `run_id`; the identifying tuple was (manifest file, config
hash, git commit, `seed_base`, `created_utc`). Every legacy manifest *does* carry a
contiguous `seed_range`, which was never used: the code seeds with
`default_rng([seed_base, replicate])`, plus CRN streams
`default_rng([crn_seed, replicate, 0xC0FFEE])` and `[crn_seed, replicate, 0xB0BB1E]`.
The `seed_base` was authoritative and the range decorative.

**Fixed forward.** Manifests carry a `run_id` used as the filename stem, and
`seed_rule` states the scheme verbatim; `tests/test_provenance.py` asserts that string
still matches `src/simulator`. The `seed_range` field is removed and the linter rejects
it.

---

### D21 — 28 of 31 manifest-referenced artifacts are absent from the repository

Manifests reference 31 artifacts. Three are present. Every raw parquet is missing, as
are 19 of 22 summary CSVs — including `phase2_sbc_uniformity.csv`,
`phase3b_screen_summary.csv`, `phase3b_confirm_summary.csv` and
`phase3a_stratified.csv`, which are the named evidence for C3, C5, C6 and C7.

`MEETING_RESULTS.md` states its ground truth is `results/summaries/*.csv` and
`results/raw/*.parquet`. From the repository as published, that ground truth cannot be
consulted, so none of C2–C10 is independently checkable. Legacy manifests compound this
by recording absolute paths from a machine that no longer exists
(`/home/claude/thesis-reliability/...`), which would not resolve even if a file were
recovered.

This is the largest open gap in the project. Options and recommendation are in
`docs/claims.md` §Reproducibility gap. New manifests record repo-relative paths, and
the linter checks that each referenced artifact exists on disk.

> **Superseded by D24.** All 31 artifacts were recovered from the pre-upload working
> copy; the data gap described above is closed. The provenance gaps it points at
> (D19 unpinned, D20 no `run_id`) remain open. Retained as written for the record.

---

### D22 — `n_replicates_requested` mixes per-cell and total counts across manifests

`phase1_baseline` records 8,500 (250 replicates × 34 cells) while `phase3b_confirm`
records 900 (300 × 3 slices) and `phase3_reference_audit` records 24 (the script
argument directly). The field therefore cannot be fed back into a script's replicate
argument without knowing which convention a given manifest used, and for
`phase3a_attribution` (200) and `phase3b_pseudotruth` (88) the per-cell value is not
recoverable from the manifest at all.

**Fixed forward.** `replicates_unit` declares the convention, and the linter flags any
manifest where `n_replicates_completed` exceeds `n_replicates_requested` while still
claiming the unit is replicates — the signature of a row count. The reconstructed
configs in `configs/experiments/` state their `replicates_basis`, and the two
unrecoverable cases carry an explicit `UNRESOLVED` block instead of a guessed number.

---

### D23 — D21's gap is partially closed: the missing summaries were printed to committed logs

D21 established that 19 of 22 manifest-referenced summary CSVs and all raw parquet are
absent. What D21 did not check is whether the *content* of those CSVs survives anywhere
else. It does, partially: `experiments/_runner.py`-era pipelines printed their final
pivot table to stdout, and 15 of the 17 `results/*.log` files (committed since
`4747866`) were never truncated, so the printed table is intact, git-blameable text.

`tools/reconstruct_from_logs.py` re-parses those tables into
`docs/recovered_from_logs/*.csv`. This is **not** a `results/` writer and mints no
`run_id` — see that directory's `README.md` for the full inventory and the ledger this
closes claim-by-claim. In short:

- **C3, C4, C5, C8, C9** — fully recoverable. `phase3b_confirm.log` in particular
  carries the entire confirmatory table and paired H2 test behind C5, the project's
  strongest result.
- **C6, C7, C10** — partially recoverable. `phase3b_screen.log` prints only "Block I,
  n=50, reference posterior"; no Block II table appears in the log despite a Block II
  figure existing in the repo (`results/figures/phase3b_atlas_blockII_ref_n50.png`), so
  whatever these claims draw from Block II remains unverifiable.
- **`phase1_table1_and_3_reproduction.csv`, `phase2_false_certainty.csv`,
  `phase2_coverage_map_stratified.csv`, `phase2_coverage_stratified.csv`, and the full
  per-policy `phase3b_pseudo_truth.csv`** — never printed to any log. Not recoverable
  from this source, or apparently from any source now in the repository.
- **Raw parquet and any statistic not in a printed table** (e.g. per-replicate
  residuals, MC SEs beyond the printed columns) — gone regardless; a printed pivot
  table is a lossy projection of the raw data, not a substitute for it.

Recovered values were spot-checked by hand against the source `.log` text during
construction and matched exactly (see `docs/recovered_from_logs/README.md`).
`docs/claims.md` evidence column is updated to distinguish claims backed by original
(never-existed-in-repo) summary CSVs from claims now backed by this recovery.

> **Superseded by D24.** The originals were found and restored, so the log-based
> reconstruction is no longer the evidence for any claim, and the artifacts it
> references — `docs/recovered_from_logs/` and `tools/reconstruct_from_logs.py` — no
> longer exist in the tree. The reconstruction agreed with the originals everywhere the
> two overlapped, which is the one durable fact from this entry. Retained as written.

---

### D24 — D21's gap fully closed: the originals were never lost, only never pushed

D21 and D23 both assumed the missing artifacts had to be reconstructed, because the
GitHub repository — the only copy either audit had looked at — never had them. A local
working copy at `~/Downloads/thesis-reliability` turned out to be the actual source the
GitHub upload was made from ("Add files via upload", commit `58cb10d`), and it still has
its own `.git` history, separate from and older than this repository's. It was never
itself pushed anywhere; the GitHub copy is a partial upload from it, missing exactly
`results/raw/` and most of `results/summaries/`.

**All 31 manifest-referenced artifacts exist there**, with two exceptions that needed one
extra step: `results/raw/phase3a_histories.parquet` and `results/raw/phase3b_designs.parquet`
showed as uncommitted deletions in that copy's own working tree (`git status` there showed
them `deleted`, unstaged) — recovered with `git restore` against that copy's own history,
no reconstruction involved.

Every recovered summary CSV was diffed against its D23 log-reconstruction as a
consistency check before copying: **identical values everywhere they overlapped**, to
whatever precision the printed log line carried (the CSVs carry full float64 precision;
D23's `q0.5` row was dropped from `phase3b_confirm.log`'s print statement even though the
CSV has it, e.g.) — no case where the two sources disagreed on a number.

**Fixed by copying, not reconstruction.** `results/raw/` now has all 10 parquet files;
`results/summaries/` now has all 30 CSVs, byte-identical to the working copy they came
from (confirmed via `git diff`, no working-tree changes reported for the 4 files this
repository already had committed). The full test suite (121 passed, 1 deselected) was
re-run with the restored data present and shows no change.

D23's `docs/recovered_from_logs/` output is removed as redundant — not because it was
wrong, but because the actual originals now supersede it. `tools/reconstruct_from_logs.py`
is removed with it; the method it used is documented here and in D23 in case a similar
situation (log survives, summary CSV doesn't) recurs and no working copy is available
next time.

**Still open.** Legacy manifests remain unpinned (D19) and carry no `run_id` (D20) — this
recovery restores the *data* the manifests describe, not the *provenance* around them.
`manifest_lint.py --strict` will keep flagging every legacy manifest's artifacts as
"missing on disk," because it resolves the literal absolute path recorded
(`/home/claude/thesis-reliability/...`) rather than falling back to a repo-relative
lookup — a pre-existing gap in the linter itself, unrelated to whether the file exists.

---

### D25 — The linter, not the data, was reporting the missing artifacts; and D23/D24 were deleted from this ledger

Two things happened after D24 was written, both in commit `c87d5a5`.

**The linter bug.** D24's closing paragraph states that `manifest_lint.py --strict` will
keep reporting every legacy artifact as "missing on disk" even though the files are now
present, and attributes this to the linter resolving the recorded absolute path
literally. That was correct as a symptom and incomplete as a diagnosis. The artifact
check tried the path as-is, then fell back to a repo-relative lookup only when
`Path.is_absolute()` was true — and for a POSIX string like
`/home/claude/thesis-reliability/results/raw/phase1_baseline_raw.parquet`,
`Path.is_absolute()` is `False` on Windows, so the fallback never ran. The check was
therefore reporting host-OS path semantics, not the presence or absence of a file, and it
would have reported "missing" whether or not the artifact existed.

Now fixed: an absolute path is detected independently of the host OS and re-rooted at the
first recognised top-level repo directory before the existence check. The path recorded
in each manifest is left untouched — the manifest is a historical record, and rewriting
it to match a new checkout is exactly the fabrication DEC-10 forbids. Verified: zero
"missing on disk" lines across all 13 legacy manifests. D24's closing paragraph should be
read as superseded on this point only; its recovery account is unaffected.

**The deletion.** The same commit removed the D23 and D24 entries from this file
entirely, described in its message as trimming the ledger "back to plain statements of
current state ... without the session-specific recovery narrative." That is a rewrite of
an append-only ledger, which `CLAUDE.md` §5 forbids: a superseded entry is marked
superseded, not deleted. The stated motive — current state should be readable without
the narrative — is a real concern and is already served by `docs/claims.md`, which *is*
regenerated at each gate and is the right home for a plain statement of where things
stand. This ledger's job is the opposite one: to record that D21 was believed for a
while, that a log-based reconstruction was built against it, and that both were then
overtaken by finding the originals. A reader who only sees "all artifacts present" has no
way to tell that the reproducibility gap was ever real, or how close the project came to
shipping a reconstruction in place of its data.

Both entries have been restored to this file byte-identical to their pre-deletion text,
with superseded markers appended to D21 and D23 rather than edits to their bodies. No
`results/` artifact, config or manifest was involved in either the deletion or the
restoration.

---

### D26 — D13's exactness audit was step-0 only; extended past the prior, the estimator holds and the trap is measured

**The gap.** D13 states that the `entropy_sigma` implementation "has been implemented and
independently verified against an exact reference," citing correlation 0.996 and criterion
efficiency 1.000. That verification comes from `experiments/phase1_quadrature_audit.py`,
whose closed form requires $\mu \sim N(\mu_0, s_\mu^2)$ **independent of** $\sigma$. The
first observation destroys that independence. So the audit establishes the estimator at
step 0 and says nothing about the remaining twenty-nine steps of a thirty-step run.

That gap was load-bearing. D14 measures the $\sigma$ criterion at $-27.7\%$ bias with
**36.5% of estimates negative** at Rotem's own settings ($c=0.1$, $R=100$). A criterion
that small relative to its own noise could be producing an essentially arbitrary argmax
once the posterior moves, in which case the edge-pinning D13 attributes to the mathematics
would be partly *this implementation's* estimator failing — and D13's central inference
would not follow.

**How it was closed.** `experiments/phase1_sigma_criterion_drift.py` replaces the closed
form with the reference grid posterior, which stays exact after data arrive. On a product
grid over $(\mu, \eta=\log\sigma)$,
$P(Y=1\mid\sigma_j,x) = \sum_i p(\mu_i\mid\sigma_j)\,\Phi((x-\mu_i)/\sigma_j)$
is a weighted sum over cells, so $I(\sigma;Y\mid x)$ is quadrature-exact with no density
estimation anywhere. At step 0 it reproduces the closed form to $3.9\times10^{-5}$
relative; that agreement is what licenses using it at steps 5, 10, 20 and 30.

**Result: the estimator is exonerated.** Criterion efficiency — the exact criterion value
at the stimulus the section 5.3 estimator selected, over the exact maximum — across 30
replicates, `probit_ind_well`, $\mu=30$, $\sigma=3$:

| step | `entropy_sigma` | `entropy_vector` |
|---|---|---|
| 0 | 1.0000 | 1.0000 |
| 5 | 0.9969 ± 0.0009 | 0.9931 ± 0.0011 |
| 10 | 0.9970 ± 0.0000 | 0.9920 ± 0.0012 |
| 20 | 0.9987 ± 0.0006 | 0.9840 ± 0.0018 |
| 30 | 0.9904 ± 0.0034 | 0.9780 ± 0.0018 |

Worst single audited state of 300: **0.905**. The estimator never collapses, so D13's
attribution stands and needs no retraction — the edge-pinning is the criterion, not a bug
here. What changes is the strength of the claim: "verified exact" can now be said of the
whole trajectory rather than of the prior alone.

**A corollary for D14.** 37–47% of `entropy_sigma`'s estimates are negative at *every*
audited step, yet efficiency stays $\ge 0.99$. The kernel bias is near-constant in $x$, so
it shifts the criterion without moving its argmax. D14's recommendation to clip at zero
remains right as hygiene, but the negativity was never what drove the design.

**What is new: the trap is now measured, not conjectured.** The exact $\sigma$-optimum is
at a grid edge at every step under `entropy_sigma` (edge rate 1.00 throughout, edge band
5% of the candidate range). Under `entropy_vector` it leaves the edge immediately and sits
interior, at $x \approx 27.5 \to 33.7$, i.e. about $\mu + 1.2\sigma$ — the textbook place
to probe a scale. The difference is $\mu$:

| at step 30 | `entropy_sigma` | `entropy_vector` |
|---|---|---|
| posterior sd$(\mu)$ | 4.75 | 0.94 |
| available $\sigma$-information, $\max_x I(\sigma;Y\mid x)$ | 0.0037 | **0.0239** |

There is **6.5× more information about $\sigma$ available** under the design that never
targets $\sigma$. The $\sigma$-optimal probe is interior only once $\mu$ is pinned; with
$\mu$ vague the optimum is the edge, and the edge teaches neither $\mu$ nor $\sigma$.
`entropy_sigma` is therefore in a self-reinforcing trap: greedy one-step
$\sigma$-information keeps $\mu$ vague, and vague $\mu$ keeps the criterion pointing at the
edge.

This is the quantitative mechanism behind an observation that had been recorded but not
explained — that this implementation's `entropy_sigma` is the **worst** design in the study
at estimating $\sigma$, its own target (rank 5th or 6th of 5–6 in all six settings of
`results/summaries/phase1_vs_published.csv`, 0/6 cells reproducing), while every other
policy reproduces on the $\sigma$ target 6/6. A greedy criterion maximising a *share* of a
collapsing total is not the same thing as a design that estimates its target well.

**Scope and tier.** *Screening*: 30 replicates, one cell (`probit_ind_well`, $\mu=30$,
$\sigma=3$), audit steps $\{0,5,10,20,30\}$. It establishes the estimator is sound along
these trajectories; it does not sweep $\sigma_0$ or the misspecified curves. Nothing here
bears on the attribution question in D13, which still needs Rotem's code.

Artifacts: `results/summaries/phase1_sigma_criterion_drift.csv`,
`phase1_sigma_criterion_drift_raw.csv`,
`results/manifests/phase1_sigma_criterion_drift.json`.
