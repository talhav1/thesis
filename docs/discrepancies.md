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
