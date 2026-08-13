# Phase 2 validation report

**Scope.** Phases 0–2 of the implementation plan. Phase 3 (the misspecification
failure atlas) has **not** been started, per the instruction to stop at this gate.

**Bottom line.**

- The likelihood-factorisation result holds as a numerical invariant, is enforced by
  tests, and is confirmed empirically: the exact posterior passes simulation-based
  calibration with Rotem's adaptive design running inside the simulator. **The
  "adaptive dependence invalidates the posterior" question is closed.**
- Five of the six designs reproduce Rotem's published MSEs. The sixth,
  entropy-of-$\sigma$, does not, and the criterion has been verified exact against a
  closed-form reference — so the discrepancy is in the comparison, not in the
  criterion. It is documented and flagged for the advisor.
- **Phase 1 gate: passed with one documented exception.**
- **Phase 2 gate: passed.** The reference posterior passes SBC on every target under
  both designs. The particle posterior is close but measurably worse, which under the
  plan's rule makes computational approximation a thesis chapter rather than a defect
  to fix.

Every number below is reproducible from `results/raw/` plus its manifest in
`results/manifests/`. Commit `f6de1ab`.

---

## 1. Phase 0 — the formal corrections

`manuscript/phase0_note.md` contains the statements; here is their verification status.

| Claim | How it is enforced | Result |
|---|---|---|
| Product likelihood $\propto_\theta$ full-history likelihood, ignorable policy | $\log p_\theta(H_n) - \sum_i \log p_\theta(y_i \mid x_i)$ computed over a 72-point $\theta$ grid | constant to $<10^{-9}$ for a deterministic adaptive, a randomised adaptive, and a randomised non-adaptive policy |
| The two posteriors coincide | total variation between them | $<10^{-12}$ |
| The invariant has content | **non-ignorable** oracle policy | residual spread $>1$ nat; posteriors differ by $>0.01$ in TV |
| Two distinct quantiles $\equiv (\mu,\sigma)$ | round trip and MI equality over five $(p_1,p_2)$ pairs | $<10^{-10}$; a 12-quantile system gives the identical answer |
| One quantile is strictly less informative | MI comparison | scalar MI $\le$ vector MI everywhere, gap $>10^{-3}$ |

The negative control matters: without it, an implementation that simply ignored the
design term would pass the first row. It does not pass the third.

**55 tests, all passing**, covering the ten required checks plus numerical-fidelity
checks on every optimisation in the hot path.

---

## 2. Phase 1 — reproducing Rotem's baseline

**Run.** 250 replicates × 34 (setting × policy) cells, $n=30$ with a slice at $n=20$,
$N=10{,}000$ particles, common random numbers across policies. 8,500 experiments,
17,000 recorded rows, **0 failures**, 3,175 s.

### 2.1 Agreement with the published tables

160 published cells were compared. Each reproduced MSE carries a Monte Carlo standard
error (median 9.0% of the MSE); the thesis reports no uncertainty for its own $S=500$
figures, so `z_combined` assumes the published cell carries a comparable one.

| | cells | within 2 combined SE | median $\lvert$ratio$-1\rvert$ |
|---|---|---|---|
| all six designs | 160 | 76.2% | 0.170 |
| **excluding entropy-of-$\sigma$** | 132 | **90.9%** | 0.138 |
| entropy-of-$\sigma$ only | 28 | 7.1% | 0.899 |

By design:

| design | cells | within 2 SE | median $\lvert$ratio$-1\rvert$ |
|---|---|---|---|
| entropy of $\mu$ | 28 | 100.0% | 0.124 |
| Dror–Steinberg | 28 | 96.4% | 0.143 |
| entropy of $(\mu,\sigma)$ | 28 | 96.4% | 0.156 |
| entropy of $x_{0.95}$ | 28 | 92.9% | 0.147 |
| Bruceton | 20 | 60.0% | 0.161 |
| **entropy of $\sigma$** | 28 | **7.1%** | 0.899 |

Bruceton's 60% is expected and explained: it is not restricted to the candidate grid
and its start point is drawn from a distribution the thesis specifies only loosely
(discrepancy D4).

### 2.2 The one design that does not reproduce

Maximising $I(\sigma; Y_{n+1} \mid x)$ drives the design to the **extreme ends of the
candidate grid**. Averaged over 250 replicates in `probit_ind_well`, the realised
stresses span exactly $[16.00, 44.00]$ — the grid bounds — with 10.8 distinct levels,
an overlapping pattern in **0%** of runs, and zero rejuvenations.

This is the criterion behaving correctly, not an implementation error. At the prior,
$P(Y=1\mid\sigma,x) = \Phi\big((x-\mu_0)/\sqrt{\sigma^2+s_\mu^2}\big)$, so
$I(\sigma;Y\mid x)$ can be computed exactly by one-dimensional quadrature. That
reference shows the criterion is **exactly zero at $x=\mu_0$** — because $\Phi(0)=1/2$
whatever $\sigma$ is — and rises monotonically to both grid edges, which are tied to a
relative $8\times10^{-13}$. The implementation correlates 0.996 with the exact
criterion and selects a stimulus retaining **1.000** of the exact optimum.

The consequence explains the MSE pattern completely: the design learns almost nothing
about $\mu$, so the reported $\mu$ error is set by the prior's calibration.

| setting | prior centre | truth | published MSE($\mu$) | reproduced |
|---|---|---|---|---|
| `probit_ind_well` | 30 | 30 | 2.061 | **0.013** |
| `probit_ind_poor` | 22 | 30 | 1.805 | **42.0** $\approx 8^2$ |

Rotem reports moderate values in both, i.e. her entropy-of-$\sigma$ design *does* learn
about $\mu$. Without her code this cannot be settled. Full write-up: discrepancy D13.

The substantive reading is arguably strengthened by this: a single-parameter
information criterion with a nuisance parameter can produce a design pinned to the
arbitrary bounds of the candidate grid — precisely the weak-identification failure
mode the thesis is about.

### 2.3 Method ordering

Mean rank correlation with the published ordering is 0.688 across 28 (setting, target)
cells, with the same method best in 60.7% of them. **Dropping the entropy-of-$\sigma$
column** — the one design §2.2 shows is not comparable — the mean rank correlation rises
to **0.868** and the same method is best in **75%** of cells: `probit_dep_well` 1.000,
`probit_ind_well` 0.975, and 0.80–0.85 for the remaining four settings.

Qualitative findings that do reproduce:

- Dror–Steinberg and entropy-of-$(\mu,\sigma)$ perform very similarly. Paired over
  common random numbers, the MSE difference is within $\pm 2$ paired SE in 23 of 24
  (setting, target) comparisons.
- Entropy-of-$x_{0.95}$ gives the best $x_{0.95}$ and $x_{0.99}$ estimates in **3 of
  the 4** probit settings, at the cost of worse $\mu$ and $\sigma$. In the fourth
  (`probit_ind_well`) it is second to entropy-of-$\sigma$, whose apparent advantage
  there is the artefact described in §2.2.
- Under the false Beta-CDF curve every target degrades, the tails roughly twice as much
  as the median (§2.6).
- MSE decreases monotonically in $n$ for **all 15** policy × target combinations at
  $n = 20, 30, 50$, confirming the claim the thesis asserts without showing.

Common random numbers cut the standard error of policy comparisons by a median of
12.5% relative to unpaired estimates.

### 2.4 The quadrature audit (new)

The same closed-form reference bounds the error of the section 5.3 estimator. At
Rotem's stated settings ($c = 0.1\,\mathrm{sd}$, $R = 100$):

| target | exact max $I$ | mean bias | bias / max | estimates $<0$ | criterion efficiency |
|---|---|---|---|---|---|
| $\mu$ | 0.342 nats | $-0.0016$ | $-0.5\%$ | 0% | 1.000 |
| $\sigma$ | 0.017 nats | $-0.0047$ | $-27.7\%$ | **36.5%** | 1.000 |

The bias is downward, as the mechanism predicts — rectangular kernel smoothing pulls
$P_{r,x}$ towards $1/2$ and inflates the conditional entropy — and roughly constant in
absolute terms, so it matters in proportion to how small the true criterion is. For
$\sigma$, more than a third of the estimated values are **negative**, which no mutual
information can be.

Bandwidth is load-bearing: at $c = 0.4\,\mathrm{sd}$ the $\sigma$ design retains only
$6\times10^{-5}$ of the exact optimum. Rotem's $c=0.1$ is inside the safe range, but
nothing in the thesis establishes that.

*Recommendation:* clip estimated mutual information at zero and report the criterion's
own numerical error. A negative value is a diagnostic, not something to take an argmax
over.

### 2.5 The utility surface $U_p(r; H_n)$

Computed at $n = 5, 15, 30$ along a seeded adaptive run.

| $n$ | $p=0.5$ | $p=0.9$ | $p=0.95$ | $p=0.99$ |
|---|---|---|---|---|
| 5 | $r=0.70$ | 0.86 | 0.86 | 0.90 |
| 15 | 0.42 | 0.94 | 0.94 | 0.94 |
| 30 | 0.38 | 0.90 | 0.94 | 0.94 |

Two things stand out. First, the separation is real but **compressed**: targeting
$q_{0.99}$ pushes the optimal stimulus only to the $r\approx0.94$ level, not to the
0.99 level — the design wants to stimulate well below the quantile it is trying to
estimate, because that is where the response is still informative. Second, the three
tail targets are nearly indistinguishable from one another by $n=15$, which is the
$U_p(r)$ counterpart of the two-quantile redundancy proposition.

The peak utility falls from 0.093 to 0.021 nats between $n=5$ and $n=30$. Note that
0.021 nats is the same order as the quadrature bias measured in §2.4 — so the *late*
utility surface should be read with that error bar attached. Quadrature density
integral stayed in $[0.9905, 1.0084]$ throughout, so the quadrature itself was healthy.

### 2.6 What misspecification does to the intervals — a Phase 3 pointer

Comparing the correct probit truth with Rotem's Beta-CDF truth, at $n=30$, averaged
over the five comparable designs (per-cell standard error 0.006–0.014):

| target | MSE, probit | MSE, Beta | coverage, probit | coverage, Beta | mean width, probit | mean width, Beta |
|---|---|---|---|---|---|---|
| $q_{0.5}$ | 0.90 | 1.42 | 0.960 | 0.945 | 4.28 | 5.23 |
| $q_{0.95}$ | 3.00 | 5.51 | 0.957 | 0.962 | 9.86 | 12.29 |
| $q_{0.99}$ | 5.43 | 9.65 | 0.957 | 0.962 | 13.55 | 16.90 |

**MSE roughly doubles in the tails, and coverage does not move.** The width column
says why: under the wrong curve the posterior widens by about 25%, which is enough to
absorb the bias. At this horizon and this severity of misspecification, Rotem's Beta
case produces a posterior that is *wrong but appropriately humble* — not a confidently
wrong one.

This matters for planning Phase 3. The thesis is about false certainty, and the
headline misspecification case inherited from Rotem does not exhibit it at $n=30$. The
ladder will need to be pushed somewhere the width does not inflate to compensate:
larger $n$ (where the posterior concentrates faster than the bias shrinks), or shapes
the probit can mimic locally over the region the design actually visits while
diverging in the tail. That is a design requirement for the ladder, and it is better
discovered now than after the atlas has been run.

The one place where under-coverage does show up is `beta_ind_poor`, at 0.920 for $\mu$
and 0.935 for $q_{0.95}$ — the cell that also has the true $q_{0.95}$ outside the
design grid.

### 2.7 Degeneracies — reported, never dropped

Over 17,000 rows: **0** failures, **0** all-zero or all-one paths, **2,293** (13.5%)
with no overlapping pattern (identically, the 2,293 non-finite MLEs), **119** with
particle ESS below 100, **3,000** rows whose $q_{0.95}$ target lies outside the design
grid, and **15,165** rows (89%) in which at least one rejuvenation proposal fell
outside the parameter space ($\beta_1 \le 0$, i.e. negative scale).

That last figure is worth the advisor's attention: the Gaussian proposal in
$\beta$-space routinely proposes negative scales. Those particles correctly receive
zero weight, but it means the effective proposal size is smaller than $N$ and the
rejuvenation step is less efficient than it appears.

The 3,000 out-of-grid rows are `beta_ind_poor`: the true $q_{0.95}=36.27$ against a
design grid of $[8,36]$. Any $q_{0.95}$ reported there is a model extrapolation beyond
every stimulus the experiment could apply — a design-support violation already present
in Rotem's own settings.

---

## 3. Phase 2 — the correct-model calibration baseline

**Run.** 600 SBC draws × 2 policies, $n=30$, both inference backends on identical
histories. Plus a 3×3 fixed-$\theta$ coverage map, 150 replicates per cell.

### 3.1 SBC — the reference posterior

**All 10 (policy × target) cells pass** the Kolmogorov–Smirnov uniformity test at
$\alpha = 0.05$; minimum $p = 0.141$, maximum $D = 0.0467$ against a 95% band of
0.0554. Rank means lie in $[0.481, 0.498]$ with standard error 0.0117. Prior-predictive
coverage of the 95% intervals lies in $[0.945, 0.968]$.

This is the empirical counterpart of Proposition 1. The design in the `entropy_vector`
arm is Rotem's mutual-information policy, run inside the simulator, so the responses
are genuinely adaptively dependent. **If the product likelihood were the wrong
likelihood under an adaptive design, this arm would fail while the `uniform_grid`
control passed. Both pass.**

### 3.2 SBC — Rotem's particle posterior

9 of 10 cells pass; `uniform_grid` / $q_{0.95}$ gives $D = 0.0605$, $p = 0.024$. Across
20 tests one such value is unremarkable, and it does not survive a multiplicity
correction (Bonferroni threshold 0.0025). **The particle posterior is not shown to be
miscalibrated at this sample size.**

But it is measurably *worse* than the reference, consistently:

| | mean $\lvert$rank mean $- 0.5\rvert$ | max KS $D$ | worst coverage |
|---|---|---|---|
| reference | 0.0116 | 0.0467 | 0.945 |
| particle | 0.0162 | 0.0605 | 0.930 ($\sigma$) |

Directly comparing the two posteriors on the same 1,350 histories from the coverage
map:

| target | mean shift, in reference SDs | sd ratio (particle / reference) |
|---|---|---|
| $\mu$ | 0.0105 | 0.9972 [0.985, 1.013] |
| $\sigma$ | 0.0123 | 0.9978 [0.979, 1.018] |
| $q_{0.95}$ | 0.0117 | 0.9977 [0.980, 1.018] |
| $q_{0.99}$ | 0.0119 | 0.9977 [0.980, 1.018] |

So at $N=10{,}000$ the particle posterior's location is off by about **1.2% of a
posterior standard deviation** and its spread is about **0.2% too narrow**. Small, but
systematically in the direction of overconfidence, and it is the same direction as the
rank-mean and coverage gaps above.

**Per the plan's gate, this makes computational approximation a dedicated chapter
rather than a bug.** The reference passes, so the implementation is not stopped.

### 3.3 Coverage stratified by observable path status

The most informative result in Phase 2. Among SBC draws under the adaptive design:

| stratum | draws | $q_{0.95}$ coverage, reference | $q_{0.95}$ coverage, particle |
|---|---|---|---|
| overlapping pattern | 586 | 0.956 ± 0.009 | 0.954 ± 0.009 |
| **no overlap** | 14 | 0.857 ± 0.094 | **0.571 ± 0.132** |

Two things. First, the reference posterior *also* loses coverage when the experiment
never brackets the transition — but only mildly, and it is not required to hold
nominal coverage in a conditional stratum. Second, the particle posterior degrades far
more (0.571 vs 0.857), i.e. **the computational approximation fails hardest exactly
where the statistical problem is hardest**. With 14 draws this is indicative, not
established; it is the single most important thing to power properly in Phase 3.

Under the non-adaptive `uniform_grid` design the no-overlap stratum has 228 draws and
the gap is much smaller (reference 0.943, particle 0.934 / 0.925). Note the adaptive
design produces no-overlap paths only 2.3% of the time versus 38% for the uniform
design — adaptivity is doing its job — but the few it does produce are much worse.

### 3.4 Fixed-$\theta$ coverage map

3 × 3 grid over $\mu_0 \in \{26, 30, 34\}$, $\sigma_0 \in \{1.5, 3, 6\}$, 150
replicates per cell (coverage SE ≈ 0.016–0.023).

Coverage for $q_{0.95}$ and $q_{0.99}$ ranges over $[0.913, 0.980]$. The pattern:

- **$\sigma_0 = 6$** (a curve much shallower than the prior expects, with $\sigma$ near
  the prior's upper range) gives the lowest coverage: 0.913 and 0.920 at $\mu_0 = 30$.
- **$\sigma_0 = 3$ at $\mu_0 = 26$** over-covers: 0.980.
- The largest deviations are $\lvert z\rvert \approx 2.6$, which at 150 replicates is
  suggestive rather than conclusive.

As the plan insists, **this is not a calibration failure.** Fixed-$\theta$ coverage is
not required to equal the nominal level; the posterior is Bayes-calibrated over draws
from the prior, which §3.1 confirms. The map's purpose is to show *where* in parameter
space the conditional departures live, and the answer is: at large $\sigma$, where the
curve is shallow and the tail quantile is far from any stimulus that was applied.

The **false-certainty rate** — interval misses the truth *and* is in the narrowest
quartile — is 0.0%–2.7% across all cells for both $q_{0.95}$ and $q_{0.99}$ (and at
most 4.0% for any target). Under a correct model it is low everywhere, as it should
be. It is the baseline against which
Phase 3's misspecified curves will be measured, and it is the metric to watch there.

---

## 4. Gate assessment

**Phase 1 gate** — *"baseline results match the reported qualitative rankings and agree
quantitatively within Monte Carlo uncertainty or explained implementation
differences."*

**Passed, with one documented exception.** Five of six designs agree in 90.9% of 132
cells; the qualitative findings reproduce; the monotone-in-$n$ claim is confirmed. The
sixth design's criterion has been verified exact against a closed-form reference, so
the disagreement is explained in the sense the gate requires — it is located, not
hand-waved — though not resolved.

**Phase 2 gate** — *"if the reference method fails SBC, stop and fix the
implementation. If only the particle method fails, retain computational approximation
as a dedicated thesis chapter."*

**Passed.** The reference passes SBC on all 10 cells under both designs. The particle
method does not fail either, but is measurably and systematically less well calibrated,
so the chapter is warranted on the evidence rather than on a failure.

---

## 5. Honest limits of this run

1. **Replication.** The plan's target is a coverage standard error of 0.005, needing
   ~10,000 replicates per cell. This run used 250 (Phase 1), 600 (SBC) and 150
   (coverage map), giving standard errors of roughly 0.012–0.023 on coverage and 9% on
   MSE. Every table carries its achieved SE. Conclusions stated above are ones that
   survive at this precision; the $\sigma_0=6$ under-coverage and the no-overlap
   stratum are flagged as suggestive.
2. **The no-overlap stratum has 14 draws** under the adaptive design. This is the
   headline signal and it is the least powered number in the report.
3. **$n = 30$ only** in the main Phase 1 run, with $n = 50$ checked on one setting.
4. **3pod is not implemented** (not required by any Phase 0–2 gate), nor is the
   two-stimulus model (Phase 7).
5. **The reference posterior is converged to $10^{-3}$ of a posterior standard
   deviation**, not to machine precision. 91 of 1,200 SBC draws (7.6%) and 58 of 1,350
   coverage-map draws did not reach that tolerance at the $1025^2$ grid cap; these are
   recorded in the manifests and were *not* excluded. Their achieved tolerance is
   nonetheless tight — median $3.7\times10^{-4}$, 95th percentile $1.3\times10^{-3}$,
   worst $7.9\times10^{-3}$ of a posterior standard deviation — so the reference
   remains far more accurate than the Monte Carlo error of any table here. Raising the
   grid cap would remove them at roughly 4x the cost.
6. **The prior's log-normal dispersion is ambiguous in the thesis** (D1). The default
   reading is used throughout; it shifts absolute MSEs but not rankings.

---

## 6. Recommended next steps

Before Phase 3:

1. **Ask the advisor about entropy-of-$\sigma$** (D13). If Rotem's code is available,
   one run resolves it. If not, the thesis should report the criterion's degeneracy as
   a finding and drop the comparison.
2. **Adopt the clipped mutual information** and report the criterion's numerical error
   alongside the design (D14).
3. **Power the no-overlap stratum.** Phase 3's most valuable measurement is coverage
   conditional on path type, and the adaptive design produces the interesting paths
   only ~2% of the time. Stratified or importance-weighted sampling of that stratum
   will be needed; running more replicates blindly is wasteful.
4. **Redesign the misspecification ladder before running it.** §2.6 shows that
   Rotem's Beta-CDF case doubles tail MSE without moving coverage, because the
   posterior widens by ~25% and absorbs the bias. A ladder built only from curves that
   behave this way will not exhibit the phenomenon the thesis is named after. Push to
   larger $n$, and add shapes the probit can mimic over the visited region while
   diverging in the tail.
5. **Then run the atlas** with the false-certainty rate as the headline metric, using
   §3.4 as the correct-model baseline (0.0–2.7% for $q_{0.95}$ and $q_{0.99}$).
