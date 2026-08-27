# Phase 6 protocol, rung 1 — the robit safeguard — v1.0

**Status: frozen before any confirmatory Phase 6 run.**
A 6-replicate pilot has been run (`configs/experiments/phase6_robit_pilot.json`,
`--allow-dirty`) and its numbers were visible when §6 and §7 were written. That
is stated rather than concealed: the pilot's remit was to choose the r-axis
resolution and to time a cell, and replicate counts derived from measured cost
are the reason it exists. §1, §4 and §5 — the hypotheses, the metric
definitions and the maximum acceptable premium — are set from principle and
from Phases 2–4A, **not** from the pilot's outcome; §5 in particular is derived
from an information-equivalence argument, not calibrated to a number the pilot
produced. Version history is at the end.

Established from Phases 0–4A and **not re-litigated here**: the product
likelihood is exact under an ignorable policy (C1); the reference posterior
passes SBC under adaptive and non-adaptive designs (C3); tail false certainty
under adaptive design is real and confirmatory (C5); the exploratory control
does not restore coverage (C10); and the failure is **undetectable** from the
data the design collects — in 65.5% of adaptive-MI `tail_1.0` runs the
misspecified and correct curves produced bit-identical responses, and the best
of five checks, one of them handed the true curve, rejects at exactly the
nominal level (`manuscript/phase4_undetectability_note.md`).

That last result is what makes this phase necessary and constrains its shape.
**A safeguard cannot be a diagnostic.** There is nothing in the sample for
model criticism to find, so the remedy has to act on the fitted family, the
design, or the reported interval. This rung acts on the first two.

---

## 0. The claim being tested, stated so it can fail

The probit's defect is **not** that $\nu = \infty$ is wrong. It is that
$\nu = \infty$ is assumed with *certainty*. Under
$p_\theta(x) = \Phi((x-\mu)/\sigma)$ the estimand
$q_{0.99} = \mu + \sigma z_{0.99}$ is a fixed affine function of two parameters
that the design pins down near the median, so the posterior for $q_{0.99}$
inherits the precision of data collected nowhere near it. Enlarging the fitted
family to a robit,

$$p_\theta(x) = T_{1/r}\!\big((x-\mu)/\sigma\big), \qquad r = 1/\nu \in [0, 0.5],$$

cuts that transfer, because $z_{0.99}$ stops being a constant and becomes
$t^{-1}_{1/r}(0.99)$, which runs from 2.326 at $r=0$ to 6.965 at $r=0.5$.

Phase 4A predicts that $r$ will be **nearly unidentified** at $n = 50$. That is
the mechanism, not a problem: an unidentified tail parameter should *widen*
$q_{0.99}$ rather than sharpen it. Accordingly, evidence that $r$ is learned
from the data is **not** required for the rung to work, and would in fact be
evidence against the stated mechanism.

The parameterisation and the r-prior panel are fixed by `docs/decisions.md`
DEC-14 and are not revisited here.

---

## 1. Hypotheses

Each is stated so that it can fail, with the effect of interest and the null
that would be reported instead.

**H1 (confirmatory) — the widening repairs the failure cell.**
For the `tail_1.0` DGP under the existing MI$(\mu,\sigma)$ design at $n = 50$,
the robit fit's $q_{0.99}$ coverage exceeds the probit fit's on the *same*
histories.
*Effect of interest:* paired coverage gain $\ge 0.15$ under at least two of the
three r-priors of DEC-14.
*Null:* gain $< 0.05$ under every prior, reported as a **negative result** for
the rung and not weakened into something else.

**H2 (confirmatory) — the premium under a correct model is affordable.**
For the `probit` DGP (the truth *is* the fitted probit at $r=0$), the robit fit
at $n = 50$ satisfies both: $q_{0.99}$ coverage $\ge 0.90$, and the median
paired width ratio $W = \text{width}_{\text{robit}}/\text{width}_{\text{probit}}$
is at or below the maximum acceptable premium $P_{\max}$ of §5, for at least one
prior in the panel.
*Null:* no panel member clears $P_{\max}$, in which case the rung is reported as
**too expensive at this horizon**, and the recommendation moves to the next rung
— **not** to a re-tuned $r_{\max}$, a fourth prior, or a shrunk panel.

**H3 (confirmatory) — a truth nested in the fitted family is repaired.**
`robit_1.0` is $T_3((x-30)/2.76389)$ **exactly** — verified to $2.3\times10^{-14}$
in probability on a 15-point grid — so it sits at $r = 1/3$, interior to
$[0, 0.5]$, and its true $q_{0.99} = 42.550$ is attainable by the fitted family.
The robit fit's $q_{0.99}$ coverage at $n = 50$ is therefore expected within
2 combined MC standard errors of the correct-model control.
*If it is not, the implementation is wrong* and this is treated as a §2-class
failure, not as a finding.

**H4 (confirmatory) — a truth outside the family is only partly repaired.**
`tail_1.0` perturbs the *upper* quantile function only; the t link is symmetric,
so the truth is outside the robit family. The robit fit's $q_{0.99}$ coverage
under MI$(\mu,\sigma)$ at $n=50$ is expected to improve over the probit fit
(H1) and to remain **below** the `probit`-DGP control by $\ge 0.05$.
The shortfall is reported as a **finding** — family enlargement repairs
*anticipated* misspecification only — and is not tuned around. A finding of
*full* repair would be equally publishable and would weaken the interpretation
of H4, not the rung.

**H5 (confirmatory) — the horizon predictions.** Two, tested on the same cells,
`probit` and `tail_1.0` under MI$(\mu,\sigma)$ at $n \in \{50, 100, 200\}$:

- **H5a.** Under the **probit fit** on `tail_1.0`, $q_{0.99}$ coverage is
  *non-increasing* in $n$, with coverage at $n=200$ below coverage at $n=50$ by
  $\ge 0.05$. The mechanism is that the bias is fixed by the KL projection while
  the interval shrinks like $n^{-1/2}$, so more data makes the error more
  confident.
- **H5b.** Under the **robit fit**, the $q_{0.99}$ width shrinks *more slowly*:
  the shrinkage ratio $S = \bar w(n{=}200)/\bar w(n{=}50)$ satisfies
  $S_{\text{robit}} \ge S_{\text{probit}} + 0.10$ under at least two priors.
  *Stated in advance:* the strong form — the robit width stops shrinking
  altogether, $S_{\text{robit}} \approx 1$ — is **not** what is predicted here,
  because $\mu$ and $\sigma$ keep tightening with $n$ even when $r$ does not, and
  $q_{0.99}$ is a function of all three. The pilot is consistent with the weak
  form and not the strong one, and H5b is written in the weak form for that
  reason; this is recorded as a revision of the pre-pilot expectation rather
  than as a confirmation.

Nothing in this project has previously gone above $n = 50$, so H5a and H5b are
both untested.

**H6 (confirmatory) — does the MI criterion go to the tail on its own?**
$r$ is identified only in the tail, so a criterion scoring information about $r$
has a reason to go there that the two-parameter criterion does not have.
Comparing MI$(\mu,\sigma,r)$ against MI$(\mu,\sigma)$ at $n = 50$ on the same
DGP: the fraction of the stimulus budget placed above the perturbation onset
$x_0 = \mu_0 + \sigma_0 z_{0.85} = 33.11$ differs by $\ge 0.05$.
*The direction is not pre-specified.* The pilot suggests MI$(\mu,\sigma,r)$
explores **less**, which if confirmed is a substantive result — a criterion that
scores tail information can be *deterred* from the tail, because a heavy-tailed
particle cloud already predicts near-certain response up there and so expects to
learn nothing from asking.

**H7 (confirmatory) — the mechanism, via the Phase 4A split.**
Runs are stratified by `realized_flips == 0` (blind: the responses are
bit-identical to what the correct probit would have produced) versus $> 0$
(informed). Under `tail_1.0`/MI$(\mu,\sigma)$ the probit fit gives 0.206 blind
against 0.841 informed. The prediction is that the robit fit moves the **blind**
runs and barely touches the informed ones:
gain$_{\text{blind}} \ge$ gain$_{\text{informed}} + 0.10$.
*If it moves both equally the stated mechanism is wrong and the write-up says
so*, in those words. The split variable is a counterfactual requiring the true
curve, so this is a mechanism decomposition and not a diagnostic, and the
conditioning is post hoc — it locates the failure, it does not estimate the
effect of an intervention.

**H8 (exploratory).** Whether the coverage gain of the robit fit is larger for
$q_{0.99}$ than for $q_{0.95}$, and absent at $q_{0.5}$. Exploratory because
$q_{0.5}$ is $r$-free by construction ($t^{-1}_\nu(0.5) = 0$ for every $\nu$),
so the $q_{0.5}$ column is a *check on the implementation* — it must show
essentially no movement — rather than a hypothesis about the safeguard.

---

## 2. Hard invariants — not hypotheses

Per `CLAUDE.md` §2, these are properties of the mathematics. A failure means the
implementation is wrong and is **never** repaired by relaxing a tolerance,
marking a test `xfail`, or narrowing its scope.

| # | statement | test |
|---|---|---|
| N1 | $T_\nu$ at $r = 0$ **is** $\Phi$ — in the probability, in the aggregated log-likelihood, and in the quantile map, to floating point | `tests/test_robit_nesting.py::test_link_at_r_zero_is_exactly_the_probit` and two siblings |
| N2 | With the r-prior degenerate at 0, the three-parameter reference posterior reproduces the two-parameter one on the same history and the same box, to $<10^{-8}$ of a posterior sd in **mean, sd and rank statistic**, for every target | `test_degenerate_r_prior_reproduces_the_two_parameter_posterior` |
| N3 | The same, on an **adaptive** history that revisits a handful of grid points many times — the case that exercises the collapse-to-counts path and the saturated-probability branches | `test_nesting_holds_on_an_adaptive_history` |
| N4 | Weights and log evidence agree to $10^{-14}$ / $10^{-10}$ | `test_degenerate_r_prior_reproduces_the_weights_and_evidence` |
| N5 | **The negative control must keep failing.** With real prior mass on $r > 0$ the robit $q_{0.99}$ sd must *exceed* the probit's by $\ge 5\%$ under every panel prior, while $q_{0.5}$ moves by $< 0.35$ posterior sd. Any change that makes this test pass is a regression | `test_a_non_degenerate_r_prior_does_not_reproduce_the_probit` |
| N6 | The three-dimensional cell-smoothing kernel degenerates **exactly** to `posterior_grid._trapezoid_cdf` when one side length is zero, and to a ramp and a step when two are | `tests/test_robit_grid.py`, first three tests |

N5 is the one that gives N1–N4 content: a three-parameter implementation that
quietly ignored $r$ would satisfy every nesting statement vacuously.

---

## 3. Arms — crossed, never bundled

### 3.1 The 2×2

|  | design = MI$(\mu,\sigma)$ (`mi2`) | design = MI$(\mu,\sigma,r)$ (`mi3`) |
|---|---|---|
| **fit = probit** | the Phase 3 baseline | isolates the **design** effect |
| **fit = robit** | isolates the pure **widening** effect | the combined safeguard |

These are **not** folded together. The $(\text{robit fit}, \texttt{mi2})$ cell
is what attributes any repair to the enlarged family alone; the
$(\text{probit fit}, \texttt{mi3})$ cell is what attributes any repair to the
design alone. A result that only exists in the combined cell must be reported as
such.

Both fits are applied to the **same history** within a replicate, so the family
contrast is exactly paired and the width ratio of §4 is a within-replicate
quantity.

`mi2` is Rotem's `EntropyVectorPolicy` unchanged, over her 10,000-particle
cloud with her rejuvenation rule. `mi3` is the **same policy object** over a
three-parameter cloud: `EntropyVectorPolicy` computes
$h(\mathbb{E}[p]) - \mathbb{E}[h(p)]$ over whatever parameter vector the cloud
carries, so no policy code changes and the separability of the design axis is a
fact rather than a claim.

**No rejuvenation in `mi3`.** Rotem's rejuvenation is a Gaussian refresh in the
GLM parameterisation $\beta = (-\mu/\sigma, 1/\sigma)$, which has no meaning for
$r$ and no three-parameter analogue that would still be *her* method. Extending
it is a research question of its own and belongs to a later rung. Two
consequences, recorded rather than absorbed: the particle values never change,
so the response-probability matrix is built once per replicate (which is what
makes a Student-t design affordable at all); and the effective sample size
decays monotonically. `mi3` is therefore given 40,000 particles against `mi2`'s
10,000. Neither count is a design factor — the design step costs 1–3 s against
9–25 s for a single grid fit, so nothing here trades accuracy for time.

### 3.2 The r-prior panel

Fixed by DEC-14: `reference_uniform` (Beta(1,1) on $r/r_{\max}$, prior mean
$\nu = 4$), `near_probit` (Beta(1,9), $\nu = 20$), `heavy_tail` (Beta(3,1),
$\nu = 2.7$, density vanishing at $r=0$). **Every** reported number appears under
all three. There is no primary prior.

Under `mi3` the prior is a property of the *arm* — an analyst who reports under
a prior designs under it too — so `mi3` is run once per prior and the cell is
coherent. Under `mi2` the design does not involve $r$, so one history serves all
three fits.

### 3.3 DGPs, and what each one is for

| DGP | truth | role |
|---|---|---|
| `probit` ($\mu_0=30$, $\sigma_0=3$) | $r = 0$, i.e. **at the boundary of the fitted family** | the **cost** arm: any extra width is the premium paid for the free parameter when the probit is true. Compared against the Phase 2 baseline, coverage 0.945–0.968 |
| `robit_1.0` | $T_3((x-30)/2.76389)$, i.e. $r = 1/3$ **interior to the fitted family** | should be repaired. If it is not, the implementation is wrong (H3) |
| `tail_1.0` | quantile function perturbed above $p = 0.85$ only; **outside** the t family, which is symmetric | partial repair at best. The shortfall is the finding (H4). This is the Phase 3B/4A failure cell |

True estimands, computed numerically from each DGP:

| DGP | $q_{0.5}$ | $q_{0.95}$ | $q_{0.99}$ |
|---|---|---|---|
| `probit` | 30.000 | 34.935 | 36.979 |
| `robit_1.0` | 30.000 | 36.505 | 42.550 |
| `tail_1.0` | 30.000 | 37.935 | 42.859 |

### 3.4 Factors

| factor | levels |
|---|---|
| fitted family | probit, robit |
| design | `mi2`, `mi3` |
| r-prior | `reference_uniform`, `near_probit`, `heavy_tail` |
| DGP | `probit`, `robit_1.0`, `tail_1.0` |
| horizon $n$ | 50 (full cross); **50, 100, 200** on the key cells only |
| target | $q_{0.5}$, $q_{0.95}$, $q_{0.99}$ |

**Key cells for the horizon axis** are `probit` and `tail_1.0` under `mi2` —
the failure cell and its cost control, both under the *existing* design, because
H5 is a statement about what happens to a reported interval as $n$ grows and not
about the design. The horizon axis is **folded into this experiment**, not run
separately, so a horizon number and a family number come from the same run,
the same seeds and the same manifest.

Prior, stimulus grid ($[16,44]$, 200 points), calibration and credible level are
the Phase 3B Block I settings unchanged, so every Phase 6 number lands beside a
Phase 3 number rather than beside a new one.

---

## 4. Estimands, inference, and metrics

**Estimands are the physical quantiles of the true curve** (DEC-1), computed
numerically from the DGP. Under the robit fit, $\mu$ and $\sigma$ are still not
estimands and $r$ is *never* an estimand — it is a nuisance parameter whose whole
purpose is to be uncertain. `r_mean`, `r_sd` and the mass at each end of
$[0, r_{\max}]$ are reported as **diagnostics of the mechanism**: mass piling at
$r = 0$ says the data prefer the probit, mass at $r_{\max}$ says the $\nu \ge 2$
bound is binding, and both must be said rather than absorbed.

**Inference is the deterministic reference posterior on both arms.** The probit
side runs at the DEC-7 production resolution ($513^2$), unchanged, so the
probit-fit column *is* the Phase 3 estimator. The robit side runs on a
$257^2 \times n_r$ grid over $(\mu, \log\sigma, r)$ with the resolution certified
in §7.

**Coverage is read off the rank statistic** (DEC-3),
$\alpha/2 < P(Q < Q^\star \mid \text{data}) < 1-\alpha/2$ at $\alpha = 0.05$,
identically for both fits — so a gap between the probit and robit columns is a
difference in the posterior and never in the convention. The interval-inversion
solver (`_invert_cdf`) uses the *same* bracket, iteration count and relative
tolerance on both sides, because D19 records what it cost this project when
those settings changed inside a results-producing commit, and widths are a
primary Phase 6 outcome.

### 4.1 Primary metrics — threshold-free and scale-free

The $w^\ast$-based **false-certainty rate is not primary here**, and this is a
deliberate departure from the Phase 3 protocol §4. D16/D17/D18 record that
$w^\ast$ was frozen from a single cell, which makes it non-comparable across
cells whose curves have different scales; and the robit fit changes interval
widths *by construction*, so a threshold calibrated under the probit would not
mean the same thing on the robit side. Every primary metric below is
threshold-free.

| metric | definition | why |
|---|---|---|
| **coverage** | $\Pr[\alpha/2 < \text{rank} < 1-\alpha/2]$ | the headline, and the one thing directly comparable to Phases 2–4A |
| **$z$** | $(q^\star - \mathbb{E}[Q\mid\text{data}])/\mathrm{sd}(Q\mid\text{data})$, per run | the miss in the posterior's *own* units. Negative $z$ is an **over**-estimate of the true quantile; positive $z$ means the truth lies above the posterior mean, the dangerous direction for a safety threshold |
| **$\Pr(\lvert z\rvert > 4)$** | rate over replicates | the confidently-wrong event, with no threshold to freeze. Under a calibrated posterior it is $\approx 10^{-4}$ |
| **miss in half-widths** | $(\mathbb{E}[Q] - q^\star)\,/\,\tfrac12(\text{hi}-\text{lo})$ | the same miss as a reader of the reported interval would experience it; $\lvert\cdot\rvert > 1$ is exactly non-coverage |
| **dangerous-error rate** | $\Pr[\hat q_p - q^\star_p < -\delta]$, $\delta = 0.5\,\sigma^{\text{true}}$ | unchanged from the Phase 3 protocol §4, and comparable across DGPs because $\delta$ is a fraction of the *true* curve's scale |
| **paired width ratio $W$** | per replicate, $\text{width}_{\text{robit}}/\text{width}_{\text{probit}}$ on the identical history, then summarised | the cost of the safeguard, and the quantity §5 bounds. Formed per replicate and *then* averaged — a ratio of two cell means would confound the family contrast with between-replicate variation in what the design happened to learn |
| **shrinkage ratio $S$** | $\bar w(n{=}200)/\bar w(n{=}50)$ within a fit | H5b |

Also reported per cell: bias, RMSE, mean and median width, mean posterior sd,
the blind/informed split of §H7, $r$-posterior diagnostics, reference boundary
mass, minimum ESS, and the design-side summaries carried over from Phase 4A
(`x_max`, `frac_above_onset`, `design_kl_nats`, `realized_flips`).

The Phase 3 false-certainty rate against the frozen $w^\ast$ **is** computed
where a threshold exists, and is reported in a secondary table with the
D16/D17/D18 caveat attached. It is never the basis of a Phase 6 claim.

### 4.2 Nothing is dropped

All-zero, all-one, no-overlap, low-ESS, out-of-grid and exception-raising runs
are recorded with flags and counted in the manifest's `degeneracy_counts`. A
cell whose minimum ESS falls below the pre-registered floor of **100** is
reported as *computationally inadequate*, not discarded and not reinterpreted as
a design finding. C8 is the precedent: turning rejuvenation off beat Rotem's
rule despite a 30× lower ESS, so a low ESS is a number to report.

---

## 5. Maximum acceptable premium — fixed before the confirmatory run

**This is the number that decides whether the rung is adopted, and it is set
from an information-equivalence argument, not from what the pilot produced.**

Under a correct model, a posterior standard deviation shrinks like $n^{-1/2}$.
Widening an interval by a factor $W$ is therefore worth exactly as much
information as discarding a fraction

$$F(W) = 1 - W^{-2}$$

of the experiment. That gives the premium a unit an experimenter can price:

| $W$ | equivalent fraction of the experiment discarded |
|---|---|
| 1.2 | 31% |
| **1.4** | **49%** |
| 1.7 | 65% |
| **2.0** | **75%** |
| 2.5 | 84% |
| 3.0 | 89% |

Two bars, both on the **median paired width ratio $W$ for $q_{0.99}$ under the
`probit` DGP at $n = 50$**, and both required to hold for at least one member of
the DEC-14 panel:

- $P_{\max} = 2.0$ — **the adoption bar.** A safeguard costing more than three
  quarters of the experiment's effective information about the tail is not a
  safeguard; it is a decision to run a smaller experiment, which the
  experimenter could have taken directly and more cheaply.
- $P_{\text{pref}} = 1.4$ — **the unqualified-recommendation bar.** At or below
  half the experiment, the rung is recommended without qualification.

Together with the coverage floor of H2 ($q_{0.99}$ coverage $\ge 0.90$ under the
correct model, against the Phase 2 baseline of 0.945–0.968), these define the
outcome:

| outcome | reading |
|---|---|
| some prior clears $P_{\text{pref}}$ **and** H1 | rung 1 works; recommend it |
| some prior clears $P_{\max}$ but none clears $P_{\text{pref}}$, and H1 | rung 1 works but is expensive; report the price in units of $F(W)$ and let the experimenter choose |
| no prior clears $P_{\max}$ | rung 1 is **too expensive at $n=50$**; report it as such and move to the next rung |
| H1 fails | rung 1 does not repair the failure; **negative result**, reported as one |

**Pre-committed against tuning.** If no panel member clears $P_{\max}$, the
response is *not* a smaller $r_{\max}$, *not* a fourth prior, and *not* dropping
`heavy_tail` from the panel. $r_{\max} = 0.5$ is the finite-variance bound and
the panel is frozen by DEC-14. A premium that is unaffordable under every prior
an honest analyst would write down is the result.

**One asymmetry stated in advance.** $P_{\max}$ bounds the cost under the
*correct* model only. Under a misspecified DGP a large $W$ is not a cost — it is
the safeguard working, and it is reported as such. The premium is a
correct-model quantity by construction.

---

## 6. Replication and stopping

Two stages, and the pilot is neither of them.

**Pilot (run).** 6 replicates per cell, 16 cells, `--allow-dirty`. Coverage
standard error up to 0.204. Its outputs may be used to choose numerical
settings, to time cells, and to state what a confirmatory run would need. **No
number from it may be reported as an effect**, and none is used to select which
cells are run confirmatorily — the cell set of §3 is fixed by design, not by
screening.

**Screening tier.** 40 replicates per cell over the full §3.4 cross. Coverage
standard error $\le 0.079$; the paired width ratio's standard error is far
smaller because it is a within-replicate contrast. Every screened cell is
reported, including negative ones.

**Confirmatory tier.** The cells named in H1–H5, run in batches until either

1. the Monte Carlo standard error of the cell's primary metric is $\le 0.02$
   (coverage; $\approx 240$ replicates), or
2. the cell's CPU budget is exhausted,

whichever comes first. The stopping rule depends only on the replicate count and
elapsed cost — **never on the observed value of any metric** — so it cannot bias
the estimate. The achieved standard error is reported for every cell.

The Phase 3 confirmatory standard was $\le 0.008$, needing $\approx 1{,}900$
replicates. That is not affordable here and the reason is arithmetic, recorded
in advance: a robit grid fit costs about an order of magnitude more than a probit
one, and the confirmatory cells carry three r-priors each. Claims that would need
0.008 to resolve are reported as **suggestive, not confirmatory**, in the same
words Phase 3 used. The effect sizes H1–H7 name (0.05–0.15 in coverage) are
resolvable at 0.02.

**Cost basis.** The measured per-replicate costs from the pilot are recorded in
§7 and in the manifest, and the replicate counts above are derived from them
rather than assumed.

**Common random numbers** across arms within a (DGP, horizon) cell: the latent
uniforms are keyed on `crn_seed` and the replicate index, so the `mi2` and `mi3`
arms of a replicate face the same latent draws and differ only in where they
placed the stimuli. Within an arm, the probit and robit fits see the *identical*
history.

**Seeding** is `default_rng([seed_base, replicate])` per `CLAUDE.md` §3. There is
no contiguous seed range.

---

## 7. Numerical rules

**The r-axis resolution is certified, not asserted.** The r axis is second-order
(`posterior_grid_robit._r_quadrature_logweights`: the $[0, r_{\max}]$ boundary is
a *hard prior* boundary at which the density is finite and positive under two of
the three panel priors, so a rectangle rule would carry a first-order endpoint
error and did — the $q_{0.99}$ mean moved 0.245 stress units between $n_r=9$ and
$n_r=129$, halving with each doubling). The pilot builds the same posterior on
the same adaptive histories across the ladder $n_r \in \{5, 9, 17, 33, 65\}$ and
records two quantities: the distance to the finest rung, and a Richardson
estimate of the error still present at each rung (the gap to the next rung, over
3, which is the right estimate for a second-order scheme). The production
resolution is the coarsest rung whose *estimated* error clears the DEC-7
tolerance of $10^{-3}$ posterior standard deviations on every tracked target
under every panel prior.

**Pilot evidence** (`results/summaries/phase6_robit_pilot_resolution.csv`,
run `phase6_robit_pilot_20260827T154952Z_cbaa79`). Worst case over three
adaptive `tail_1.0` histories × three panel priors × three targets, in units of
the finest rung's posterior standard deviation:

| $n_r$ | distance to $n_r=65$ | Richardson error estimate | build (s) |
|---|---|---|---|
| 5 | 1.13e-1 | 2.6e-2 | 1.4 |
| 9 | 3.71e-2 | 9.2e-3 | 2.8 |
| 17 | 9.59e-3 | 2.5e-3 | 5.7 |
| **33** | **1.96e-3** | **6.5e-4** | **11.3** |
| 65 | 0 (reference) | — | 22.6 |

The deltas fall by a factor of about 4 per doubling at every rung and under
every prior, which is the second-order behaviour
`_r_quadrature_logweights` claims and is itself a check on the quadrature.
**$n_r = 33$ is the production resolution**: its estimated error is
$6.5\times10^{-4}$ sd, inside the $10^{-3}$ tolerance, and $n_r = 17$ is outside
it at $2.5\times10^{-3}$. The worst rung is always `near_probit`, whose density
at the $r = 0$ endpoint is largest, which is where the trapezoid endpoint
correction does the most work.

Realised $(\mu,\eta)$ boundary mass over all 828 fitted rows of the pilot:
$5.9\times10^{-14}$, against the DEC-7 tolerance of $10^{-10}$. The box is not
the limiting approximation anywhere in this phase.

**Boundary mass.** `boundary_mass()` is the $(\mu,\eta)$ edge mass only — the
DEC-7 quantity, unchanged — so the adequacy convention carries over. Mass at
$r = 0$ or $r = r_{\max}$ is **substantive, not diagnostic**: it says the data
prefer the probit, or that the $\nu \ge 2$ bound is binding. It is reported
separately as `r_mass_at_zero` and `r_mass_at_max` and never treated as a
too-small box.

**Localisation runs on the three-parameter posterior itself**, not on the probit
posterior with padding. Under a heavy-tail prior the $(\mu,\sigma)$ posterior
genuinely moves, and a box chosen by a model that is not the one being
integrated is a box whose adequacy is assumed rather than measured.

**No probability floor** is applied in any log-likelihood. A Student-t CDF can
underflow to exactly 0 in the far tail where `log_ndtr` cannot; stimuli with a
zero count are *dropped* rather than weighted by zero, so that $-\infty \times 0$
never forms a NaN, while genuinely infinite entries stay infinite.

**Likelihood weights stay in log space.** Every configuration, seed, manifest,
commit and Monte Carlo standard error is recorded.
`python tools/manifest_lint.py --strict` must pass before the gate.

### 7.1 Measured cost, and the replicate counts that follow

From the pilot (`run_id` `phase6_robit_pilot_20260827T154952Z_cbaa79`, 16 cells
× 6 replicates = 96 design runs, 828 fitted rows, 0 failures, wall 3,164 s on
4 worker processes of which ~430 s is the resolution audit). Per-call means, one
core per worker:

| step | $n=50$ | $n=100$ | $n=200$ |
|---|---|---|---|
| `mi2` design (10,000 particles, rejuvenating) | 1.23 s | 1.41 s | 2.06 s |
| `mi3` design (40,000 particles, no rejuvenation) | 2.72 s | — | — |
| probit reference fit, $513^2$, 3 targets | 2.14 s | 2.19 s | 2.23 s |
| **robit reference fit, $257^2\times33$, 3 targets** | **25.4 s** | **30.6 s** | **36.4 s** |

The robit fit is **12× the probit fit**, not the two orders of magnitude that
equal per-axis resolution would imply, and it is the whole cost of the phase:
the design step is 5% of a cell and is not worth optimising. Interval inversion,
not grid construction, is the dominant term — `_invert_cdf` calls `cdf_at` a few
hundred times per replicate — which is why the three targets are priced together
above rather than per target.

Per replicate of the full §3.4 cross: $\approx 1{,}470$ core-seconds
(3 `mi2` cells at $n=50$ at 80 s, 9 `mi3` cells at 30 s, 4 horizon cells at
96–113 s). Hence:

| tier | replicates/cell | core-hours | wall at 8 workers |
|---|---|---|---|
| pilot (run) | 6 | 2.4 | 53 min (measured, at 4 workers) |
| screening | 40 | 16.3 | $\approx$ 2 h |
| confirmatory (H1–H5 cells only) | 240 | 44 | $\approx$ 5.5 h |

Both long runs are to be launched detached, per `CLAUDE.md` §7.

**A memory constraint, recorded because it changed the run.** The first pilot
attempt at 8 workers lost workers to the OS memory manager: each robit posterior
held a smoothing plan per target simultaneously, eight grid-sized arrays each,
about 600 MB resident per worker. `RobitGridPosterior._smoothing_plan` now keeps
only the most recently requested quantity's plan — callers walk the targets in
sequence, so there was no reuse to lose — which brought steady-state residency
to 137–163 MB per worker. 8 workers is safe at that footprint; the numbers above
assume it. This changed no number: the plan is a pure function of the grid, and
`tests/test_robit_grid.py::test_cdf_at_matches_the_unplanned_kernel` pins
`cdf_at` to the unplanned kernel at $10^{-15}$.

**ESS.** Minimum over the pilot: 16.7. Two of the 96 runs fall below the floor
of 100, both `mi2` at $n = 200$ and both the same replicate index — Rotem's
rejuvenating cloud at 10,000 particles, not the robit design. The `mi3` arm's
minimum is 1,517 at $n = 50$ despite carrying no rejuvenation at all. The floor
therefore binds on the *baseline* arm at the new horizons, and the confirmatory
run must either raise `mi2`'s particle count at $n \ge 100$ or report those cells
as computationally inadequate. It may **not** quietly drop them (C8).

---

## 8. Decision gates

**The rung is adopted** if H1 holds and §5's adoption bar is cleared.
**The rung is reported and not adopted** if H1 holds and no prior clears
$P_{\max}$ — the widening works but costs more than the experiment it protects.
**The rung fails** if H1 fails, and that is a result: it would mean the failure
survives family enlargement, which sharpens the case that the remedy must act on
the design or on the reported interval.

**H3 is not a gate; it is a check.** If a truth *exactly inside* the fitted
family is not repaired, the run is stopped and the implementation is debugged.

**Proceed to a further rung only after** the validation report, and only with
the ledgers updated: `docs/discrepancies.md` and `docs/decisions.md` appended,
`docs/claims.md` regenerated at the gate. Per `CLAUDE.md` §8 the audit happens in
a fresh session, and the results file wins.

**Not in scope for this rung:** a general robust-Bayes method, a decision-
calibration layer, any multivariate extension, the AI-safety application, and
any change to the design that is not exactly "the same MI criterion over a
larger parameter vector".

---

## 9. What the pilot could not settle, and is not claimed to

Recorded here so the confirmatory report cannot quietly inherit a pilot number.

1. **Every effect.** At 6 replicates the coverage standard error is 0.204. No
   coverage difference in this project's range is resolvable, and the pilot's
   coverage columns exist to prove the pipeline runs, not to estimate anything.
2. **The direction of H6.** One-replicate and six-replicate exploration
   fractions differ by more than the effect H6 names.
3. **Whether $q_{0.99}$ premium and $q_{0.99}$ repair trade off across priors.**
   The pilot can show the ordering exists; it cannot establish that the ordering
   is monotone or that any panel member is on the efficient frontier.
4. **Anything at $n = 100$ or $n = 200$.** Two of the three horizons are seen at
   6 replicates by a single design arm.
5. **The premium under `mi3`.** The `mi3` arm is run at $n = 50$ only, by design
   (§3.4), so the horizon behaviour of the combined safeguard is out of scope for
   this rung entirely and must not be inferred from the `mi2` horizon cells.

---

## Appendix A — what the 6-replicate pilot showed

**Read this as orientation, not as evidence.** Every coverage figure below comes
from 6 replicates: the standard error is 0.204 at $p = 0.5$ and 0 at $p = 0$ or
$1$ only because a 6-run proportion cannot resolve anything finer. **No number
here may be cited, and none of it is confirmatory.** It is recorded because the
protocol above was written with it in view and a later reader is entitled to see
what was in view. Source:
`results/summaries/phase6_robit_pilot_summary.csv`,
`..._premium.csv`, `results/raw/phase6_robit_pilot_raw.parquet`, run
`phase6_robit_pilot_20260827T154952Z_cbaa79`, commit `81d399e` **dirty, waiver
recorded**.

**A1. The mechanism is confirmed, and it is stark: $r$ is not learned at all.**
Posterior against prior for the tail parameter, `tail_1.0` under
MI$(\mu,\sigma)$:

| prior | prior mean $r$ | prior sd | posterior mean, $n{=}50$ | posterior mean, $n{=}200$ | posterior sd, $n{=}200$ |
|---|---|---|---|---|---|
| `heavy_tail` | 0.3750 | 0.0968 | 0.3763 | 0.3768 | 0.0955 |
| `reference_uniform` | 0.2500 | 0.1443 | 0.2534 | 0.2575 | 0.1417 |
| `near_probit` | 0.0500 | 0.0452 | 0.0498 | 0.0512 | 0.0467 |

The posterior for $r$ is the prior for $r$, to three decimal places, at $n = 200$,
under a design that placed 40% of its budget above the perturbation onset. Phase
4A predicted near-unidentifiability at $n = 50$; the pilot suggests it does not
improve with four times the data. This is the mechanism working as stated in §0,
and it is also what makes §5's premium a *structural* cost rather than one that
more data would retire — which is the single most consequential thing the
confirmatory run has to settle.

**A2. H5a's direction is visible and large.** `tail_1.0`, MI$(\mu,\sigma)$,
$q_{0.99}$, probit fit:

| $n$ | coverage | mean width | mean $z$ | $\Pr(\lvert z\rvert>4)$ | dangerous-error rate |
|---|---|---|---|---|---|
| 50 | 0.833 | 9.45 | 1.41 | 0.000 | 0.667 |
| 100 | 0.667 | 5.84 | 2.49 | 0.333 | 0.833 |
| 200 | **0.000** | 3.66 | **4.58** | **0.667** | **1.000** |

More data makes the probit fit more confidently wrong, exactly as the KL-
projection argument says it must. Under the robit fit with `reference_uniform`
the same cells give coverage 1.000 at all three horizons with widths 20.2, 14.7,
11.1 — so the shrinkage ratios are $S_{\text{probit}} = 0.387$ against
$S_{\text{robit}} = 0.550$ (`reference_uniform`), 0.496 (`heavy_tail`), 0.419
(`near_probit`). The weak form of H5b is consistent with two of three priors;
the strong form is not, and §1 records that revision.

**A3. The central tension the confirmatory run must resolve.** Median paired
width ratio $W$ for $q_{0.99}$ under the **correct probit** at $n=50$, beside the
coverage gain on `tail_1.0` at the same horizon:

| prior | $W$ (correct model) | equivalent experiment discarded | clears $P_{\max}=2.0$? | coverage gain on `tail_1.0` |
|---|---|---|---|---|
| `near_probit` | 1.11 | 19% | yes, and clears $P_{\text{pref}}$ | **0.000** |
| `reference_uniform` | 2.15 | 78% | **no** | +0.167 |
| `heavy_tail` | 2.42 | 83% | **no** | +0.167 |

The prior cheap enough to adopt is the one that does not repair, and the priors
that repair are the ones §5 would reject. If this ordering survives at 240
replicates, the honest conclusion is that rung 1 does not clear its own bar, and
§5 has already pre-committed against rescuing it by moving the bar, shrinking the
panel, or lowering $r_{\max}$.

**A4. A prior that excludes the correct model is confidently wrong in the other
direction.** Under the correct probit at $q_{0.99}$, `heavy_tail` gives coverage
0.333 at $n=50$ and 0.000 at $n=200$ *despite* intervals 2.4–3.1× wider than the
probit's, with mean $z = -2.17$ at $n=200$. `heavy_tail`'s density **vanishes**
at $r = 0$ (DEC-14), so the correct model is excluded from the fitted family and
the posterior is pushed upward faster than the widening can cover. This is not a
defect of the implementation; it is the same false-certainty mechanism with its
sign reversed, and it is why DEC-14 requires all three priors rather than the one
that most widens the tail.

**A5. H6 points the opposite way to the naive expectation.** At $n=50$ on
`tail_1.0`, the fraction of budget above the onset is 0.400 under
MI$(\mu,\sigma)$ against 0.143 / 0.257 / 0.307 under MI$(\mu,\sigma,r)$
(`heavy_tail` / `reference_uniform` / `near_probit`), and the design KL falls
from 0.600 nats to 0.057 / 0.141 / 0.337. The blind-run rate *rises* from 0.667
to 1.000 / 0.833 / 0.667. A criterion that scores information about $r$ appears
to be **deterred** from the tail, plausibly because a heavy-tailed particle
cloud already predicts a near-certain response up there and so expects to learn
nothing by asking. If confirmed this is a result about mutual information as a
design criterion, not about the robit; H6's direction is deliberately left
unspecified in §1 for that reason.

**A6. The implementation checks pass.** $q_{0.5}$ is $r$-free by construction and
must not move: probit width 3.678 against `near_probit` robit 3.764, with mean
$z$ of $-0.369$ against $-0.364$ — the widening is a tail effect, not a global
loss of precision. Maximum $(\mu,\eta)$ boundary mass over all 828 rows is
$5.9\times10^{-14}$. Zero failures, zero all-zero and zero all-one paths.

**A7. What the pilot leaves entirely open.** Everything in §9, plus: whether A1
survives at a horizon where $\sigma$ is pinned down much harder; whether A3's
ordering is monotone in the prior's mass near $r=0$, or whether some fourth
prior — which §5 forbids introducing *after* seeing this — would sit on a better
frontier. That last question is a real one and belongs to a **later rung**, posed
in advance rather than discovered in the results.

---

## Version history

- **v1.0** — frozen before any confirmatory Phase 6 run, after the 6-replicate
  pilot. §7's resolution choice and §6's cost basis are derived from the pilot,
  which is the pilot's stated remit. §1's H5b is written in the weak form and
  the pre-pilot strong form is recorded beside it, so a later reader can see
  that the expectation was revised before the confirmatory run rather than
  after it. §5 was written from the information-equivalence argument and is
  independent of every pilot number.
