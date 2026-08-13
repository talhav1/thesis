# Phase 3 validation report

**Scope.** Phase 3 only, against the pre-registered `phase3_protocol.md` (v1.1,
frozen at commit `2c452d7` before any Phase 3 experiment ran). No safeguards, no
multivariate extension, no AI-safety application, no later theoretical phases.

**Bottom line.** A false-certainty regime exists and is reproducible, powered and
attributable — but it is *not* where the literature's intuition points. Wrong
link functions that are wrong in the middle of the curve are absorbed almost
completely. What breaks tail inference is misspecification **located in the tail
being estimated**, and the adaptive design makes it materially worse than a
non-adaptive one on the same data.

| hypothesis | verdict | headline number |
|---|---|---|
| **H1** particle vs reference on no-overlap paths | **confirmed** | paired −0.561 ± 0.014 ($q_{0.95}$, 1,320 paths) |
| **H2** adaptive raises false certainty vs non-adaptive | **confirmed** | paired +0.147 ± 0.030 ($q_{0.99}$, 300 CRN pairs) |
| **H3** as literally pre-registered | **not met** | coverage 0.433 ✓ but mean width 7.99 > $w^\ast$ 6.51 ✗ |
| H3, run-level false-certainty rate | confirmed | 0.250 ± 0.025 vs control 0.013 ± 0.007 |
| **H4** policy-dependent pseudo-truth | **confirmed** | spread 2.47 ± 0.08 stress units |
| **H5** design-support ≠ misspecification | supported (exploratory) | dangerous-error rate ~1.0 for *every* design |

---

## 1. Established findings

Reproducible, adequately powered at the achieved precision, practically
meaningful against the pre-registered thresholds, and attributable to a named
mechanism rather than to implementation error.

### 1.1 Tail-located misspecification produces confidently wrong tail inference

Confirmatory cell, `tail_1.0` × adaptive MI $(\mu,\sigma)$, $n=50$, 300
replicates, reference posterior:

| | $q_{0.95}$ | $q_{0.99}$ |
|---|---|---|
| coverage | 0.633 ± 0.028 | **0.433 ± 0.029** |
| false-certainty rate | 0.253 ± 0.025 | **0.250 ± 0.025** |
| dangerous-error rate | 0.637 ± 0.028 | **0.880 ± 0.019** |
| bias | −2.17 | −4.82 |
| mean width | 6.05 | 7.99 |

Correct-probit control, same policy and horizon, 300 replicates: coverage
0.967 ± 0.010, false certainty 0.013 ± 0.007. The false-certainty rate rises
**nineteen-fold** (18.8x); the difference is 0.237 ± 0.026, i.e. 9.1 standard errors.

The `tail_perturbed` curve is *exactly* probit for $x \lesssim 33.1$ — the region
the adaptive design visits — and differs only above it. So the likelihood sees
essentially nothing wrong, the posterior does not widen in compensation, and the
extrapolated tail quantile is wrong with a straight face. This is the mechanism
the thesis is named after, isolated by construction.

**Attribution.** Not numerical: across all 228 screened cells the
particle-minus-reference coverage difference has mean +0.002 and maximum
absolute value 0.067, so both backends agree. Not a rescaling: the curve is
matched to the probit on median and slope at the median at every severity. Not a
support failure: this is Block I, target inside the grid.

### 1.2 Adaptivity makes it worse — H2

Paired on common random numbers, 300 pairs, `tail_1.0`, $n=50$:

| target | adaptive MI | non-adaptive fixed | paired difference |
|---|---|---|---|
| $q_{0.95}$ | 0.253 | 0.150 | **+0.103 ± 0.032** |
| $q_{0.99}$ | 0.250 | 0.103 | **+0.147 ± 0.030** |

Both exceed the pre-registered 0.05 effect of interest, at 3.2 and 4.9 standard
errors. Coverage moves the same way (0.433 adaptive vs 0.633 fixed at
$q_{0.99}$), and the mechanism is visible in the widths: the adaptive design
produces intervals 37% narrower (7.99 vs 12.65) on data that are equally wrong.
The adaptive design concentrates the experiment where the model already fits,
which is precisely where it learns nothing about whether the model is wrong.

### 1.3 The same curve has different pseudo-truths under different policies — H4

Probit KL projections under each policy's empirical design distribution,
replicate-level bootstrap, Block I:

| DGP | target | physical | fixed | adaptive MI | adaptive $q_{0.95}$ | exploratory | spread |
|---|---|---|---|---|---|---|---|
| tail 1.0 | $q_{0.99}$ | 42.86 | 37.13 | 37.72 | 39.48 | 39.59 | **2.47 ± 0.08** |
| robit 1.0 | $q_{0.99}$ | 42.55 | 37.65 | 38.11 | 39.08 | 39.96 | **2.31 ± 0.05** |
| tail 1.0 | $q_{0.95}$ | 37.94 | 35.05 | 35.50 | 36.68 | 36.93 | **1.88 ± 0.06** |
| robit 1.0 | $q_{0.95}$ | 36.51 | 35.41 | 35.74 | 36.35 | 37.05 | **1.65 ± 0.04** |
| probit (control) | $q_{0.99}$ | 36.979 | 36.979 | 36.979 | 36.979 | 36.979 | **0.000** |

Four Block I cells exceed the pre-registered 1.5-unit threshold. The control landing on the physical truth to three decimal places under **all four**
policies -- spread 0.000 -- validates the projection machinery.

Two consequences worth stating plainly. First, **"the fitted parameters" is not
well defined under misspecification** — it is a property of the experiment as
much as of the system. Second, **every policy's pseudo-truth is biased low** for
these curves, by up to 5.73 units; the policies differ in how badly they miss,
not in whether they miss. No amount of data fixes this, because it is where the
posterior is *converging*.

### 1.4 Rotem's rejuvenation degrades the particle posterior on steep-curve paths — H1

On 1,320 adaptively generated no-overlap histories with a correct probit model
and a very steep true curve, paired against the reference on identical data:

| arm (matched on $\mu,\sigma$) | $q_{0.95}$ particle | reference | paired difference |
|---|---|---|---|
| no overlap | 0.418 | 0.980 | **−0.561 ± 0.014** |
| overlap control | 0.611 | 0.874 | −0.263 ± 0.014 |

No-overlap adds **−0.299 ± 0.020** beyond the stratum effect for $q_{0.95}$ and
−0.345 ± 0.020 for $q_{0.99}$. The particle posterior is 16% too narrow. This
upgrades the Phase 2 hypothesis (14 cases) to a finding at ~2,275 cases.

**Attribution is the interesting part** (see D15). An ablation on identical
histories shows the gap is *caused by the rejuvenation rule*, not merely by
having too few particles: switching rejuvenation off reduces the gap from −0.307
to −0.080 despite dropping effective sample size from 845 to 27, and lowering the
KL threshold from 0.1 to 0.02 reduces it to −0.047. Sixteen times more particles
buys only a 1.5× improvement where Monte Carlo scaling predicts 4×.

**Scope, stated honestly.** This is a corner. At prior-typical $\sigma=3$ every
arm agrees with the reference to 0.0000 paired coverage difference. The enriched
strata carry a 68% event rate by construction; unconditionally, the pilot puts
no-overlap paths at ~2% under moderate curves and near zero for $\sigma \ge 1.2$
with the truth inside the grid. Established conclusion 3 from Phase 2 stands: at
$N=10{,}000$ under prior-typical parameters the particle posterior is very close
to the reference.

---

## 2. Negative findings

These are results, not gaps, and they constrain the thesis at least as much as
the positive ones.

### 2.1 Classical link misspecification is essentially harmless here

At full severity, with median and slope matched, $q_{0.99}$ coverage under the
adaptive MI design: logistic 0.933, cloglog **1.000**, mixture 0.967, against a
correct-probit control of 0.967. False certainty 0.067, 0.000 and 0.033 against a
control of 0.033. Rotem's Beta-CDF case: coverage 0.967, false certainty 0.033.

**Getting the link wrong in the middle of the curve does not damage tail
inference in this model.** This directly confirms and generalises the Phase 2
observation about the Beta-CDF case, and it is a substantive negative result: a
thesis chapter that simply ran a ladder of alternative links would have found
nothing.

### 2.2 Under the cloglog curve the posterior errs *conservatively*

The cloglog curve has a shorter upper tail than the matched probit (true
$q_{0.99}=34.94$ vs 36.98), and every policy's pseudo-truth **overshoots** it
(36.05–37.25). An overstated safe threshold is the benign direction of error.
Not every misspecification is dangerous, and the direction is predictable from
the geometry.

### 2.3 None of Phase 3B is numerical

Across 228 screened cells: particle-minus-reference coverage difference, mean
+0.0023, 5th–95th percentile [0.000, +0.033], maximum absolute 0.067. The
model-error and numerical-error channels are cleanly separated, and Phase 3B
lives entirely in the first.

### 2.4 Prior misspecification is not the driver

`tail_1.0` × adaptive MI, $q_{0.95}$ coverage: calibrated prior 0.455–0.667,
narrow prior 0.591, shifted prior 0.455. The failure is present with a
well-calibrated prior and is not much worsened by a bad one.

---

## 3. Suggestive findings

Directionally clear at screening precision (30 replicates, coverage SE
0.03–0.09), not confirmed.

- **Heavy tails (robit, $\nu=3$)** behave like a milder version of the tail
  perturbation: $q_{0.99}$ coverage 0.500 (adaptive) vs 0.833 (fixed), false
  certainty 0.133 vs 0.100. Same signature, roughly half the size.
- **Shallow curves amplify it.** `tail_1.0` at $\sigma_0=6$ gives $q_{0.99}$
  coverage 0.227 — the worst anywhere in Phase 3B — against 0.955 for the correct
  probit at the same scale. Not confirmed, and see §5 for why its false-certainty
  number is not comparable.
- **The broad exploratory design is the most honest.** It has the lowest false
  certainty in almost every Block I cell (0.067 at `tail_1.0` $q_{0.99}$ against
  0.267 adaptive), bought with ~30% wider intervals. This is the natural
  starting point for Phase 6 and it did not have to be true.

---

## 4. Design-support failure (Block II) — a separate mechanism

Kept separate throughout, and it behaves differently.

| DGP | policy | coverage $q_{0.99}$ | dangerous error | unresolved | bias |
|---|---|---|---|---|---|
| tail 1.0 | adaptive MI | 0.318 | **1.000** | 0.000 | −5.68 |
| tail 1.0 | fixed | 0.955 | 1.000 | 0.955 | −9.62 |
| probit, truth at 38 | adaptive MI | 1.000 | 0.591 | 0.091 | −1.44 |
| probit, truth at 38 | fixed | **0.000** | 1.000 | 1.000 | −11.52 |

Every target lies outside the sampled stimulus range in 100% of runs.

Two things that do **not** happen in Block I:

1. **Adaptivity is protective for the point estimate.** With the truth at
   $\mu=38$ and the grid ending at 36, the adaptive design walks to the edge and
   recovers a usable estimate; the non-adaptive design, pinned near $\mu_0=22$,
   never observes a response at all (unresolved rate 1.000) and has coverage
   exactly **zero**.
2. **The dangerous-error rate is at or near 1.0 for every design.** Extrapolating
   a fitted probit to a quantile outside the stimulus range actually applied is
   unsafe regardless of how the design was chosen.

This is why the two blocks must not be pooled: in Block I adaptivity is the
aggravating factor; in Block II it is the mitigating one.

---

## 5. Numerical failures and protocol deviations

All recorded in `docs/discrepancies.md`, D15–D18.

- **Reference adequacy audited, passed.** The production fixed-resolution
  reference ($513^2$ after localisation) was checked against a $2049^2$ build on
  24 of the *hardest* histories in the study (Phase 3A no-overlap paths, where
  the posterior is tightest relative to the box). Maximum mean difference
  $4.1\times10^{-4}$ posterior standard deviations, maximum rank difference
  $4.4\times10^{-3}$, and **zero coverage disagreements in 72 comparisons**. The
  Phase 3A result is not a reference artefact.
- **D16 — the false-certainty threshold is not scale-free.** $w^\ast$ was frozen
  at $\sigma_0=3$ and is not comparable across the curve-scale sub-block. The
  apparent H3 hits at $\sigma_0=1.5$ are an artefact of that mismatch. The
  threshold was **not** redefined after the fact; the affected numbers carry the
  caveat instead.
- **D17 — H3's mean-width clause was a drafting error.** It mixes a cell-level
  average with a run-level event. Reported as failing, with the run-level rate
  alongside. Not changed.
- **D18 — confirmatory precision.** Target SE 0.008; achieved 0.0066 (control,
  stopped on the criterion) and 0.025 / 0.018 (cells A and B, stopped at the
  300-replicate cap). Effects are 3–9 SE so no conclusion turns on it, but no
  claim here should be quoted at 0.008.
- **Screening precision** is 30 replicates per cell (coverage SE 0.03–0.09), not
  the 50 in v1.0. Recorded in protocol v1.1 before results were seen.
- **Nothing dropped.** Phase 3A: 3,340 histories generated, 2,275 no-overlap
  retained, 360 all-zero and 370 all-one paths kept. Phase 3B: 0 exceptions
  across 2,320 replicates; unresolved runs, out-of-grid targets and low-ESS runs
  all recorded and reported.

---

## 6. Phase 3 gate

**The gate is met.** At least one failure mechanism is:

- **reproducible** — present at screening (30 reps) and confirmatory (300 reps)
  scale, across two severities of the tail-perturbed family, across the robit
  family, and across three priors;
- **adequately powered** — H2 at 3.2–4.9 SE, the false-certainty elevation at
  9 SE, H4 at >20 SE, H1 at 41 SE;
- **practically meaningful** — a dangerous-error rate of 0.880 ± 0.019 at
  $q_{0.99}$ means that in seven runs out of eight the estimated safe threshold
  understates the true one by more than half a curve scale;
- **attributable** — to misspecification plus adaptivity, not implementation
  error: both inference backends agree (§2.3), the reference is audited (§5), the
  curves are matched on median and slope, and the non-adaptive contrast is paired
  on common random numbers.

**Recommendation: proceed to Phase 6 (safeguards), with the scope narrowed by
what Phase 3 found.**

The failure is not "misspecification" in general — it is *misspecification the
design never looks at*. That has a direct implication for what a safeguard must
do: it must spend some budget where the model is **not** already fitting well,
which is exactly the exploration-mixture idea in the plan, and the broad
exploratory design's behaviour in §3 is preliminary evidence that it works. A
model-averaging safeguard is less obviously matched to this mechanism, because
the competing links agree over the visited region too.

Three things to fix first, all cheap:

1. Restate H3 on the run-level false-certainty rate (D17), and make $w^\ast$
   scale-relative (D16).
2. Change the particle rejuvenation trigger to an ESS criterion, or lower the KL
   threshold (D15) — otherwise Phase 6 inherits a known numerical failure.
3. Run the confirmatory tier at the intended 0.008 (D18): ~1,900 replicates per
   cell, roughly 2 hours per cell on two cores.

**Not recommended:** a broad ladder of alternative link functions. §2.1 shows
that arm returns nothing in this model.

---

## 7. What would change these conclusions

- H1's scope rests on strata with a 68% event rate. If the unconditional rate of
  steep-curve no-overlap paths in a real application is genuinely ~2%, the
  practical importance is small even though the effect is large.
- The tail-perturbed family is *constructed* to be invisible to the design. That
  is a fair adversarial test, and it is honest to say it is an adversarial one:
  whether real sensitivity curves behave this way is a question about
  explosives, not about statistics, and the thesis should say so.
- Block I horizons above 50 were not run in the confirmatory tier. The plan's
  hypothesis that bias persists while the posterior shrinks — making false
  certainty *worse* at larger $n$ — is untested here and is the single most
  valuable addition.
