# Phase 3 failure atlas

Companion to `phase3_protocol.md` (v1.1) and `phase3_validation_report.md`.
This document is the *map*: what fails, where, how badly, and by which
mechanism. Judgements about power and about the Phase 3 gate live in the
validation report.

Everything is at horizon $n=50$ unless stated, from the **reference posterior**
(primary inference), on the physical quantiles of the true curve.

---

## 0. The curves

Twelve data-generating curves. Every alternative is matched to the reference
probit ($\mu_0=30$, $\sigma_0=3$) on **median** and **slope at the median**
($0.132981$), so nothing below is a rescaling artefact. Verified before use:
all monotone, quantile inversion error $\le 2\times10^{-11}$.

| DGP | $q_{0.5}$ | $q_{0.95}$ | $q_{0.99}$ | latent SD | note |
|---|---|---|---|---|---|
| probit (correct) | 30.000 | 34.935 | 36.979 | 3.000 | control, nested at $\lambda=0$ |
| logistic 0.5 / 1.0 | 30.000 | 35.212 / 35.536 | 37.816 / 38.639 | 3.21 / 3.41 | |
| cloglog 0.5 / 1.0 | 30.000 | 34.335 / 33.815 | 36.197 / 34.935 | 3.19 / 3.34 | **shorter** upper tail |
| robit($\nu$=3) 0.5 / 1.0 | 30.000 | 35.577 / 36.505 | 39.738 / 42.550 | 3.88 / 4.60 | heavy tails |
| mixture 0.5 / 1.0 | 30.000 | 34.867 / 34.801 | 36.830 / 36.678 | 2.96 / 2.92 | shoulder |
| tail-perturbed 0.5 / 1.0 | 30.000 | 36.435 / 37.935 | 39.919 / 42.859 | 3.36 / 3.77 | **exactly probit below $x\approx33.1$** |
| Beta-CDF (Rotem's) | 29.356 | 36.274 | 38.928 | 3.90 | inherited, *not* matched |

The tail-perturbed family is the one built specifically to defeat the mechanism
Phase 2 identified, where the Beta-CDF's global shape mismatch inflated the
posterior width enough to absorb its own bias. Here the likelihood sees almost
nothing wrong over the region the design visits.

---

## 1. Block I — in-support misspecification

Calibrated prior, grid $[16,44]$, all targets inside. 30 replicates per cell
(screening precision, coverage SE $\approx 0.03$–$0.09$).

### 1.1 Coverage of the physical quantile

$q_{0.99}$, reference posterior:

| DGP | adaptive MI $(\mu,\sigma)$ | adaptive MI $q_{0.95}$ | non-adaptive fixed | broad exploratory |
|---|---|---|---|---|
| probit | 0.967 | 0.900 | 0.967 | 0.900 |
| logistic 1.0 | 0.933 | 0.900 | 0.967 | 0.800 |
| cloglog 1.0 | **1.000** | 0.933 | 0.867 | 0.800 |
| mixture 1.0 | 0.967 | 0.967 | 0.967 | 0.933 |
| Beta-CDF | 0.967 | 0.933 | 1.000 | 0.900 |
| robit 1.0 | **0.500** | 0.700 | 0.833 | 0.700 |
| tail 0.5 | 0.833 | 0.800 | 0.867 | 0.767 |
| **tail 1.0** | **0.400** | 0.600 | 0.600 | 0.500 |

**The ordering of families is the finding.** Logistic, cloglog and mixture
misspecification — the classical "wrong link" cases — barely move coverage at
all, even at full severity. What breaks coverage is misspecification *located in
the tail being estimated*: robit (heavy tails) and above all the targeted tail
perturbation.

This is a sharp negative result for the intuition that "wrong link ⟹ unreliable
tail inference". A wrong link that is wrong in the *middle*, where the data are,
is absorbed. A curve that agrees with the model everywhere the design looks and
disagrees only where the estimand lives is not absorbed at all.

### 1.2 False certainty: misses the truth **and** width $\le w^\ast$

Thresholds frozen from the correct-probit adaptive baseline before any
misspecified cell was scored (hash `b73c1db6fb3a8d9f`):
$w^\ast_{0.5,50}=2.524$, $w^\ast_{0.95,50}=4.962$, $w^\ast_{0.99,50}=6.505$.

$q_{0.99}$:

| DGP | adaptive MI | adaptive $q_{0.95}$ | fixed | exploratory |
|---|---|---|---|---|
| probit | 0.033 | 0.100 | 0.000 | 0.000 |
| logistic 1.0 | 0.067 | 0.100 | 0.033 | 0.067 |
| cloglog 1.0 | 0.000 | 0.033 | 0.000 | 0.000 |
| robit 1.0 | 0.133 | 0.200 | 0.100 | 0.033 |
| tail 0.5 | 0.133 | 0.200 | 0.100 | 0.067 |
| **tail 1.0** | **0.267** | 0.233 | 0.167 | 0.067 |

Against the correct-model control of 0.033, the tail-perturbed curve raises
false certainty about **eight-fold** under the adaptive MI design. The broad
exploratory design is consistently the *least* falsely certain — it buys
honesty with width (mean $q_{0.99}$ width 9.8 vs 7.5 adaptive).

**H3, as literally pre-registered, is not met in Block I.** The condition was
coverage $<0.80$ *and* **mean** width $\le w^\ast$. For `tail_1.0` × adaptive MI
at $q_{0.99}$, coverage is 0.400 but the mean width is 7.48, just above
$w^\ast=6.505$. The per-replicate false-certainty rate is nonetheless 0.267,
because a quarter of individual runs are both narrow and wrong. The distinction
is real and is reported as stated rather than relaxed.

### 1.3 Widths tell the story

Mean $q_{0.99}$ interval width, `tail_1.0`: adaptive MI 7.48, adaptive
$q_{0.95}$ 8.55, fixed 12.55, exploratory 9.77. The adaptive design produces
**the narrowest intervals and the worst coverage** — which is what "confidently
wrong" means. The non-adaptive design is wrong too (coverage 0.600) but says so.

---

## 2. Block II — design-support failure

Poorly-calibrated grid $[8,36]$; every target lies **outside the sampled
support in 100% of runs**. Reported separately and never pooled with Block I.

| DGP | policy | target | truth | coverage | bias | width | false certainty | dangerous error | unresolved |
|---|---|---|---|---|---|---|---|---|---|
| tail 1.0 | adaptive MI | $q_{0.99}$ | 42.86 | **0.318** | −5.68 | 7.18 | **0.409** | **1.000** | 0.000 |
| tail 1.0 | adaptive $q_{0.95}$ | $q_{0.99}$ | 42.86 | 0.364 | −5.16 | 6.93 | 0.364 | 0.955 | 0.000 |
| tail 1.0 | fixed | $q_{0.99}$ | 42.86 | 0.955 | −9.62 | 17.75 | 0.000 | 1.000 | 0.955 |
| probit shifted to 38 | fixed | $q_{0.95}$ | 42.94 | **0.000** | −10.43 | 15.40 | 0.000 | 1.000 | 1.000 |
| probit shifted to 38 | adaptive MI | $q_{0.95}$ | 42.94 | 1.000 | −1.16 | 11.03 | 0.000 | 0.545 | 0.091 |

Two distinct things happen here and they should not be conflated:

1. **Adaptivity rescues the point estimate and destroys the interval.** With the
   truth at $\mu=38$ and a grid ending at 36, the adaptive design walks to the
   grid edge and recovers a usable estimate (bias −1.16) with honest width; the
   non-adaptive design, pinned near $\mu_0=22$, never sees a response at all
   (unresolved rate 1.000) and has coverage exactly **zero**.
2. **Under the tail perturbation the adaptive design is the worst of both.**
   Coverage 0.318, false certainty 0.409, dangerous-error rate 1.000 — every
   single run underestimates $q_{0.99}$ by more than half a curve scale.

The `dangerous error` column is the practically important one: it is the rate at
which the estimated safe threshold understates the true one by more than
$0.5\sigma^{\text{true}}$. In Block II it is at or near 1.0 for almost every
cell, adaptive or not. **Extrapolating a fitted probit to a quantile outside the
stimulus range that was actually applied is unsafe regardless of design.**

---

## 3. Sub-blocks: curve scale and prior

### 3.1 Scale (`Ib`) — and a threshold caveat

`tail_1.0` × adaptive MI: at $\sigma_0=1.5$, $q_{0.99}$ coverage 0.500 with
width 4.39; at $\sigma_0=6$, coverage 0.227 with width 13.93.

**Caveat, discovered in analysis and recorded here rather than silently fixed.**
$w^\ast$ was frozen at $\sigma_0=3$, as the protocol requires. But the *true
curve scale* sets the intrinsic width scale, so comparing a $\sigma_0=1.5$ cell
against a $\sigma_0=3$ threshold flags almost everything as "narrow", and a
$\sigma_0=6$ cell as never narrow. The false-certainty column is therefore
**not comparable across the scale sub-block**, and the apparent H3 hits at
$\sigma_0=1.5$ (coverage 0.636, width 3.33 $\le$ 4.96) are an artefact of that
mismatch, not evidence. The protocol's "one threshold per $(p,n)$" rule
implicitly assumed a fixed true scale. A scale-relative threshold
($w^\ast/\sigma^{\text{true}}$) is the obvious repair and is recommended for
Phase 6, but it is *not* substituted here, because changing a pre-registered
definition after seeing results is exactly what the protocol forbids.

What *is* comparable within the sub-block: at $\sigma_0=6$ the tail-perturbed
curve gives $q_{0.99}$ coverage 0.227 with RMSE 12.05 — the worst coverage
anywhere in Phase 3B — while the correct probit at the same scale is at 0.955.

### 3.2 Prior (`Ic`)

`tail_1.0` × adaptive MI, $q_{0.95}$: calibrated 0.455 (Block I equivalent
0.667), narrow prior 0.591, shifted prior 0.455. False certainty 0.409–0.455.
The failure is **not** driven by prior misspecification: it is present with a
well-calibrated prior and is not much worsened by a bad one.

---

## 4. Policy-dependent pseudo-truth (H4)

Probit KL projections under each policy's empirical design distribution, with a
replicate-level bootstrap. Block I, $\sigma_0=3$:

| DGP | target | physical | min over policies | max over policies | spread | SE |
|---|---|---|---|---|---|---|
| tail 1.0 | $q_{0.99}$ | 42.86 | 37.13 (fixed) | 39.59 (exploratory) | **2.47** | 0.08 |
| — adaptive MI 37.71, adaptive $q_{0.95}$ 39.47 | | | | | |
| robit 1.0 | $q_{0.99}$ | 42.55 | 37.65 (fixed) | 39.96 (exploratory) | **2.31** | 0.05 |
| tail 1.0 | $q_{0.95}$ | 37.94 | 35.05 (fixed) | 36.93 (exploratory) | **1.88** | 0.06 |
| robit 1.0 | $q_{0.95}$ | 36.51 | 35.41 (fixed) | 37.05 (exploratory) | **1.65** | 0.04 |
| cloglog 1.0 | $q_{0.99}$ | 34.94 | 36.05 | 37.25 | 1.20 | 0.08 |
| logistic 1.0 | $q_{0.99}$ | 38.64 | 37.31 | 38.10 | 0.80 | 0.02 |
| probit (control) | $q_{0.99}$ | 36.979 | 36.979 | 36.979 | **0.000** | — |

**H4 is confirmed.** Four Block I cells exceed the pre-registered 1.5-unit
threshold, with standard errors under 0.1. The same physical curve is projected
onto probit curves whose implied $q_{0.99}$ differ by up to 2.47 stress units
depending only on *which policy ran the experiment*.

Two further points the table makes:

- **Every policy's pseudo-truth is biased low** for the tail-perturbed and robit
  curves — by up to 5.73 units for `tail_1.0` $q_{0.99}$. So the posterior is not
  converging to the physical truth under *any* design; the policies differ in how
  badly they miss, not in whether they miss.
- The ordering is systematic: `fixed_design` projects lowest and `uniform_grid`
  highest, because the broad exploratory design puts weight further up the curve
  and so its KL projection is pulled toward the tail. This is the mechanism
  stated in the plan, observed directly.

---

## 5. Is any of this numerical?

Across all **228** screened (block, DGP, policy, target) cells at $n=50$, the
particle-minus-reference coverage difference has mean $+0.0023$, 5th–95th
percentile $[0.000, +0.033]$, maximum absolute value $0.067$.

**No Phase 3B conclusion is numerical.** Under the prior-typical parameters of
Blocks I and II, Rotem's particle posterior and the exact reference agree to
within screening noise, exactly as Phase 2 established. The failures above are
properties of the model and the design, not of the filter.

That separation is what makes Phase 3A's contrasting result interpretable: the
particle filter *does* break, but only in a regime (very steep curves, deep in
the prior tail) that Blocks I and II never enter.

---

## 6. Summary map

| mechanism | where it bites | signature | attributable to |
|---|---|---|---|
| **Tail-located misspecification** | curve agrees with probit over the visited region, differs at the estimand | coverage 0.40–0.50, narrowest intervals, false certainty 8× control | misspecification + adaptivity |
| **Heavy tails (robit)** | same, milder | coverage 0.50–0.70 | misspecification |
| **Wrong link in the middle** (logistic, cloglog, mixture) | — | **no effect**; coverage 0.93–1.00 | negative result |
| **Design-support failure** | target outside applied stimuli | dangerous-error rate ~1.0 for every design | extrapolation |
| **Non-adaptive design, wrong prior centre** | truth far from $\mu_0$ | coverage 0.000, unresolved 1.000 | weak exploration |
| **Particle degeneracy** | steep curves in the prior tail (Phase 3A) | particle sd 16% too narrow, coverage gap −0.56 | numerical (rejuvenation rule) |
