# Phase 3 protocol — v1.1

**Status: frozen before any Phase 3 experiment was run.**
Version history is at the end. Any change to a *confirmatory* definition after
results are seen must be recorded there with its reason; changes to exploratory
analyses may be made freely but must be labelled exploratory in the report.

Established from Phases 0–2 and **not re-litigated here**: the product
likelihood is exact under an ignorable policy; the reference posterior passes
SBC under adaptive and non-adaptive designs; the particle posterior at
$N=10{,}000$ is very close to the reference; the 14-case no-overlap coverage
gap is a hypothesis, not a finding; Rotem's Beta-CDF case raises tail MSE
without raising false certainty; the entropy-of-$\sigma$ discrepancy is
unresolved and that policy is **excluded** from all Phase 3 comparisons.

---

## 1. Hypotheses

Each is stated so that it can fail.

**Phase 3A — computational fidelity on difficult paths**

- **H1 (confirmatory).** On adaptively generated histories with *no overlapping
  pattern*, under the correct probit model, the particle posterior's coverage of
  the physical quantiles differs from the reference posterior's coverage on the
  *same* histories.
  *Effect of interest:* paired difference $\ge 0.02$ in absolute value.
  *Null:* $|\Delta| < 0.02$, i.e. no practically material difference.
  H1 is about the **paired** difference on identical histories. Conditional
  coverage below 0.95 in either backend is *not* evidence for H1 and is not a
  calibration failure — conditioning on an observable path feature is not the
  event SBC calibrates.

**Phase 3B — misspecification**

- **H2 (confirmatory).** Under a misspecified response curve whose target
  quantile lies inside the stimulus grid, the adaptive mutual-information design
  yields a **higher false-certainty rate** for $q_{0.95}$ or $q_{0.99}$ than a
  non-adaptive design at the same horizon and DGP.
  *Effect of interest:* difference $\ge 0.05$.
- **H3 (confirmatory).** There exists at least one (curve family, severity,
  policy, horizon) cell with coverage of $q_{0.95}$ or $q_{0.99}$ **below 0.80**
  while the mean interval width is **at or below** the correct-model threshold
  $w^*_{p,n}$ of §4. This is the "confidently wrong" regime the thesis is named
  after. If no such cell exists, H3 is reported as a **negative result** and not
  weakened into something else.
- **H4 (confirmatory).** For the same physical response curve, different
  policies induce different probit KL projections, and the implied
  pseudo-true $q_{0.95}$ differ across policies by $\ge 0.5\sigma_0$ (1.5 stress
  units at $\sigma_0 = 3$).
- **H5 (exploratory).** Design-support failure (target near or outside the
  stimulus grid) degrades coverage more than in-support misspecification of
  comparable RMSE. Exploratory because the two blocks cannot be matched on RMSE
  by construction, only compared after the fact.

---

## 2. Scenarios

### 2.1 Response-curve families

All alternatives are matched to the reference probit ($\mu_0=30$, $\sigma_0=3$)
on **median** and **slope at the median** ($\phi(0)/\sigma_0 = 0.132981$) by
numerical solution for the family's location and scale. Severity is then varied
**continuously** by convex combination on the probability scale,

$$p_\lambda(x) = (1-\lambda)\,\Phi\!\big((x-\mu_0)/\sigma_0\big) + \lambda\, p_{\text{alt}}(x),
\qquad \lambda \in [0,1].$$

This is monotone for every $\lambda$, equals the probit exactly at $\lambda=0$,
and — because both components share the median and the slope there — **preserves
both matched features at every $\lambda$**. Any observed failure is therefore
attributable to shape, not to a rescaling.

| # | family | matched parameters | severities $\lambda$ |
|---|---|---|---|
| 1 | probit (correct) | — | 0 |
| 2 | logistic | scale solved | 0.5, 1.0 |
| 3 | complementary log-log | location, scale solved | 0.5, 1.0 |
| 4 | Beta-CDF (Rotem's) | **not matched** — inherited case, labelled | 1.0 |
| 5 | robit, $\nu=3$ | scale solved | 0.5, 1.0 |
| 6 | monotone two-component probit mixture | offsets and scale solved | 0.5, 1.0 |
| 7 | targeted tail perturbation | exact below $p=0.85$ | shift 0.5, 1.0 $\sigma_0$ at $q_{0.95}$ |

Family 7 is constructed by perturbing the **quantile function**,
$q(p) = \mu_0 + \sigma_0 z_p + \kappa\,\sigma_0 A \max\{0, (p-0.85)/0.15\}^2$,
and inverting numerically. It is identical to the probit for $p \le 0.85$
(i.e. $x \lesssim 33.1$, which covers the region the adaptive designs actually
visit) and diverges only near $q_{0.95}$ and $q_{0.99}$. $\kappa$ is set so the
shift at $p=0.95$ equals the stated multiple of $\sigma_0$.

**Verification, run before any experiment:** every DGP is checked for
monotonicity on a dense grid, its physical quantiles are computed numerically
and cross-checked against a root-find of $p(x)=p$, and its median and slope are
checked against the probit reference. A DGP failing any check is not used.

### 2.2 Two separate blocks

These are analysed and reported separately and **never pooled**:

- **Block I — in-support misspecification.** Calibrated prior, stimulus grid
  $[16,44]$; every reported target quantile lies strictly inside the grid.
- **Block II — design-support failure.** Poorly-calibrated grid $[8,36]$ and/or
  a shifted truth, so that $q_{0.95}$ and/or $q_{0.99}$ lie near or outside the
  grid. Extrapolation beyond the design support is a distinct mechanism from
  shape misspecification.

### 2.3 Factors

| factor | levels |
|---|---|
| horizon $n$ | 20, 30, 50 (screening); 20, 30, 50, 100 (confirmatory) |
| target quantile | $q_{0.5}$, $q_{0.95}$, $q_{0.99}$ |
| prior | calibrated ($\mu_0=30$); shifted ($\mu_0=22$); narrow ($s_\mu=2$, $\tau^2=0.25$) |
| curve scale $\sigma_0$ | 1.5 (steep), 3 (moderate), 6 (shallow) |
| policy | see below |

### 2.4 Policies

Four, all ignorable:

1. `entropy_vector` — Rotem's adaptive MI design for $(\mu,\sigma)$.
2. `entropy_q0.95` — target-quantile adaptive design.
3. `fixed_design` — **non-adaptive reference**: stimuli cycled round-robin
   through five fixed levels $\mu_0 + \{-1.5,-0.75,0,0.75,1.5\}\times \tilde\sigma$
   with $\tilde\sigma$ the prior median scale. Chosen once from the prior, never
   updated.
4. `uniform_grid` — **broad exploratory**: $X_i \sim$ Uniform(candidate grid).

`entropy_sigma` is excluded (established conclusion 6). Dror–Steinberg and
Bruceton are not run in Phase 3: Phase 1 showed DS tracks `entropy_vector`
closely, and Bruceton is not restricted to the same design space.

---

## 3. Estimands and inference

**Estimands are the physical quantiles of the true curve**: $q_{0.5}$,
$q_{0.95}$, $q_{0.99}$, computed numerically from the DGP. Fitted probit
$(\mu,\sigma)$ are *not* estimands under misspecification and are reported only
as diagnostics.

**Primary inference:** the deterministic reference posterior. **Secondary
overlay:** Rotem's particle posterior on the identical history, to separate
model error from numerical error.

Coverage is evaluated as $\alpha/2 < P(Q < Q^\star \mid \text{data}) < 1-\alpha/2$
at $\alpha = 0.05$, identically for both backends (Phase 2 convention, D11).

---

## 4. False-certainty thresholds — fixed before misspecified results

$$w^\ast_{p,n} = \text{25th percentile of the reference posterior's 95\% interval width for } q_p$$

under the **correct probit** DGP ($\mu_0=30$, $\sigma_0=3$), calibrated prior,
`entropy_vector` design, at horizon $n$. One threshold per $(p, n)$, computed
once from a dedicated baseline run, written to
`configs/phase3_thresholds.json` with a hash, and then applied **unchanged** to
every DGP, policy and block at that horizon. Thresholds are never recomputed
within a misspecified cell.

Two events are reported:

- **False certainty:** $\Pr(\text{interval misses } Q^\star \text{ and width} \le w^\ast_{p,n})$.
- **Dangerous error:** $\Pr(\hat q_p - q_p^\star < -\delta)$ with
  $\delta = 0.5\,\sigma_0^{\text{true}}$ — the estimated threshold understates
  the true one by more than half a curve scale, i.e. an operating stress
  declared acceptable that is materially unsafe. $\delta$ is a fraction of the
  *true* curve's scale so it is comparable across DGPs. Sensitivity at
  $\delta = 1.0\,\sigma_0^{\text{true}}$ is also reported.

Also reported per cell: coverage, bias, RMSE, mean and median interval width,
asymmetric threshold decision loss over cost ratios $\{1,2,5,10,20,50,100\}$,
unresolved-experiment rate (all-zero, all-one or no overlap), and
overlap/sampled-support summaries.

---

## 5. Replication and stopping

Two stages.

**Screening.** Fixed replicate counts, no interim inspection: 50 replicates per
cell (Block I core), 40 per cell (scale and prior sub-blocks and Block II).
Coverage standard error $\approx 0.02$–$0.03$ — enough to rank cells, not to
certify any. **Every screened cell is reported, including negative ones.**

**Confirmatory.** Three cells, chosen by the rule in §6, run in batches of 250
replicates until either

1. the Monte Carlo standard error of the cell's primary metric is $\le 0.008$, or
2. the cell's CPU budget of 3,000 s is exhausted,

whichever comes first. The stopping rule depends only on the replicate count and
elapsed cost — **never on the observed value of any metric** — so it cannot bias
the estimate. The achieved standard error is reported for every cell.

*Deviation from the plan, recorded here in advance:* the implementation plan
asks for coverage standard error $\le 0.005$, which needs $\approx 1{,}900$
replicates per cell. On the two cores available that is $\approx 1.5$ h per cell.
The budget permits $\approx 0.008$. Claims that would need 0.005 to resolve are
reported as suggestive, not confirmatory.

**Phase 3A stopping.** Generate histories under stratified enrichment until
either 2,000 no-overlap histories are retained or the 3A CPU budget (4,000 s) is
exhausted. Report the total generated, the event rate per stratum, and the
achieved standard error. Every no-overlap path is retained; a size-matched
overlap control sample is drawn from the same strata.

**Common random numbers** across policies within a (DGP, prior, scale, horizon)
cell. **Nothing is dropped**: all-zero, all-one, no-overlap, low-ESS,
out-of-grid, non-finite MLE and exception-raising runs are recorded with flags
and counted in the manifest.

---

## 6. Confirmatory cell selection rule

Fixed now, applied mechanically to the screening output:

1. **Cell A** = the (DGP, policy) cell in Block I with the largest screened
   false-certainty rate for $q_{0.95}$ or $q_{0.99}$ at $n=50$, among adaptive
   policies. Ties broken by lower coverage, then by DGP order in §2.1.
2. **Cell B** = the same DGP and target as Cell A, with policy `fixed_design`.
   This is the contrast that attributes any failure to adaptivity rather than to
   the DGP.
3. **Cell C** = correct probit with the same policy as Cell A — the control that
   fixes the false-certainty rate under a correct model at the same horizon.

Confirmatory runs extend to $n=100$ so that persistent bias under a shrinking
posterior can be seen.

If the largest screened false-certainty rate is not distinguishable from the
correct-model control at screening precision, Cells A–C are still run — a
powered null is a result — and H3 is reported as negative.

---

## 7. Policy-dependent pseudo-truth

For each (DGP, policy) in the confirmatory set and for the Block I core cells:

1. estimate the empirical stimulus distribution $\nu_a$ produced by the policy,
   pooled over replicates at the final horizon;
2. compute the probit KL projection
   $\theta^\dagger_a = \arg\max_\theta \int [p^\star(x)\log p_\theta(x) +
   (1-p^\star(x))\log(1-p_\theta(x))]\,d\nu_a(x)$
   by direct numerical optimisation over $(\mu,\log\sigma)$;
3. derive the implied fitted quantiles $q_p(\theta^\dagger_a)$;
4. compare against the **physical** $q_p^\star$ and against the posterior mean
   at the largest horizon.

H4 tests whether $q_{0.95}(\theta^\dagger_a)$ differs across policies. A
bootstrap over replicates gives its Monte Carlo uncertainty.

---

## 8. Numerical rules

- Likelihood weights stay in log space.
- Estimated scalar mutual information is **clipped at zero** for any new Phase 3
  design; the unclipped minimum and the fraction clipped are recorded. Rotem's
  original unclipped convention remains available via a config flag for
  sensitivity analysis (D14).
- The reference posterior runs at a **validated fixed resolution** in production.
  Adequacy is established by `experiments/phase3_reference_audit.py`, which
  compares the fixed-resolution build against the self-converging adaptive build
  across DGPs, policies and horizons; the audit must show agreement below
  $10^{-3}$ of a posterior standard deviation or the resolution is raised.
  Runs whose reference fails its diagnostics are flagged, not silently used.
- Every configuration, seed, manifest, commit and Monte Carlo standard error is
  recorded.

---

## 9. Decision gates

**Proceed to Phase 6 (safeguards) only if at least one failure mechanism is
simultaneously:** reproducible across seeds and neighbouring cells; adequately
powered at the confirmatory precision achieved; practically meaningful against
the §4 thresholds; and attributable to adaptivity, misspecification or weak
exploration rather than to implementation error — which requires that the
reference and particle backends agree on it.

**If no false-certainty regime is found**, that is the reported result. The
recommendation then becomes to evaluate the alternative thesis contribution:
that adaptive Bayesian inference in this model remains appropriately uncertain
across a broad, controlled misspecification class — a negative result with real
content, since it contradicts the motivating intuition.

**Stop after the validation report.** No safeguards, no multivariate extension,
no AI-safety application, no later theoretical phases.

---

## Version history

- **v1.0** — frozen before any Phase 3 experiment.
- **v1.1** — recorded *during* the screening run, before any misspecified result
  was inspected. Two implementation deviations, both driven by the two-core
  compute budget, neither touching a confirmatory definition:
  1. **Reference posterior evaluated at n = 50 only** in the screening tier,
     rather than at every horizon. Cost: a reference build plus two quantile
     bisections is ~1.4 s per replicate per horizon, and evaluating three
     horizons would have tripled the screening cost. Consequence: the
     *primary* (reference-based) analysis, including the frozen $w^\ast$
     thresholds and every false-certainty number, exists at **n = 50 only**.
     Horizon trends at n = 20 and n = 30 are reported from the particle
     posterior and are labelled secondary. Phase 2 established the two agree
     closely under prior-typical parameters, which is the regime Block I sits
     in; Phase 3A quantifies where they do not.
  2. **Screening replicate counts** set to 40 (Block I) and 30 (sub-blocks and
     Block II) rather than 50/40. Coverage standard error ~0.035 rather than
     ~0.03. Screening remains exploratory either way.

  Neither change alters the hypotheses, the estimands, the threshold *rule*,
  the confirmatory selection rule, or the gates.
