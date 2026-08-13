# Phase 0 note — formal corrections and the revised thesis question

*Prepared for the advisor meeting. Companion to `Thesis_Implementation_Plan.md`.
Every claim below is enforced by a test in `tests/`; the test name is given in
each case so the note and the code cannot drift apart.*

---

## 1. Adaptive dependence does not invalidate the product likelihood

**Setup.** Let $H_i=(X_1,Y_1,\dots,X_i,Y_i)$ be the history, $\theta$ the response
parameter, and let the design be chosen by a policy

$$X_i \mid H_{i-1} \sim q_i(\cdot \mid H_{i-1}).$$

Call the policy **ignorable** if it is *predictable* (measurable with respect to
$H_{i-1}$) and *parameter-free* ($q_i$ does not depend on $\theta$).

**Proposition 1.** Under an ignorable policy,

$$p_\theta(H_n)=\prod_{i=1}^{n} q_i(x_i\mid H_{i-1})\,p_\theta(y_i\mid x_i)
=\Big[\underbrace{\prod_{i=1}^{n} q_i(x_i\mid H_{i-1})}_{\text{free of }\theta}\Big]\prod_{i=1}^{n} p_\theta(y_i\mid x_i),$$

hence for the observed history

$$p_\theta(H_n)\;\propto_\theta\;\prod_{i=1}^{n} p_\theta(y_i\mid x_i),$$

and the posterior obtained from Rotem's product likelihood is the **exact**
Bayesian posterior under the assumed response model.

*Proof.* Factor the joint law of $H_n$ by the chain rule along the natural
filtration, alternating design and response steps. Each design factor is
$q_i(x_i\mid H_{i-1})$, which by assumption does not involve $\theta$; each
response factor is $p_\theta(y_i\mid x_i)$ because, given $X_i=x_i$, the response
depends on the past only through $x_i$. Collecting the design factors gives a
constant in $\theta$ for the observed history. $\square$

**What this closes.** The marginal law of $(Y_1,\dots,Y_n)$ *is* dependent — the
$Y_i$ are not i.i.d. and not even exchangeable — but the likelihood is a
conditional object, and the conditioning removes exactly that dependence. Rotem's
likelihood is not a pseudo-likelihood, and no correction for "ignoring the
dependence" is warranted. **This question is closed and is not the thesis.**

**Where ignorability can actually fail.** The proposition has real content, and
the conditions are not vacuous:

| Failure | Example in practice |
|---|---|
| policy not parameter-free | operator adjusts stresses using knowledge of the true system, or an unrecorded pilot study |
| policy not predictable | stimulus chosen using an outcome not yet in the record, or retrospective editing of the design |
| policy depends on unrecorded data | manual overrides, discarded runs, "bad unit" exclusions |

The last row is the practically dangerous one: **dropping runs is itself a policy**,
and a policy that conditions on the outcome is not ignorable.

**Tests.** `test_factorization.py::test_product_likelihood_is_proportional_to_full_history`
checks that $\log p_\theta(H_n)-\sum_i\log p_\theta(y_i\mid x_i)$ is constant across a
$\theta$-grid to $<10^{-9}$, for a deterministic adaptive policy, a randomised
adaptive policy (softmax over mutual information) and a randomised non-adaptive
policy. `test_posteriors_agree_under_ignorable_policies` checks the two posteriors
agree in total variation to $<10^{-12}$.
`test_non_ignorable_policy_breaks_the_invariant` is the negative control: an oracle
policy that steers towards the *true* median makes the residual vary by more than
1 nat and shifts the posterior by more than 0.01 in total variation. Without that
control the first test would be passed by an implementation that simply ignored
the design term.

---

## 2. Statistical exactness vs numerical approximation

These are different failure modes and the thesis must not conflate them.

| | source | detected by | remedy |
|---|---|---|---|
| **Statistical** | wrong response family; non-ignorable design | SBC failure that persists under an exact posterior; misspecification experiments | model class, design |
| **Numerical** | finite particle set, weight degeneracy, rejuvenation, quadrature | disagreement between the particle posterior and a converged reference | more particles, better SMC |

The project therefore carries **two** posteriors for the same model: a
deterministic refined-grid reference on $(\mu,\eta=\log\sigma)$, and Rotem's
weighted-particle posterior. The reference is only used once it has demonstrated
convergence against a denser grid, and reports `converged=False` otherwise, so an
unconverged reference can never be silently consumed.

*Numerical note.* Two implementation details turned out to matter more than
expected and are recorded because they would otherwise look like statistical
findings:

1. Weights must be carried in log space. Rotem's written recursion normalises raw
   likelihoods, which underflows to $0/0$ around $n\approx40$ in double precision.
2. The reference posterior's CDF must resolve the partial grid cell cut by the
   threshold. Counting whole cells leaves an $O(h)$ lattice error which, on the wide
   box that $\sigma$'s heavy tail forces, dominated every other error in the system.
   Resolving the partial cell restored $O(h^2)$ and improved refinement stability
   by a factor of ~75.

---

## 3. Two quantiles determine the model; a third adds nothing

Under the probit location-scale family, $q_p=\mu+\sigma z_p$ with $z_p=\Phi^{-1}(p)$.

**Proposition 2.** For $p_1\neq p_2$ the map $(\mu,\sigma)\mapsto(q_{p_1},q_{p_2})$ is a
bijection onto its range, with inverse

$$\sigma=\frac{q_{p_2}-q_{p_1}}{z_{p_2}-z_{p_1}},\qquad \mu=q_{p_1}-\sigma z_{p_1}.$$

Consequently, for any stimulus $x$ and history $H_n$,

$$I\big((q_{p_1},q_{p_2});Y_{n+1}\mid x,H_n\big)=I\big((\mu,\sigma);Y_{n+1}\mid x,H_n\big),$$

since mutual information is invariant under bijective reparameterisation of either
argument. **Targeting "all quantiles" is therefore redundant once two distinct
quantiles are targeted**, and the value is the same for every choice of the pair.

**Tests.** `test_reparameterization.py` checks the round trip to $10^{-10}$ over five
$(p_1,p_2)$ pairs, checks MI equality to $10^{-10}$ for each pair, checks that a
twelve-quantile coordinate system gives the identical answer, and checks that a
*single* quantile carries strictly less information — so the scalar estimator is
demonstrably not just returning the vector answer.

**The version of David's question that is not redundant.** Indexing candidate
*stimuli* by response level is a genuinely different object. With
$\tilde q_r(H_n)$ the posterior summary of the stimulus at which the response
probability is $r$, define

$$U_p(r;H_n)=I\big(q_p;Y_{n+1}\mid X_{n+1}=\tilde q_r(H_n),H_n\big).$$

This surface answers *where on the sensitivity curve to push* in order to learn a
median, a scale, or a tail threshold, and it is estimated in Phase 1.

---

## 4. The revised thesis question

> In a Bayesian adaptive sensitivity experiment, when are posterior credible
> intervals and threshold decisions about response **quantiles** trustworthy, and
> how can the design be made safer when the response model is misspecified or the
> experiment has not explored an informative stimulus region?

The primary estimands are physical quantiles — $q_{0.5}$, $q_{0.95}$, $q_{0.99}$ —
not $(\mu,\sigma)$, because under misspecification the fitted $(\mu,\sigma)$ have no
physical referent while a quantile of the true curve always does. The primary
metrics are calibration and decision loss, not MSE: an interval can be narrow,
wrong, and still have acceptable average MSE.

---

## 5. One-page experiment summary

**Ladder of response curves.** Correct probit → logistic (matched median and
slope) → complementary log-log / skew-probit → Beta-CDF (Rotem's own asymmetric
case) → robit → monotone two-component mixture. Misspecification is varied
*continuously* from zero, with median and local slope matched, so that any failure
is attributable to shape rather than to an arbitrary rescaling.

**Factors.** target quantile $p\in\{0.5,0.95,0.99\}$; horizon $n\in\{20,30,50,100\}$;
prior (calibrated / shifted / narrow / heavy-tailed); stimulus grid (truth inside /
near boundary / outside); curve scale (steep / moderate / shallow); policy
(non-adaptive reference, Rotem MI for $(\mu,\sigma)$, MI for one quantile,
two-quantile utility, exploration mixture); inference (grid reference, Rotem
particles, robust alternative).

**Metrics.** Credible-interval coverage per quantile; bias and RMSE; interval
width; **false-certainty rate** — the probability that an interval misses the truth
*while being narrower than a prespecified threshold*; asymmetric threshold decision
loss reported over a range of cost ratios rather than one unverifiable ratio; and
the probability of an unresolved experiment. Reported in four separate layers
(SBC, fixed-$\theta$ coverage, path-stratified coverage, decision calibration),
never collapsed into a single score.

**Replication.** Driven by a target Monte Carlo standard error, not by matching
Rotem's $S=500$; common random numbers across policies so that comparisons are
paired.

**A design-support hazard already present in Rotem's settings.** Under her
poorly-calibrated prior the stimulus grid is $[8,36]$, but the true
$x_{0.99}=36.98$ lies **outside it**. Any $q_{0.99}$ reported from that setting is a
model extrapolation beyond every stimulus the experiment could have applied. This
is exactly the weak-identification regime the thesis is about, and it is present in
the baseline rather than manufactured for it.

---

## 6. Gate for this phase

Advisor agreement that (i) the dependence-based approximation question is closed by
Proposition 1, and (ii) the primary estimand becomes a physical quantile of the
true response curve.
