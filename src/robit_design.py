"""The robit design arm: a particle posterior over (mu, sigma, r) and its state.

Separable by construction.  The Phase 6 pilot crosses two axes that must not be
folded together:

  * **fitted family** -- what the reported posterior assumes.  Probit, or robit
    with a free tail parameter.  This is `posterior_grid` versus
    `posterior_grid_robit`, applied to the *same* history.
  * **design** -- what the mutual-information criterion is computed over.
    ``I((mu, sigma); Y | x)`` as Rotem defines it, or ``I((mu, sigma, r); Y | x)``.

The second is the interesting question on its own: r is identified only in the
tail, so a criterion that scores information about r has a reason to go there
that the two-parameter criterion does not have.  Whether it actually does is an
empirical question this module makes askable.

Implementation notes
--------------------
`EntropyVectorPolicy` computes ``h(E[p]) - E[h(p)]`` over whatever parameter
vector the particle cloud carries, so it is I((mu, sigma, r); Y | x) with no
change once the cloud is three-dimensional and `probs_T` uses the robit link.
That is why this module subclasses `PolicyState` rather than touching the
policy: the design code is shared, and the arms differ only in the posterior
they are handed.

**No rejuvenation.**  Rotem's rejuvenation step is a Gaussian refresh in the
GLM parameterisation ``beta = (-mu/sigma, 1/sigma)``, which has no meaning for
r and no obvious three-parameter analogue that would still be *her* method.
Extending it is a research question of its own and belongs to a later rung, not
to the first safeguard.  Two consequences, both recorded rather than absorbed:

  * the particle *values* never change, so the (L, N) response-probability
    matrix is built once per replicate instead of once per rejuvenation.  This
    is what makes a Student-t design affordable at all: the t CDF is far more
    expensive than `ndtr`, and paying for it 50 times per run would not be.
  * effective sample size decays monotonically over the run.  It is recorded at
    every step and summarised per cell, and the protocol pre-registers an
    adequacy floor below which the robit-*design* arm is reported as
    computationally inadequate rather than as a design finding.  C8 is the
    precedent: turning rejuvenation off beat Rotem's rule despite a 30x lower
    ESS, so a low ESS is a number to report, not a reason to discard a run.

r is drawn from the **discretised** r-prior, on the same node set the reference
grid uses.  Two reasons: particles then share degrees of freedom, so one
`t.cdf` call serves every particle at a node; and the design's prior and the
reference's prior are then the same object evaluated the same way, so a
difference between them cannot be a discretisation artefact.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .policies import PolicyState
from .robit import RPrior, robit_logprobs_z, robit_prob_matrix_T, robit_z

DEFAULT_N_PARTICLES = 20_000


class RobitParticlePosterior:
    """Weighted prior-particle posterior over (mu, sigma, r).

    Deliberately the simplest correct thing: draw from the prior, weight by the
    normalised likelihood of the whole history in log space, never resample.
    Under an ignorable policy the product likelihood is exact (C1), so this is
    the exact posterior up to the Monte Carlo error of the cloud -- an error
    that is measured by ESS and reported, not assumed away.
    """

    def __init__(self, prior, r_prior: RPrior, n_particles: int,
                 rng: np.random.Generator, n_r_nodes: int = 17):
        self.prior = prior
        self.r_prior = r_prior
        self.n = n_particles
        self.rng = rng

        mu, sigma = prior.sample(n_particles, rng)
        self.mu = mu
        self.sigma = sigma
        self.r_nodes = r_prior.nodes(n_r_nodes)
        self.r = _draw_r(r_prior, self.r_nodes, n_particles, rng)

        # Never bumped: there is no rejuvenation, so every value-keyed cache
        # downstream (`PolicyState._cached`) stays valid for the whole run.
        self.generation = 0
        self.log_lik = np.zeros(n_particles)
        self._set_weights()

        self.x_hist: list[float] = []
        self.y_hist: list[int] = []
        self.ess_trace: list[float] = []

    # -- weights ----------------------------------------------------------
    def _set_weights(self):
        finite = np.isfinite(self.log_lik)
        if not np.any(finite):
            raise FloatingPointError("all robit particles have zero posterior weight")
        norm = logsumexp(self.log_lik[finite])
        logw = np.full_like(self.log_lik, -np.inf)
        logw[finite] = self.log_lik[finite] - norm
        self.logw = logw
        self.w = np.exp(logw)

    def ess(self) -> float:
        return float(1.0 / np.sum(self.w**2))

    # -- sequential update ------------------------------------------------
    def update(self, x: float, y: int):
        z = (float(x) - self.mu) / self.sigma
        ll = np.empty_like(z)
        for rv in self.r_nodes:
            sel = self.r == rv
            if not np.any(sel):
                continue
            lp, lq = robit_logprobs_z(z[sel], float(rv))
            ll[sel] = lp if y == 1 else lq
        self.log_lik = self.log_lik + ll
        self._set_weights()
        self.x_hist.append(float(x))
        self.y_hist.append(int(y))
        self.ess_trace.append(self.ess())
        return self.ess_trace[-1]

    # -- summaries --------------------------------------------------------
    def values(self, quantity: str):
        if quantity == "mu":
            return self.mu
        if quantity == "sigma":
            return self.sigma
        if quantity == "r":
            return self.r
        if quantity.startswith("q"):
            return self.mu + self.sigma * robit_z(float(quantity[1:]), self.r)
        raise ValueError(f"unknown quantity {quantity!r}")

    def mean(self, quantity: str) -> float:
        return float(np.dot(self.w, self.values(quantity)))

    def sd(self, quantity: str) -> float:
        v = self.values(quantity)
        m = float(np.dot(self.w, v))
        return float(np.sqrt(max(np.dot(self.w, (v - m) ** 2), 0.0)))

    def snapshot(self) -> dict:
        return {"ess": self.ess(), "max_weight": float(self.w.max()),
                "ess_min": float(np.min(self.ess_trace)) if self.ess_trace else np.nan,
                "n_distinct_r": int(len(np.unique(self.r))),
                "n_particles": self.n}


def _draw_r(r_prior: RPrior, nodes, n: int, rng: np.random.Generator):
    """Draw r from the prior discretised onto `nodes` (rectangle rule)."""
    if len(nodes) == 1:
        return np.full(n, float(nodes[0]))
    lp = r_prior.logpdf(nodes)
    lp = lp - logsumexp(lp[np.isfinite(lp)])
    p = np.where(np.isfinite(lp), np.exp(lp), 0.0)
    p = p / p.sum()
    return nodes[rng.choice(len(nodes), size=n, p=p)]


class RobitPolicyState(PolicyState):
    """`PolicyState` with the robit link in the response-probability matrix.

    Nothing else changes: `prob_entropies_T`, `sorted_by` and the
    generation-keyed cache are inherited, so `EntropyVectorPolicy` selects the
    argmax of I((mu, sigma, r); Y | x) with no modification to the policy.
    """

    def probs_T(self):
        return self._cached(
            "P_T",
            lambda: robit_prob_matrix_T(self.candidates, self.posterior.mu,
                                        self.posterior.sigma, self.posterior.r),
        )
