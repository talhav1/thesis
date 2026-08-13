"""The likelihood-factorisation invariant.

Proposition (Phase 0).  Let H_i = (X_1, Y_1, ..., X_i, Y_i) and suppose the
design policy is *predictable and parameter-free*, i.e.

    X_i | H_{i-1} ~ q_i( . | H_{i-1} )      with q_i not depending on theta.

Then

    p_theta(H_n) = prod_i q_i(x_i | H_{i-1}) p_theta(y_i | x_i)
                 = [ prod_i q_i(x_i | H_{i-1}) ] * prod_i p_theta(y_i | x_i),

and the bracket is constant in theta.  Hence for the observed history

    p_theta(H_n)  ∝_theta  prod_i p_theta(y_i | x_i),

so Rotem's product likelihood yields the *exact* Bayesian posterior under the
assumed response model.  Adaptive dependence among the Y_i does not make it a
pseudo-likelihood.

This module turns that statement into something a machine can check, by
computing both sides for an observed history over a grid of theta.  The
difference must be exactly constant in theta.  `OraclePolicy` supplies the
counterexample: when q_i depends on theta the difference varies, the bracket
no longer factors out, and the product-likelihood posterior is wrong.

Nothing else in the codebase depends on this module; it exists so that the
invariant is enforced by the test suite rather than by a comment.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .policies import PolicyState
from .response_models import probit_loglik
from .rotem_particles import ParticlePosterior


def product_loglik(x, y, mu, sigma):
    """log prod_i p_theta(y_i | x_i), the term Rotem uses."""
    return probit_loglik(np.asarray(x), np.asarray(y),
                         np.atleast_1d(mu), np.atleast_1d(sigma)).sum(axis=-1)


def replay_policy_logprobs(policy, prior, candidates, x, y, n_particles, seed,
                           theta_for_policy=None):
    """Recompute log q_i(x_i | H_{i-1}) for an observed history.

    The policy's internal state is rebuilt step by step from the history, so
    the returned terms are exactly those that appeared when the data were
    generated.  `theta_for_policy` is only consulted by policies that are
    *not* ignorable; ignorable policies ignore it entirely, which is the whole
    point.
    """
    rng = np.random.default_rng(seed)
    posterior = ParticlePosterior(prior, n_particles, rng)
    state = PolicyState(candidates=candidates, posterior=posterior)
    if theta_for_policy is not None and hasattr(policy, "set_theta"):
        policy.set_theta(*theta_for_policy)

    terms = np.empty(len(x))
    for i in range(len(x)):
        terms[i] = policy.log_prob(float(x[i]), state)
        posterior.update(float(x[i]), int(y[i]))
        state.x_hist = list(x[: i + 1])
        state.y_hist = list(y[: i + 1])
        state.step = i + 1
        state.invalidate()
    return terms


def full_history_logprob(policy, prior, candidates, x, y, mu, sigma,
                         n_particles=2000, seed=0):
    """log p_theta(H_n), including the design terms.

    Returns an array over the supplied (mu, sigma) pairs.
    """
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)
    out = np.empty(len(mu))
    for k in range(len(mu)):
        q = replay_policy_logprobs(
            policy, prior, candidates, x, y, n_particles, seed,
            theta_for_policy=(mu[k], sigma[k]),
        )
        out[k] = q.sum() + product_loglik(x, y, mu[k], sigma[k])[0]
    return out


def factorisation_residual(policy, prior, candidates, x, y, mu, sigma,
                           n_particles=2000, seed=0):
    """log p_theta(H_n) - log prod_i p_theta(y_i | x_i), as a function of theta.

    Under an ignorable policy this equals sum_i log q_i(x_i | H_{i-1}) for
    every theta, hence is constant.  Its spread across theta is therefore a
    direct, dimensionless measure of ignorability failure.
    """
    full = full_history_logprob(policy, prior, candidates, x, y, mu, sigma,
                                n_particles=n_particles, seed=seed)
    prod = product_loglik(x, y, np.atleast_1d(mu), np.atleast_1d(sigma))
    return full - prod


def posterior_from_loglik(loglik, log_prior):
    """Normalised posterior weights on a theta grid from any log-likelihood."""
    lp = np.asarray(loglik) + np.asarray(log_prior)
    finite = np.isfinite(lp)
    w = np.zeros_like(lp)
    w[finite] = np.exp(lp[finite] - logsumexp(lp[finite]))
    return w


def total_variation(w1, w2) -> float:
    return 0.5 * float(np.sum(np.abs(np.asarray(w1) - np.asarray(w2))))
