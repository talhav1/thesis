"""Required test 1: likelihood proportionality under adaptive designs.

This is the mathematical invariant of the whole project.  It is checked for a
deterministic adaptive policy, a randomised adaptive policy, and a randomised
non-adaptive policy -- and then *violated* on purpose by a non-ignorable
policy, so the test cannot pass vacuously.
"""

import numpy as np
import pytest

from src.factorization import (
    factorisation_residual,
    posterior_from_loglik,
    product_loglik,
    total_variation,
)
from src.policies import (
    EntropyVectorPolicy,
    OraclePolicy,
    PolicyState,
    SoftmaxEntropyPolicy,
    UniformGridPolicy,
)
from src.response_models import ProbitCurve
from src.rotem_particles import ParticlePosterior

N_PARTICLES = 800
N_STEPS = 12


def _simulate(policy, prior, candidates, curve, seed):
    rng = np.random.default_rng(seed)
    posterior = ParticlePosterior(prior, N_PARTICLES, rng)
    state = PolicyState(candidates=candidates, posterior=posterior)
    xs, ys = [], []
    u = np.random.default_rng(seed + 5000).random(N_STEPS)
    for k in range(N_STEPS):
        x, _ = policy.select(state, rng)
        y = int(u[k] < float(np.atleast_1d(curve.prob(np.array([x])))[0]))
        posterior.update(x, y)
        xs.append(x)
        ys.append(y)
        state.x_hist, state.y_hist, state.step = xs, ys, k + 1
        state.invalidate()
    return np.asarray(xs), np.asarray(ys)


@pytest.mark.parametrize(
    "policy_factory,label",
    [
        (lambda: EntropyVectorPolicy(), "deterministic adaptive"),
        (lambda: SoftmaxEntropyPolicy(temperature=0.05), "randomised adaptive"),
        (lambda: UniformGridPolicy(), "randomised non-adaptive"),
    ],
)
def test_product_likelihood_is_proportional_to_full_history(
    prior, candidates, theta_grid, policy_factory, label
):
    """log p_theta(H_n) - log prod_i p_theta(y_i|x_i) must not depend on theta."""
    curve = ProbitCurve(30.0, 3.0)
    policy = policy_factory()
    x, y = _simulate(policy, prior, candidates, curve, seed=11)

    mu, sigma = theta_grid
    resid = factorisation_residual(
        policy_factory(), prior, candidates, x, y, mu, sigma,
        n_particles=N_PARTICLES, seed=11,
    )
    finite = np.isfinite(resid)
    assert finite.all(), f"{label}: policy assigned zero probability to a realised design"
    spread = float(np.ptp(resid))
    assert spread < 1e-9, f"{label}: residual varies by {spread:.3e} across theta"


@pytest.mark.parametrize(
    "policy_factory",
    [lambda: EntropyVectorPolicy(), lambda: SoftmaxEntropyPolicy(temperature=0.05)],
)
def test_posteriors_agree_under_ignorable_policies(prior, candidates, theta_grid,
                                                   policy_factory):
    """Same posterior from the product and the full-history likelihoods."""
    curve = ProbitCurve(30.0, 3.0)
    x, y = _simulate(policy_factory(), prior, candidates, curve, seed=23)
    mu, sigma = theta_grid
    log_prior = prior.logpdf(mu, sigma)

    prod = product_loglik(x, y, mu, sigma)
    full = prod + factorisation_residual(
        policy_factory(), prior, candidates, x, y, mu, sigma,
        n_particles=N_PARTICLES, seed=23,
    )
    w_prod = posterior_from_loglik(prod, log_prior)
    w_full = posterior_from_loglik(full, log_prior)
    assert total_variation(w_prod, w_full) < 1e-12


def test_non_ignorable_policy_breaks_the_invariant(prior, candidates, theta_grid):
    """Negative control: a theta-dependent design must break proportionality.

    Without this the first test would be satisfied by any implementation that
    simply ignored the policy term.
    """
    curve = ProbitCurve(30.0, 3.0)
    oracle = OraclePolicy(curve, p=0.5, sharpness=2.0)
    x, y = _simulate(oracle, prior, candidates, curve, seed=37)

    mu, sigma = theta_grid
    resid = factorisation_residual(
        OraclePolicy(curve, p=0.5, sharpness=2.0), prior, candidates, x, y,
        mu, sigma, n_particles=N_PARTICLES, seed=37,
    )
    spread = float(np.ptp(resid[np.isfinite(resid)]))
    assert spread > 1.0, (
        "the oracle policy should make the design term theta-dependent; "
        f"observed spread {spread:.3e}"
    )

    log_prior = prior.logpdf(mu, sigma)
    w_prod = posterior_from_loglik(product_loglik(x, y, mu, sigma), log_prior)
    w_full = posterior_from_loglik(product_loglik(x, y, mu, sigma) + resid, log_prior)
    assert total_variation(w_prod, w_full) > 0.01, (
        "under a non-ignorable design the two posteriors must differ materially"
    )


def test_sequential_updates_match_full_recomputation(prior, candidates):
    """Required test 2: incremental weights == weights recomputed from scratch."""
    curve = ProbitCurve(30.0, 3.0)
    rng = np.random.default_rng(101)
    posterior = ParticlePosterior(prior, N_PARTICLES, rng, rejuvenate=False)
    mu0, sigma0 = posterior.mu.copy(), posterior.sigma.copy()

    state = PolicyState(candidates=candidates, posterior=posterior)
    policy = EntropyVectorPolicy()
    xs, ys = [], []
    u = np.random.default_rng(202).random(N_STEPS)
    for k in range(N_STEPS):
        x, _ = policy.select(state, rng)
        y = int(u[k] < float(np.atleast_1d(curve.prob(np.array([x])))[0]))
        posterior.update(x, y)
        xs.append(x)
        ys.append(y)
        state.x_hist, state.y_hist = xs, ys
        state.invalidate()

    batch = product_loglik(np.asarray(xs), np.asarray(ys), mu0, sigma0)
    w_batch = np.exp(batch - batch.max())
    w_batch /= w_batch.sum()
    assert np.max(np.abs(w_batch - posterior.w)) < 1e-12
