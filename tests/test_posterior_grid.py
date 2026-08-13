"""Required tests 5 and 6: reference-posterior normalisation, refinement and recovery."""

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import lognorm, norm

from src.posterior_grid import (
    GridPosterior,
    GridSpec,
    build_reference_posterior,
    initial_spec,
    weighted_quantile,
)
from src.priors import rotem_prior
from src.response_models import probit_loglik


def test_weights_normalise(prior, small_history):
    x, y = small_history
    post = build_reference_posterior(prior, x, y, tol=1e-4, n_max=513)
    assert abs(post.w.sum() - 1.0) < 1e-12
    assert np.all(post.w >= 0)


def test_refinement_converges_and_reports_evidence(prior, small_history):
    x, y = small_history
    post, history = build_reference_posterior(prior, x, y, return_history=True)
    assert post.converged, f"reference did not converge: {post.convergence}"
    assert post.convergence["boundary_mass"] < 1e-9
    # the tracked summaries must have stopped moving
    assert post.convergence["max_relative_delta"] < 1e-3
    # and the log evidence must be stable across the last doubling
    assert abs(history[-1]["log_evidence"] - history[-2]["log_evidence"]) < 1e-3


@pytest.mark.slow
def test_high_accuracy_reference_converges(prior, small_history):
    """The audit setting: a tenth of the default tolerance, on a denser grid."""
    x, y = small_history
    post = build_reference_posterior(prior, x, y, tol=1e-4, n_max=2049)
    assert post.converged, f"high-accuracy reference did not converge: {post.convergence}"
    assert post.convergence["max_relative_delta"] < 1e-4


def test_denser_grid_does_not_move_the_answer(prior, small_history):
    """Explicit check against a much denser grid, as the plan requires."""
    x, y = small_history
    coarse = build_reference_posterior(prior, x, y, tol=1e-3, n_max=513)
    dense = GridPosterior.build(
        prior, x, y,
        GridSpec(coarse.spec.mu_lo, coarse.spec.mu_hi,
                 coarse.spec.eta_lo, coarse.spec.eta_hi, 1537, 1537),
    )
    for q in ("mu", "sigma", "q0.95", "q0.99"):
        scale = dense.sd(q)
        assert abs(coarse.mean(q) - dense.mean(q)) < 1e-3 * scale
        assert abs(coarse.quantile(q, 0.975) - dense.quantile(q, 0.975)) < 5e-3 * scale


@pytest.mark.parametrize("dependence", ["independent", "dependent"])
def test_no_data_recovers_the_prior(dependence):
    """With an empty history the 'posterior' must be the prior."""
    prior = rotem_prior("well", dependence)
    post = build_reference_posterior(prior, np.array([]), np.array([]),
                                     tol=1e-5, n_max=1025)
    rng = np.random.default_rng(0)
    mu_s, sigma_s = prior.sample(400_000, rng)

    assert abs(post.mean("mu") - mu_s.mean()) < 6 * mu_s.std() / np.sqrt(400_000)
    assert abs(post.sd("mu") - mu_s.std()) < 0.02 * mu_s.std()
    # sigma is heavy tailed under the log-normal prior: compare on the log scale
    log_sd_grid = float(np.dot(post.w, np.log(post.sigma)))
    assert abs(log_sd_grid - np.log(sigma_s).mean()) < 0.01


def test_independent_prior_marginals_match_closed_form():
    """Analytic check of the prior densities the grid integrates."""
    prior = rotem_prior("well", "independent")
    post = build_reference_posterior(prior, np.array([]), np.array([]),
                                     tol=1e-5, n_max=1025)
    s = prior.spec
    trunc = 1.0 - norm.cdf(0.0, s.mu0, s.sigma_mu)
    mu_mean = s.mu0 + s.sigma_mu * norm.pdf(s.mu0 / s.sigma_mu) / trunc
    assert abs(post.mean("mu") - mu_mean) < 1e-3

    sigma_median = np.exp(s.sigma_log_location)
    assert abs(post.quantile("sigma", 0.5) - sigma_median) < 1e-2


def test_posterior_matches_brute_force_importance_sampling(prior, small_history):
    """Required test 6: brute-force recovery by an independent method.

    Prior-importance sampling shares no code path with the tensor grid: it is
    stochastic, needs no box, and is unbiased for every posterior expectation.
    With 4e6 draws the effective sample size here is ~1e6, giving Monte Carlo
    standard errors around 0.002-0.004, so agreement inside 4 standard errors
    is a sharp check.

    `scipy.integrate.dblquad` was tried first and rejected: on this integrand
    it returns answers that differ by 2.5e-3 in log evidence depending on the
    integration box while reporting a relative error estimate of 3e-11, i.e.
    its own error estimate is not trustworthy here.  The grid agrees with
    importance sampling; dblquad disagrees with itself.
    """
    x, y = small_history
    post = build_reference_posterior(prior, x, y, tol=1e-4, n_max=1025)

    rng = np.random.default_rng(1)
    n_draws = 4_000_000
    mu_s, sigma_s = prior.sample(n_draws, rng)
    ll = probit_loglik(x, y, mu_s, sigma_s).sum(axis=-1)
    w = np.exp(ll - ll.max())
    w /= w.sum()
    ess = 1.0 / np.sum(w**2)

    for q, vals in (("mu", mu_s), ("sigma", sigma_s)):
        est = float(w @ vals)
        se = float(np.sqrt(w @ (vals - est) ** 2 / ess))
        assert abs(est - post.mean(q)) < 4 * se, (
            f"{q}: grid {post.mean(q):.6f} vs IS {est:.6f} +/- {se:.6f}"
        )

    logZ_is = ll.max() + np.log(np.mean(np.exp(ll - ll.max())))
    assert abs(post.log_evidence - logZ_is) < 5e-3


def test_weighted_quantile_is_monotone_and_exact_on_atoms():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([0.25, 0.25, 0.25, 0.25])
    qs = weighted_quantile(v, w, [0.125, 0.375, 0.625, 0.875])
    assert np.allclose(qs, [1.0, 2.0, 3.0, 4.0])
    assert np.all(np.diff(weighted_quantile(v, w, np.linspace(0.01, 0.99, 50))) >= 0)


def test_cdf_at_is_the_rank_statistic(prior, small_history):
    x, y = small_history
    post = build_reference_posterior(prior, x, y, tol=1e-4, n_max=513)
    med = float(post.quantile("q0.95", 0.5))
    assert abs(post.cdf_at("q0.95", med) - 0.5) < 1e-3
    assert post.cdf_at("q0.95", -1e6) == 0.0
    assert abs(post.cdf_at("q0.95", 1e6) - 1.0) < 1e-12
