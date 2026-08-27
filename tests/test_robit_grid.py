"""Three-parameter reference posterior: kernel, diagnostics, refinement.

The nesting invariant lives in `test_robit_nesting.py`.  This file checks the
machinery that carries the two-parameter reference's guarantees into three
dimensions -- the exact cell-smoothing kernel, normalisation, the boundary-mass
and refinement diagnostics, and the r-axis reporting -- because `CLAUDE.md`
section 2 invariant 4 makes reference-posterior adequacy a property the project
enforces rather than assumes.
"""

import numpy as np
import pytest

from src.posterior_grid import _trapezoid_cdf
from src.posterior_grid_robit import (
    RobitGridPosterior,
    RobitGridSpec,
    _uniform_sum_cdf,
    build_robit_reference_posterior,
)
from src.robit import R_HEAVY, R_NEAR_PROBIT, R_REFERENCE, robit_prob_matrix_T
from src.robit_design import RobitParticlePosterior


# ---------------------------------------------------------------------------
# The smoothing kernel
# ---------------------------------------------------------------------------


def test_uniform_sum_cdf_degenerates_to_the_two_parameter_kernel():
    """With one side length zero it must *be* `posterior_grid._trapezoid_cdf`.

    This is what makes the three-parameter reference a strict extension: the
    quadrature the two-parameter results were computed with is a special case,
    not an approximation of it.
    """
    d = np.linspace(-3.0, 3.0, 401)
    for a, b in ((1.0, 0.4), (0.4, 1.0), (1.0, 1.0), (2.0, 0.0)):
        got = _uniform_sum_cdf(d, np.full_like(d, a), np.full_like(d, b),
                               np.zeros_like(d))
        want = _trapezoid_cdf(d, np.full_like(d, a), np.full_like(d, b))
        assert np.max(np.abs(got - want)) < 1e-14, (a, b)


def test_uniform_sum_cdf_degenerates_to_a_ramp_and_a_step():
    d = np.linspace(-2.0, 2.0, 201)
    z = np.zeros_like(d)
    ramp = _uniform_sum_cdf(d, np.full_like(d, 2.0), z, z)
    assert np.max(np.abs(ramp - np.clip((d + 1.0) / 2.0, 0, 1))) < 1e-14
    step = _uniform_sum_cdf(d, z, z, z)
    assert np.max(np.abs(step - (d > 0).astype(float))) < 1e-14


def test_uniform_sum_cdf_matches_monte_carlo_in_three_dimensions():
    """The cubic branch, against an independent stochastic evaluation."""
    rng = np.random.default_rng(0)
    h = (1.3, 0.7, 0.35)
    n = 4_000_000
    s = sum(hi * (rng.random(n) - 0.5) for hi in h)
    d = np.linspace(-1.4, 1.4, 29)
    got = _uniform_sum_cdf(d, np.full_like(d, h[0]), np.full_like(d, h[1]),
                           np.full_like(d, h[2]))
    want = np.array([np.mean(s < di) for di in d])
    se = np.sqrt(np.maximum(want * (1 - want), 1e-12) / n)
    assert np.max(np.abs(got - want) / (se + 1e-12)) < 5.0


def test_uniform_sum_cdf_is_monotone_and_bounded():
    """Monotone to round-off, and exactly 0/1 outside the cell.

    The cubic branch is a difference of cubes over a product of three widths,
    so it carries ~1e-14 of cancellation inside the support; the tolerance is a
    floating-point allowance, not a modelling one.  Outside the support both
    tails are taken in closed form, so those two values must be exact.
    """
    d = np.linspace(-4.0, 4.0, 501)
    v = _uniform_sum_cdf(d, np.full_like(d, 1.0), np.full_like(d, 0.6),
                         np.full_like(d, 0.2))
    assert np.all(np.diff(v) >= -1e-13)
    assert v[0] == 0.0 and v[-1] == 1.0


# ---------------------------------------------------------------------------
# The posterior object
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("r_prior", [R_REFERENCE, R_NEAR_PROBIT, R_HEAVY])
def test_weights_normalise(prior, small_history, r_prior):
    x, y = small_history
    post = build_robit_reference_posterior(prior, r_prior, x, y, fixed_n=129,
                                           fixed_n_r=9)
    assert abs(post.w.sum() - 1.0) < 1e-12
    assert np.all(post.w >= 0)
    assert post.spec.n_r == 9


def test_cdf_at_is_the_rank_statistic(prior, small_history):
    x, y = small_history
    post = build_robit_reference_posterior(prior, R_REFERENCE, x, y,
                                           fixed_n=193, fixed_n_r=17)
    med = float(post.quantile("q0.95", 0.5))
    assert abs(post.cdf_at("q0.95", med) - 0.5) < 2e-3
    assert post.cdf_at("q0.95", -1e6) == 0.0
    assert abs(post.cdf_at("q0.95", 1e6) - 1.0) < 1e-12


def test_covered_agrees_with_the_interval_it_reports(prior, small_history):
    """DEC-3: coverage is read off the rank statistic, and must not disagree
    with the interval the same object forms."""
    x, y = small_history
    post = build_robit_reference_posterior(prior, R_REFERENCE, x, y,
                                           fixed_n=193, fixed_n_r=17)
    lo, hi = post.credible_interval("q0.95", 0.95)
    assert post.covered("q0.95", 0.5 * (lo + hi))
    assert not post.covered("q0.95", hi + 5.0)
    assert not post.covered("q0.95", lo - 5.0)


def test_boundary_mass_excludes_the_r_faces(prior, small_history):
    """Mass at r = 0 or r = r_max is a finding, not a too-small box.

    `boundary_mass` must stay the DEC-7 quantity -- (mu, eta) edge mass -- so
    the adequacy convention carries over.  The r edges are reported separately.
    """
    x, y = small_history
    post = build_robit_reference_posterior(prior, R_NEAR_PROBIT, x, y,
                                           fixed_n=193, fixed_n_r=17)
    zero, top = post.r_edge_mass()
    assert zero > 1e-3, "near-probit prior should leave real mass at r = 0"
    assert post.boundary_mass() < 1e-8 < zero
    s = post.r_summary()
    assert 0.0 <= s["r_mean"] <= post.spec.r_max
    assert s["r_prior"] == "near_probit"


def test_refinement_ladder_doubles_all_three_axes(prior, small_history):
    """A reported convergence must be joint, never hiding an unresolved r axis.

    Run at tol = 3e-3 so the fast suite stays fast; the 1e-3 production
    tolerance is exercised in the slow test below.  The r axis is second-order
    (`_r_quadrature_logweights`), so the observed deltas fall by ~4x per
    doubling and the two tolerances differ by one rung, not by a regime.
    """
    x, y = small_history
    post, history = build_robit_reference_posterior(
        prior, R_REFERENCE, x, y, tol=3e-3, return_history=True,
        n_start=65, n_max=257, n_r_start=9, n_r_max=33,
    )
    assert len(history) > 1
    assert post.convergence["final_n_r"] > 9
    assert post.convergence["final_n"] > 65
    assert post.convergence["boundary_mass"] < 1e-8
    assert post.converged, f"robit reference did not converge: {post.convergence}"
    assert post.convergence["max_relative_delta"] < 3e-3


@pytest.mark.slow
def test_refinement_converges_at_the_production_tolerance(prior, small_history):
    x, y = small_history
    post = build_robit_reference_posterior(
        prior, R_REFERENCE, x, y, tol=1e-3,
        n_start=129, n_max=513, n_r_start=17, n_r_max=65,
    )
    assert post.converged, f"robit reference did not converge: {post.convergence}"
    assert post.convergence["max_relative_delta"] < 1e-3


def test_denser_grid_does_not_move_the_answer(prior, small_history):
    """Explicit check against a denser grid, as for the two-parameter case."""
    x, y = small_history
    coarse = build_robit_reference_posterior(prior, R_REFERENCE, x, y,
                                             fixed_n=193, fixed_n_r=17)
    s = coarse.spec
    dense = RobitGridPosterior.build(
        prior, R_REFERENCE, x, y,
        RobitGridSpec(s.mu_lo, s.mu_hi, s.eta_lo, s.eta_hi, 513, 513, 33, s.r_max),
    )
    for q in ("mu", "sigma", "q0.95", "q0.99"):
        scale = dense.sd(q)
        assert abs(coarse.mean(q) - dense.mean(q)) < 5e-3 * scale, q
        assert abs(coarse.sd(q) - dense.sd(q)) < 1e-2 * scale, q


def test_no_data_recovers_the_r_prior(prior):
    """With an empty history the r margin must be the discretised r prior."""
    for rp, want in ((R_REFERENCE, 0.25), (R_NEAR_PROBIT, 0.05), (R_HEAVY, 0.375)):
        post = build_robit_reference_posterior(prior, rp, np.array([]), np.array([]),
                                               fixed_n=129, fixed_n_r=33)
        assert abs(post.r_summary()["r_mean"] - want) < 0.02, rp.name


# ---------------------------------------------------------------------------
# The design side
# ---------------------------------------------------------------------------


def test_robit_probability_matrix_matches_a_direct_evaluation():
    from scipy.stats import t as t_dist

    rng = np.random.default_rng(3)
    x = np.linspace(20.0, 40.0, 11)
    mu = rng.normal(30.0, 3.0, 40)
    sigma = rng.lognormal(np.log(2.4), 0.3, 40)
    r = rng.choice(np.linspace(0.0, 0.5, 9), 40)
    got = robit_prob_matrix_T(x, mu, sigma, r)
    z = (x[:, None] - mu[None, :]) / sigma[None, :]
    want = np.where(r[None, :] > 0,
                    t_dist.cdf(z, 1.0 / np.where(r > 0, r, 1.0)[None, :]),
                    _ndtr(z))
    assert np.max(np.abs(got - want)) < 1e-12


def _ndtr(z):
    from scipy.special import ndtr

    return ndtr(z)


def test_robit_particle_posterior_tracks_the_grid_on_a_short_history(
    prior, small_history
):
    """The design cloud and the reference must agree where both are cheap.

    The design arm is only meaningful if the posterior it steers by is the
    posterior.  N = 40,000 prior particles on an 8-observation history keep the
    effective sample size in the thousands, so agreement to a few percent of a
    posterior standard deviation is the right expectation.
    """
    x, y = small_history
    rng = np.random.default_rng(11)
    cloud = RobitParticlePosterior(prior, R_REFERENCE, 40_000, rng, n_r_nodes=17)
    for xi, yi in zip(x, y):
        cloud.update(float(xi), int(yi))
    grid = build_robit_reference_posterior(prior, R_REFERENCE, x, y,
                                           fixed_n=257, fixed_n_r=17)
    assert cloud.ess() > 1_000
    for q in ("mu", "sigma", "q0.95"):
        assert abs(cloud.mean(q) - grid.mean(q)) < 0.1 * grid.sd(q), q


def test_robit_design_state_drives_the_existing_mi_policy(prior):
    """`EntropyVectorPolicy` must run unchanged over a three-parameter cloud.

    The design arm is meant to be separable: the *only* difference from Rotem's
    design is the parameter vector the mutual information is taken over.  If the
    policy needed modifying, that separability would be a claim rather than a
    fact.
    """
    from src.policies import EntropyVectorPolicy
    from src.priors import rotem_stimulus_grid
    from src.robit_design import RobitPolicyState

    rng = np.random.default_rng(5)
    cand = rotem_stimulus_grid("well", n_points=60)
    cloud = RobitParticlePosterior(prior, R_REFERENCE, 4_000, rng, n_r_nodes=9)
    state = RobitPolicyState(candidates=cand, posterior=cloud)
    x, info = EntropyVectorPolicy().select(state, rng)
    assert x in cand
    assert np.isfinite(info["mi_max"]) and info["mi_max"] > 0
    assert len(info["mi"]) == len(cand)


# ---------------------------------------------------------------------------
# The two hoists in the hot path must be identities, not approximations
# ---------------------------------------------------------------------------


def test_values_tiling_matches_a_direct_per_cell_evaluation(prior, small_history):
    """`values` evaluates t^{-1} on the n_r nodes and tiles.

    The r axis is a shared node set by construction, so this is an identity --
    but it is an identity that depends on the *flattening order* of the grid
    (r fastest), and a change to that order would silently misalign every
    quantile in the posterior while leaving its mean and sd plausible.  The
    direct per-cell evaluation is the definition; this pins the tiling to it.
    """
    x, y = small_history
    post = build_robit_reference_posterior(prior, R_REFERENCE, x, y,
                                           fixed_n=65, fixed_n_r=9)
    from src.robit import robit_z

    for q in ("q0.5", "q0.95", "q0.99"):
        p = float(q[1:])
        direct = post.mu + post.sigma * robit_z(p, post.r)
        assert np.max(np.abs(post.values(q) - direct)) == 0.0, q


def test_cdf_at_matches_the_unplanned_kernel(prior, small_history):
    """`cdf_at` caches the sorted widths; the cache must change no number.

    `_uniform_sum_cdf` remains the entry point the kernel tests exercise, and
    `cdf_at` goes through `_plan` + `_uniform_sum_cdf_planned` instead.  If the
    two ever diverge, every rank statistic in the phase silently moves while the
    kernel tests keep passing.
    """
    x, y = small_history
    post = build_robit_reference_posterior(prior, R_REFERENCE, x, y,
                                           fixed_n=65, fixed_n_r=9)
    d_mu, d_eta, d_r = post._cell_extents()
    for q in ("q0.5", "q0.99"):
        v = post.values(q)
        gmu, geta, gr = post._value_gradient(q)
        widths = (np.abs(gmu) * d_mu, np.abs(geta) * d_eta, np.abs(gr) * d_r)
        for probe in (post.mean(q) - 2 * post.sd(q), post.mean(q),
                      post.mean(q) + 2 * post.sd(q)):
            want = float(np.dot(post.w, _uniform_sum_cdf(probe - v, *widths)))
            assert abs(post.cdf_at(q, probe) - want) < 1e-15, (q, probe)


def test_the_plan_cache_does_not_leak_between_quantities(prior, small_history):
    """Two targets probed in either order must give the same rank statistics."""
    x, y = small_history
    a = build_robit_reference_posterior(prior, R_REFERENCE, x, y,
                                        fixed_n=65, fixed_n_r=9)
    b = build_robit_reference_posterior(prior, R_REFERENCE, x, y,
                                        fixed_n=65, fixed_n_r=9)
    forward = [a.cdf_at(q, 34.0) for q in ("q0.5", "q0.95", "q0.99")]
    backward = [b.cdf_at(q, 34.0) for q in ("q0.99", "q0.95", "q0.5")][::-1]
    assert forward == backward
