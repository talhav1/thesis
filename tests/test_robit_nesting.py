"""Hard invariant: the robit family contains the probit exactly at r = 0.

`CLAUDE.md` section 2 governs this file.  These are properties of the
mathematics, not of the implementation.  If one fails the implementation is
wrong, and it is **never** to be repaired by relaxing a tolerance, marking a
test `xfail`, or narrowing its scope.

The claim is the whole reason r = 1/nu on [0, 0.5] was chosen over nu or
log nu: r = 0 is a *reachable* point of the parameter space at which the fitted
family is the probit, not a limit approached at infinity.  Two consequences are
enforced here.

1.  **Link nesting.**  ``T_nu((x-mu)/sigma)`` at r = 0 is ``Phi((x-mu)/sigma)``
    to floating point, in the probability, in the log-likelihood and in the
    quantile map.
2.  **Posterior nesting.**  With the prior on r degenerate at 0, the
    three-parameter reference posterior *is* the two-parameter one on the same
    history -- to < 1e-8 in mean, sd and rank statistic, for every target.

Invariant 2 is stated on a **shared box**.  The two builders localise
independently, so comparing their default outputs would measure two
localisation passes rather than two models; localising once and handing the
same `GridSpec` to both isolates the only thing in question, which is whether
the extra dimension changes the answer when it carries no mass.  With a
degenerate r prior the r-axis has a single node of zero width, so the extra
dimension collapses exactly: the agreement below is at machine precision, and
the 1e-8 tolerance is slack of eight orders of magnitude, not a fitted budget.
"""

import numpy as np
import pytest

from src.posterior_grid import GridPosterior, GridSpec, build_reference_posterior
from src.posterior_grid_robit import (
    RobitGridPosterior,
    RobitGridSpec,
    build_robit_reference_posterior,
)
from src.priors import rotem_prior
from src.response_models import probit_loglik, probit_prob, quantile_from_params
from src.robit import (
    R_DEGENERATE,
    R_HEAVY,
    R_NEAR_PROBIT,
    R_REFERENCE,
    collapse,
    robit_aggregated_loglik,
    robit_prob,
    robit_z,
)

TARGETS = ("mu", "sigma", "q0.5", "q0.95", "q0.99")
TOL = 1e-8


# ---------------------------------------------------------------------------
# 1. The link
# ---------------------------------------------------------------------------


def test_link_at_r_zero_is_exactly_the_probit():
    x = np.linspace(10.0, 50.0, 401)
    p_robit = robit_prob(x, 30.0, 3.0, 0.0)
    p_probit = probit_prob(x, 30.0, 3.0)
    assert np.max(np.abs(p_robit - p_probit)) == 0.0


def test_quantile_map_at_r_zero_is_exactly_the_probit():
    for p in (0.5, 0.95, 0.99, 0.01):
        assert robit_z(p, np.array([0.0]))[0] == pytest.approx(
            quantile_from_params(0.0, 1.0, p), abs=0.0, rel=0.0
        )


def test_aggregated_loglik_at_r_zero_matches_the_probit_loglik(small_history):
    x, y = small_history
    xu, n1, n0 = collapse(x, y)
    mu = np.array([28.0, 30.0, 32.0, 31.5])
    sigma = np.array([2.0, 3.0, 4.5, 1.2])
    got = robit_aggregated_loglik(xu, n1, n0, mu, sigma, np.array([0.0]))[:, 0]
    want = probit_loglik(x, y, mu, sigma).sum(axis=-1)
    assert np.max(np.abs(got - want)) < 1e-12


def test_the_tail_parameter_actually_moves_the_upper_quantile():
    """The negative control for the nesting tests.

    Nesting alone is satisfied by a family that ignores r.  What makes r worth
    carrying is that it moves q_0.99 a great deal and q_0.5 not at all, which
    is the entire mechanism the safeguard rests on.
    """
    r = np.array([0.0, 0.1, 0.25, 0.5])
    z99 = robit_z(0.99, r)
    assert np.all(np.diff(z99) > 0.2), z99
    assert z99[-1] > 2.5 * z99[0]
    assert np.max(np.abs(robit_z(0.5, r))) < 1e-12


# ---------------------------------------------------------------------------
# 2. The posterior -- the hard invariant
# ---------------------------------------------------------------------------


def _shared_specs(prior, x, y, n=193):
    """One localisation pass, handed to both builders."""
    loc = build_reference_posterior(prior, x, y, fixed_n=n)
    s = loc.spec
    spec2 = GridSpec(s.mu_lo, s.mu_hi, s.eta_lo, s.eta_hi, n, n)
    spec3 = RobitGridSpec(s.mu_lo, s.mu_hi, s.eta_lo, s.eta_hi, n, n, 1)
    return spec2, spec3


@pytest.mark.parametrize("target", TARGETS)
def test_degenerate_r_prior_reproduces_the_two_parameter_posterior(
    prior, small_history, target
):
    x, y = small_history
    spec2, spec3 = _shared_specs(prior, x, y)
    two = GridPosterior.build(prior, x, y, spec2)
    three = RobitGridPosterior.build(prior, R_DEGENERATE, x, y, spec3)

    scale = max(two.sd(target), 1e-12)
    assert abs(three.mean(target) - two.mean(target)) < TOL * scale
    assert abs(three.sd(target) - two.sd(target)) < TOL * scale

    # the rank statistic, at three probes spanning the posterior
    for probe in (two.mean(target) - 1.5 * scale, two.mean(target),
                  two.mean(target) + 1.5 * scale):
        assert abs(three.cdf_at(target, probe) - two.cdf_at(target, probe)) < TOL


def test_degenerate_r_prior_reproduces_the_weights_and_evidence(prior, small_history):
    x, y = small_history
    spec2, spec3 = _shared_specs(prior, x, y)
    two = GridPosterior.build(prior, x, y, spec2)
    three = RobitGridPosterior.build(prior, R_DEGENERATE, x, y, spec3)
    assert np.max(np.abs(three.w - two.w)) < 1e-14
    assert abs(three.log_evidence - two.log_evidence) < 1e-10


def test_nesting_holds_on_an_adaptive_history(prior):
    """The invariant must hold on the histories the experiment actually makes.

    A hand-written history with distinct stimuli is the easy case.  An adaptive
    design revisits a handful of grid points many times, which is the case the
    collapse-to-counts path and the saturated-probability branches exercise.
    """
    x = np.array([30.0] * 9 + [32.2] * 7 + [27.8] * 6 + [33.5] * 3 + [26.0] * 2)
    y = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1] + [1] * 7 + [0] * 6 + [1, 1, 1] + [0, 0])
    spec2, spec3 = _shared_specs(prior, x, y)
    two = GridPosterior.build(prior, x, y, spec2)
    three = RobitGridPosterior.build(prior, R_DEGENERATE, x, y, spec3)
    for t in TARGETS:
        scale = max(two.sd(t), 1e-12)
        assert abs(three.mean(t) - two.mean(t)) < TOL * scale, t
        assert abs(three.sd(t) - two.sd(t)) < TOL * scale, t
        assert abs(three.cdf_at(t, two.mean(t)) - two.cdf_at(t, two.mean(t))) < TOL, t


def test_nesting_survives_the_full_builder(prior, small_history):
    """End to end, through localisation and the production fixed-n path.

    Weaker than the shared-box invariant above -- the two builders localise
    independently, so this can only be a grid-resolution agreement -- but it is
    the path the experiment calls, and a mistake in the driver rather than in
    the kernel would show up only here.
    """
    x, y = small_history
    two = build_reference_posterior(prior, x, y, fixed_n=257)
    three = build_robit_reference_posterior(prior, R_DEGENERATE, x, y, fixed_n=257)
    assert three.spec.n_r == 1
    for t in TARGETS:
        scale = max(two.sd(t), 1e-12)
        assert abs(three.mean(t) - two.mean(t)) < 1e-3 * scale, t
        assert abs(three.sd(t) - two.sd(t)) < 1e-3 * scale, t


# ---------------------------------------------------------------------------
# 3. The invariant must have content
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("r_prior", [R_REFERENCE, R_NEAR_PROBIT, R_HEAVY])
def test_a_non_degenerate_r_prior_does_not_reproduce_the_probit(
    prior, small_history, r_prior
):
    """The negative control, in the sense of `CLAUDE.md` invariant 2.

    A three-parameter implementation that quietly ignored r would pass every
    nesting test above vacuously.  With real prior mass on r > 0 the q_0.99
    posterior must move, and by a wide margin -- that movement *is* the
    safeguard.  Any change that makes this test pass is a regression.
    """
    x, y = small_history
    spec2, spec3 = _shared_specs(prior, x, y)
    spec3 = RobitGridSpec(spec3.mu_lo, spec3.mu_hi, spec3.eta_lo, spec3.eta_hi,
                          spec3.n_mu, spec3.n_eta, 17)
    two = GridPosterior.build(prior, x, y, spec2)
    three = RobitGridPosterior.build(prior, r_prior, x, y, spec3)

    assert three.sd("q0.99") > 1.05 * two.sd("q0.99"), (
        f"{r_prior.name}: robit q0.99 sd {three.sd('q0.99'):.4f} did not exceed "
        f"probit {two.sd('q0.99'):.4f}"
    )
    # q_0.5 is r-free by construction, so it must be essentially untouched:
    # the widening is a tail effect, not a global loss of precision.
    assert abs(three.mean("q0.5") - two.mean("q0.5")) < 0.35 * two.sd("q0.5")
