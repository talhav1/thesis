"""Tests for the detectability layers.

The properties enforced here are the ones a Phase 4 conclusion rests on: that
the fitted tail family *is* the data-generating family, that the probit is
exactly nested in it, that the two inversion routines agree, and that the
evidence integral reproduces the one the reference posterior already computes.
A failure in any of them would make "the data cannot detect the
misspecification" a statement about a bug rather than about the design.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from src import model_check as mc
from src.curve_families import TailPerturbedCurve, build_curve_family
from src.posterior_grid import GridPosterior, GridSpec
from src.priors import rotem_prior
from src.response_models import ProbitCurve


@pytest.fixture
def prior():
    return rotem_prior("well", "independent")


@pytest.fixture
def history():
    """An overlapping design with responses on both sides of the median."""
    rng = np.random.default_rng(11)
    x = np.round(np.linspace(24.0, 40.0, 20), 6).repeat(3)
    p = TailPerturbedCurve(shift95=1.0).prob(x)
    return x, (rng.random(len(x)) < p).astype(int)


# -- the fitted tail family reproduces the data-generating one ---------------


@pytest.mark.parametrize("shift", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("m,s", [(30.0, 3.0), (25.0, 5.0), (34.0, 1.5)])
def test_tail_family_matches_the_dgp(shift, m, s):
    """The fitted family is the DGP, at every location and scale.

    `curve_families.TailPerturbedCurve` inverts a table keyed on (mu0, sigma0);
    `model_check` inverts the standardised curve by bisection.  They are
    independent implementations of the same function, so agreement is evidence
    that neither has a sign or scale error.
    """
    x = np.linspace(m - 4 * s, m + 6 * s, 2000)
    log_p, log_q = mc._tail_logprobs(x, m, s, shift)
    expected = TailPerturbedCurve(shift95=shift, mu0=m, sigma0=s).prob(x)
    assert np.max(np.abs(np.exp(log_p) - expected)) < 1e-7
    assert np.allclose(np.exp(log_p) + np.exp(log_q), 1.0, atol=1e-12)


def test_probit_is_exactly_nested_at_zero_shift():
    """Not "close to": bit-for-bit, so the LR is a genuine 1-df statistic."""
    x = np.linspace(18.0, 46.0, 500)
    assert np.array_equal(mc._tail_logprobs(x, 30.0, 3.0, 0.0)[0],
                          mc._probit_logprobs(x, 30.0, 3.0)[0])


def test_tail_inversion_routines_agree():
    """The exact bisection and the interpolation table describe one function.

    `_tail_invert` switches between them on array size, so a discrepancy would
    silently make a Bayes factor and a likelihood ratio refer to different
    alternatives.
    """
    kappa = mc.shift_to_kappa(1.0)
    t = np.linspace(-4.0, 9.0, 1501)
    exact = mc._tail_invert_exact(t, kappa, mc.TAIL_P0)
    u0 = float(norm.ppf(mc.TAIL_P0))
    g_nodes, u_nodes = mc._tail_inverse_table(kappa, mc.TAIL_P0)
    table = np.where(t <= u0, t,
                     np.where(t < mc._U_SATURATED + kappa,
                              np.interp(t, g_nodes, u_nodes), t - kappa))
    assert np.max(np.abs(norm.cdf(exact) - norm.cdf(table))) < 1e-7


def test_exact_inversion_inverts():
    kappa = mc.shift_to_kappa(1.5)
    u = np.linspace(-3.0, 8.0, 400)
    t = mc._g(u, kappa, mc.TAIL_P0)
    assert np.max(np.abs(mc._tail_invert_exact(t, kappa, mc.TAIL_P0) - u)) < 1e-10


@pytest.mark.parametrize("family", [mc.PROBIT, mc.LOGISTIC, mc.CLOGLOG, mc.ROBIT])
def test_families_are_matched_on_median_and_slope(family):
    """Every fitted family has median m and the probit-(m, s) slope there.

    This is what makes one prior over (m, s) meaningful for all of them, and so
    what makes the Bayes factors comparisons of shape alone.
    """
    m, s, h = 31.0, 2.5, 1e-4
    p = lambda z: np.exp(family.logprobs(np.asarray(z, dtype=float), m, s, None)[0])
    assert abs(float(p(m)) - 0.5) < 1e-9
    slope = float(p(m + h) - p(m - h)) / (2 * h)
    assert abs(slope - norm.pdf(0.0) / s) < 1e-6


@pytest.mark.parametrize("family", [mc.PROBIT, mc.LOGISTIC, mc.CLOGLOG, mc.ROBIT,
                                    mc.tail_fixed(1.0)])
def test_log_probabilities_are_finite_and_complementary(family):
    x = np.linspace(-40.0, 120.0, 4000)
    log_p, log_q = family.logprobs(x, 30.0, 3.0, None)
    assert np.all(np.isfinite(log_p)) and np.all(np.isfinite(log_q))
    assert np.all(np.diff(log_p) >= -1e-12)          # monotone increasing
    both = np.exp(log_p) + np.exp(log_q)
    assert np.max(np.abs(both - 1.0)) < 1e-9


# -- likelihood, evidence, and the tests built on them ----------------------


def test_collapse_preserves_the_loglikelihood(history):
    x, y = history
    xu, n1, n0 = mc.collapse(x, y)
    collapsed = float(mc.loglik(mc.PROBIT, xu, n1, n0, 30.0, 3.0)[0])
    z = (x - 30.0) / 3.0
    direct = float(np.sum(np.where(y > 0.5, norm.logcdf(z), norm.logcdf(-z))))
    assert collapsed == pytest.approx(direct, abs=1e-10)


def test_evidence_agrees_with_the_reference_posterior(prior, history):
    """`log_evidence` for the probit must reproduce `GridPosterior.log_evidence`.

    Same integral, two code paths; the alternatives are only trustworthy
    because the model they are compared against is computed the same way the
    rest of the project computes it.
    """
    x, y = history
    spec = mc.shared_box(prior, x, y)
    mine, edge = mc.log_evidence(mc.PROBIT, prior, x, y, spec)
    theirs = GridPosterior.build(prior, x, y, spec).log_evidence
    assert mine == pytest.approx(theirs, abs=1e-9)
    assert edge < 1e-8


def test_shared_box_localises_rather_than_inflating(prior, history):
    """Localisation must converge, not run away.

    Iterating `_shrink_to_mass` a fixed number of times pads by 25% each pass
    and diverges on a coarse grid, producing a box wide enough to make every
    Bayes factor meaningless while still looking plausible.  The bound here is
    the prior box: whatever else localisation does, it must not return
    something larger than what it started from.
    """
    from src.posterior_grid import initial_spec

    x, y = history
    prior_box = initial_spec(prior, n_mu=129, n_eta=129)
    spec = mc.shared_box(prior, x, y)
    assert spec.mu_hi - spec.mu_lo < prior_box.mu_hi - prior_box.mu_lo
    assert spec.eta_hi - spec.eta_lo < prior_box.eta_hi - prior_box.eta_lo


def test_bayes_factors_are_converged_and_box_invariant(prior, history):
    """A Bayes factor must be a property of the models, not of the quadrature.

    Both knobs are checked: refining the grid and enlarging the shared box must
    leave the answer alone.  Without this the evidence layer could report a
    difference of 0.2 nats that is really a discretisation artefact -- which at
    the magnitudes seen in Phase 4 would be the entire signal.
    """
    from src.posterior_grid import GridSpec

    x, y = history
    oracle = mc.tail_fixed(1.0)

    def bf(pad, n):
        b = mc.shared_box(prior, x, y, pad=pad)
        s = GridSpec(b.mu_lo, b.mu_hi, b.eta_lo, b.eta_hi, n, n)
        return (mc.log_evidence(oracle, prior, x, y, s)[0]
                - mc.log_evidence(mc.PROBIT, prior, x, y, s)[0])

    base = bf(0.25, 385)
    assert abs(bf(0.25, 769) - base) < 1e-4          # grid refinement
    assert abs(bf(1.0, 385) - base) < 1e-4           # box enlargement


def test_lr_is_nonnegative_and_zero_when_the_probit_fits(prior, history):
    rng = np.random.default_rng(3)
    x = np.linspace(24.0, 36.0, 12).repeat(4)
    y = (rng.random(len(x)) < ProbitCurve(30.0, 3.0).prob(x)).astype(int)
    out = mc.lr_statistic(x, y)
    assert out["lr"] >= 0.0
    assert out["lr_raw"] > -1e-6            # no optimiser regression past probit
    assert 0.0 <= out["lr_shape_hat"] <= 4.0


def test_lr_has_power_when_the_design_reaches_the_perturbation():
    """The oracle test is not inert: given data where the curves differ, it fires.

    Without this, "the LR never rejects" could not be distinguished from "the
    LR is broken".  The design here spans the median *and* reaches well past
    the perturbation onset, with 400 trials -- eight times the Phase 3 budget,
    spread far wider than any Phase 3 policy spreads it.

    Both halves are needed.  Restricted to x >= 33.5 the same test has no power
    at all, because with nothing near the median the probit is free to re-fit
    its location and scale and absorb the perturbation.  Reaching the tail is
    necessary for detection; it is not sufficient.
    """
    rng = np.random.default_rng(5)
    x = np.linspace(24.0, 44.0, 20).repeat(20)
    y = (rng.random(len(x)) < TailPerturbedCurve(shift95=1.0).prob(x)).astype(int)
    out = mc.lr_statistic(x, y)
    assert out["lr"] > 6.0
    assert out["lr_shape_hat"] > 0.2


def test_reaching_the_tail_alone_does_not_confer_power():
    """The companion to the test above, and a caution against a tempting fix.

    A design placed *entirely* above the perturbation onset gives the oracle
    test nothing: the probit re-fits (m, s) to the perturbed curve over that
    range and the likelihood ratio collapses.  So "explore higher" is not by
    itself a safeguard -- what a safeguard has to preserve is the contrast
    between the region that pins the location-scale and the region that carries
    the estimand.
    """
    rng = np.random.default_rng(5)
    x = np.linspace(33.5, 44.0, 20).repeat(20)
    y = (rng.random(len(x)) < TailPerturbedCurve(shift95=1.0).prob(x)).astype(int)
    assert mc.lr_statistic(x, y)["lr"] < 1.0


# -- what the design could reveal at all ------------------------------------


def test_design_information_is_zero_under_the_reference_curve():
    ref = ProbitCurve(30.0, 3.0)
    info = mc.design_information(ref, ref, np.linspace(20.0, 40.0, 50))
    assert info["design_kl_nats"] == pytest.approx(0.0, abs=1e-12)
    assert info["expected_flips"] == pytest.approx(0.0, abs=1e-12)
    assert info["max_power_bound"] == pytest.approx(0.05)


def test_design_information_grows_with_reach_into_the_tail():
    """The whole mechanism in one assertion.

    A design confined below the perturbation onset carries essentially no
    information about it, however many trials it spends; one that reaches above
    carries some.  The failure Phase 3 found is a statement about *where* the
    budget goes, not about how large it is.
    """
    curve = TailPerturbedCurve(shift95=1.0)
    ref = ProbitCurve(30.0, 3.0)
    below = mc.design_information(curve, ref, np.linspace(24.0, 33.0, 50))
    above = mc.design_information(curve, ref, np.linspace(33.5, 43.0, 50))
    assert below["design_kl_nats"] < 1e-3
    assert below["expected_flips"] < 0.05
    assert above["design_kl_nats"] > 10 * max(below["design_kl_nats"], 1e-12)
    assert above["expected_flips"] > 1.0


def test_realized_flips_counts_the_counterfactual():
    """`realized_flips` must equal the responses that actually change.

    It is the sharpest form of the claim -- that the two hypotheses generated
    the same data -- so it is checked against the simulator's own rule
    y = 1{u < p} rather than against a formula.
    """
    rng = np.random.default_rng(17)
    x = np.linspace(30.0, 44.0, 200)
    u = rng.random(len(x))
    curve, ref = TailPerturbedCurve(shift95=1.0), ProbitCurve(30.0, 3.0)
    direct = int(np.sum((u < curve.prob(x)).astype(int)
                        != (u < ref.prob(x)).astype(int)))
    assert mc.design_information(curve, ref, x, u)["realized_flips"] == direct


def test_robit_dgp_is_visible_where_the_tail_perturbation_is_not():
    """The families differ in *where* they differ, and that is the whole story.

    A matched robit departs from the probit over the region an adaptive design
    actually visits; the tail perturbation is identically probit there.  This
    is the ordering Phase 3 observed in coverage, established here directly on
    the curves and independent of any inference.
    """
    x = np.linspace(26.0, 33.0, 50)             # the visited region
    ref = ProbitCurve(30.0, 3.0)
    robit = mc.design_information(build_curve_family("robit", 1.0), ref, x)
    tail = mc.design_information(TailPerturbedCurve(shift95=1.0), ref, x)
    assert tail["design_kl_nats"] < 1e-6
    assert robit["design_kl_nats"] > 100 * tail["design_kl_nats"]


# -- posterior predictive check ---------------------------------------------


def test_ppc_is_calibrated_under_the_correct_model(prior):
    """Predictive p-values must not reject the truth at anything like alpha.

    Loose on purpose: posterior predictive p-values are known to be
    conservative, and the experiment calibrates them against a null cell rather
    than trusting nominal levels.  What this rules out is a p-value that is
    upside down or degenerate.
    """
    rng = np.random.default_rng(23)
    ps = []
    for r in range(12):
        x = np.linspace(25.0, 35.0, 15).repeat(3)
        y = (rng.random(len(x)) < ProbitCurve(30.0, 3.0).prob(x)).astype(int)
        out = mc.posterior_predictive_check(prior, x, y, rng, n_draws=100,
                                            fixed_n=129)
        ps.append(out["ppp_chi2"])
        assert 0.0 <= out["ppp_tail"] <= 1.0
    assert np.mean(np.asarray(ps) < 0.05) <= 0.25


def test_ppc_tail_statistic_uses_the_upper_stimuli(prior):
    rng = np.random.default_rng(29)
    x = np.linspace(25.0, 40.0, 20).repeat(2)
    y = (rng.random(len(x)) < ProbitCurve(30.0, 3.0).prob(x)).astype(int)
    out = mc.posterior_predictive_check(prior, x, y, rng, n_draws=50,
                                        fixed_n=129, tail_frac=0.25)
    assert out["ppc_n_distinct"] == 20
    assert out["ppc_n_tail_bins"] == 5


def test_ppc_p_values_are_not_degenerate_under_ties(prior):
    """The mid-p convention must keep the null p-value off the boundary.

    With every stimulus in the upper bins saturated at p ~ 1, the replicated
    and observed discrepancies are equal on most draws.  Counting those ties
    whole sent `ppp_tail` to exactly 0 on a majority of *correct-model*
    replicates -- a check that rejects the truth three times in four.
    """
    rng = np.random.default_rng(41)
    zeros = 0
    for _ in range(10):
        x = np.linspace(28.0, 44.0, 16).repeat(3)     # upper half saturated
        y = (rng.random(len(x)) < ProbitCurve(30.0, 3.0).prob(x)).astype(int)
        out = mc.posterior_predictive_check(prior, x, y, rng, n_draws=100,
                                            fixed_n=129)
        assert 0.0 <= out["ppp_tail"] <= 1.0
        zeros += out["ppp_tail"] == 0.0
    assert zeros <= 2


def test_signal_split_partitions_a_cell_on_collected_signal():
    """`signal_split` feeds a manuscript table, so its arithmetic is pinned here.

    The split is on `realized_flips == 0` -- runs whose responses are
    bit-identical to what the matched probit would have produced -- and the two
    parts must partition the cell exactly, with coverage read off the same
    column the rest of the experiment uses.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    from phase4_undetectability import TARGET, signal_split

    import pandas as pd

    g = pd.DataFrame({
        "realized_flips": [0, 0, 0, 1, 2],
        f"{TARGET}_ref_covered": [False, False, True, True, True],
        f"{TARGET}_err": [-6.0, -6.0, -3.0, -2.0, -1.0],
    })
    s = signal_split(g)
    assert s["n_blind"] + s["n_informed"] == len(g)
    assert (s["n_blind"], s["n_informed"]) == (3, 2)
    assert s["coverage_blind"] == pytest.approx(1 / 3)
    assert s["coverage_informed"] == pytest.approx(1.0)
    assert s["bias_blind"] == pytest.approx(-5.0)
    assert s["bias_informed"] == pytest.approx(-1.5)


def test_signal_split_survives_an_empty_side():
    """A cell where every run is blind must not raise or fabricate a number."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    from phase4_undetectability import TARGET, signal_split

    import pandas as pd

    g = pd.DataFrame({"realized_flips": [0, 0],
                      f"{TARGET}_ref_covered": [False, True],
                      f"{TARGET}_err": [-5.0, -3.0]})
    s = signal_split(g)
    assert s["n_informed"] == 0
    assert np.isnan(s["coverage_informed"]) and np.isnan(s["bias_informed"])
    assert s["coverage_blind"] == pytest.approx(0.5)
