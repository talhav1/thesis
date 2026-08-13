"""Phase 3 pre-conditions: every DGP must be monotone, matched and invertible.

The protocol requires these checks to pass before any Phase 3 experiment runs.
A DGP that is not monotone is not a sensitivity curve; one that is not matched
on median and slope confounds shape misspecification with rescaling; one whose
quantiles are wrong makes every estimand wrong.
"""

import numpy as np
import pytest

from src.curve_families import (
    BASE_FAMILIES,
    MatchedMixtureCurve,
    TailPerturbedCurve,
    build_curve_family,
    curve_true_sd,
    match_median_and_slope,
    probit_slope_at_median,
    verify_curve,
)
from src.pseudo_truth import (
    empirical_design_distribution,
    kl_projection,
    pseudo_true_quantiles,
)
from src.response_models import BetaCDFCurve, ProbitCurve
from src.simulator import build_curve, true_targets

REF_MU, REF_SIGMA = 30.0, 3.0
LAMBDAS = (0.25, 0.5, 0.75, 1.0)


def all_curves():
    out = [("probit", build_curve_family("probit", 0.0))]
    for fam in BASE_FAMILIES:
        for lam in LAMBDAS:
            out.append((f"{fam}_{lam}", build_curve_family(fam, lam)))
    for s in (0.25, 0.5, 1.0):
        out.append((f"tail_{s}", TailPerturbedCurve(shift95=s)))
    return out


@pytest.mark.parametrize("name,curve", all_curves())
def test_every_dgp_is_monotone_and_invertible(name, curve):
    r = verify_curve(curve)
    assert r["monotone"], f"{name}: min increment {r['min_diff']:.3e}"
    assert r["spans_unit_interval"], f"{name}: range [{r['p_min']}, {r['p_max']}]"
    assert r["quantiles_consistent"], (
        f"{name}: quantile inversion error {r['max_quantile_inversion_error']:.3e}"
    )


@pytest.mark.parametrize("name,curve", all_curves())
def test_median_and_slope_are_matched_at_every_severity(name, curve):
    """The whole point of the construction: only the shape varies."""
    r = verify_curve(curve)
    assert abs(r["median"] - REF_MU) < 1e-6, f"{name}: median {r['median']}"
    assert abs(r["slope_at_median"] - probit_slope_at_median(REF_SIGMA)) < 1e-6, (
        f"{name}: slope {r['slope_at_median']}"
    )


def test_lambda_zero_is_exactly_the_probit():
    ref = ProbitCurve(REF_MU, REF_SIGMA)
    xs = np.linspace(10, 50, 401)
    for fam in BASE_FAMILIES:
        c = build_curve_family(fam, 0.0)
        assert np.max(np.abs(c.prob(xs) - ref.prob(xs))) < 1e-15, fam


def test_severity_is_monotone_in_lambda():
    """Divergence from the probit must grow with lambda, or 'severity' is a lie."""
    ref = ProbitCurve(REF_MU, REF_SIGMA)
    xs = np.linspace(15, 45, 601)
    for fam in BASE_FAMILIES:
        gaps = [np.max(np.abs(build_curve_family(fam, lam).prob(xs) - ref.prob(xs)))
                for lam in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert np.all(np.diff(gaps) > -1e-12), f"{fam}: {gaps}"
        assert gaps[-1] > gaps[0], fam


def test_tail_perturbation_is_invisible_below_p0():
    """It must be *exactly* probit over the region the design visits."""
    ref = ProbitCurve(REF_MU, REF_SIGMA)
    for s in (0.5, 1.0):
        c = TailPerturbedCurve(shift95=s)
        x_below = np.linspace(15, float(c.quantile(0.85)) - 1e-6, 500)
        assert np.max(np.abs(c.prob(x_below) - ref.prob(x_below))) < 1e-12, s
        # and it must genuinely move the upper tail
        assert c.quantile(0.95) - ref.quantile(0.95) == pytest.approx(s * REF_SIGMA, rel=1e-6)
        assert c.quantile(0.99) > ref.quantile(0.99) + 1.0


def test_matching_solver_hits_its_targets():
    slope = probit_slope_at_median(REF_SIGMA)
    for fam in BASE_FAMILIES:
        loc, scale = match_median_and_slope(fam, REF_MU, slope)
        f = BASE_FAMILIES[fam]
        kw = {"gap": 0.6} if fam == "mixture" else {}
        assert abs(float(f(REF_MU, loc, scale, **kw)) - 0.5) < 1e-10, fam
        h = 1e-5
        d = float(f(REF_MU + h, loc, scale, **kw) - f(REF_MU - h, loc, scale, **kw)) / (2 * h)
        assert abs(d - slope) < 1e-7, fam


def test_config_round_trip_through_the_simulator():
    for cd in ({"family": "probit", "mu": 30.0, "sigma": 3.0},
               {"family": "logistic", "lam": 0.5},
               {"family": "cloglog", "lam": 1.0},
               {"family": "robit", "lam": 1.0},
               {"family": "mixture", "lam": 0.5},
               {"family": "tail_perturbed", "shift95": 1.0},
               {"family": "beta_cdf"}):
        c = build_curve(cd)
        t = true_targets(c, ("mu", "sigma", "q0.5", "q0.95", "q0.99"))
        assert np.isfinite(list(t.values())).all(), cd
        assert t["q0.5"] < t["q0.95"] < t["q0.99"], cd
        assert t["sigma"] > 0, cd


# --------------------------------------------------------------------------
# Pseudo-truth
# --------------------------------------------------------------------------


def test_kl_projection_recovers_a_correctly_specified_curve():
    """Under no misspecification the projection must return the truth."""
    curve = ProbitCurve(30.0, 3.0)
    support = np.linspace(20, 40, 201)
    weights = np.full(len(support), 1.0 / len(support))
    proj = kl_projection(curve, support, weights)
    assert abs(proj["pseudo_mu"] - 30.0) < 1e-3
    assert abs(proj["pseudo_sigma"] - 3.0) < 1e-3
    q = pseudo_true_quantiles(proj)
    assert abs(q["q0.95"] - curve.quantile(0.95)) < 5e-3


def test_kl_projection_depends_on_the_design_distribution():
    """The mechanism the thesis is testing must be visible in the projection.

    Under a misspecified curve, restricting the design to different stimulus
    regions must move the projection.  If it did not, there would be no
    policy-dependent pseudo-truth to find.
    """
    curve = build_curve_family("cloglog", 1.0)
    lo = np.linspace(24, 31, 71)
    hi = np.linspace(29, 38, 91)
    p_lo = kl_projection(curve, lo, np.full(len(lo), 1 / len(lo)))
    p_hi = kl_projection(curve, hi, np.full(len(hi), 1 / len(hi)))
    q_lo = pseudo_true_quantiles(p_lo)["q0.95"]
    q_hi = pseudo_true_quantiles(p_hi)["q0.95"]
    assert abs(q_lo - q_hi) > 0.1, (q_lo, q_hi)


def test_empirical_design_distribution_snaps_to_the_grid():
    grid = np.linspace(16, 44, 200)
    x = np.array([16.0, 16.0, 30.0, 44.0])
    support, weights = empirical_design_distribution(x, grid)
    assert abs(weights.sum() - 1.0) < 1e-12
    assert len(support) == 3
    assert weights[np.argmin(np.abs(support - 16.0))] == pytest.approx(0.5)
