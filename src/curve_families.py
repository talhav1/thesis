"""Controlled monotone alternatives to the probit response curve.

Every family is matched to a reference probit on **median** and **slope at the
median**, by numerically solving for the family's own location and scale.
Severity is then varied continuously by convex combination on the probability
scale:

    p_lambda(x) = (1 - lambda) * probit(x) + lambda * alt(x).

That construction has three properties the protocol relies on:

* it is monotone for every lambda, being a convex combination of monotone
  functions;
* at lambda = 0 it is *exactly* the probit, so the correct-model control is
  nested in the family rather than merely adjacent to it;
* because both components share the median and the slope there, the mixture
  preserves both matched features at every lambda.

So a failure observed as lambda grows is attributable to the *shape* of the
curve, not to an incidental shift or rescaling -- which is what the plan means
by "vary misspecification continuously ... so failure can be attributed to
shape rather than arbitrary scale changes".

The exception is `BetaCDFCurve` (in `response_models`), Rotem's inherited case,
which is deliberately *not* matched and is always labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from functools import lru_cache

from scipy.optimize import brentq, root
from scipy.stats import beta as beta_dist
from scipy.stats import logistic as logistic_dist
from scipy.stats import norm, t as t_dist

from .response_models import ProbitCurve

REF_MU = 30.0
REF_SIGMA = 3.0


def probit_slope_at_median(sigma: float) -> float:
    return float(norm.pdf(0.0) / sigma)


# --------------------------------------------------------------------------
# Base families, each parameterised by (location, scale)
# --------------------------------------------------------------------------


def _logistic(x, loc, scale):
    return logistic_dist.cdf(x, loc=loc, scale=scale)


def _cloglog(x, loc, scale):
    # exp overflows for large (x - loc)/scale; the curve is 1 there to any
    # representable precision, so clip the exponent rather than warn.
    u = np.clip((np.asarray(x, dtype=float) - loc) / scale, -700.0, 700.0)
    return -np.expm1(-np.exp(u))


def _robit(x, loc, scale, df=3.0):
    return t_dist.cdf((np.asarray(x, dtype=float) - loc) / scale, df)


def _mixture(x, loc, scale, gap=0.6):
    """Symmetric two-component probit mixture: a monotone 'shoulder' curve.

    Components are placed at loc +/- gap*scale with common width `scale`, so
    the median is `loc` by symmetry for any gap.
    """
    x = np.asarray(x, dtype=float)
    return 0.5 * (norm.cdf((x - loc + gap * scale) / scale)
                  + norm.cdf((x - loc - gap * scale) / scale))


BASE_FAMILIES = {
    "logistic": _logistic,
    "cloglog": _cloglog,
    "robit": _robit,
    "mixture": _mixture,
}


def match_median_and_slope(family: str, median: float, slope: float,
                           x0=None, gap: float = 0.6):
    """Solve for (loc, scale) giving the target median and slope at it.

    Uses a central difference for the slope, which is exact to O(h^2) and
    avoids having to hand-derive a density for every family.
    """
    f = BASE_FAMILIES[family]
    kw = {"gap": gap} if family == "mixture" else {}
    h = 1e-4

    def residual(theta):
        loc, log_scale = theta
        scale = np.exp(log_scale)
        p_mid = float(f(median, loc, scale, **kw))
        d = float(f(median + h, loc, scale, **kw) - f(median - h, loc, scale, **kw)) / (2 * h)
        return [p_mid - 0.5, d - slope]

    start = x0 if x0 is not None else [median, np.log(1.0 / (2.5 * slope))]
    # No `tol` override: asking hybr for 1e-13 makes it declare "not making
    # good progress" on residuals it has in fact already driven to ~1e-13.
    # The explicit residual assertion below is the real acceptance test.
    sol = root(residual, start, method="hybr")
    loc, scale = float(sol.x[0]), float(np.exp(sol.x[1]))
    res = residual(sol.x)
    if max(abs(r) for r in res) > 1e-9:
        raise RuntimeError(
            f"median/slope matching failed for {family}: residual {res}, {sol.message}"
        )
    return loc, scale


# --------------------------------------------------------------------------
# Curve objects
# --------------------------------------------------------------------------


class _NumericQuantileMixin:
    """Physical quantiles by monotone root-finding on the response curve."""

    _search_lo = -200.0
    _search_hi = 400.0

    def quantile(self, p):
        scalar = np.isscalar(p)
        ps = np.atleast_1d(p).astype(float)
        out = np.empty(len(ps))
        for i, pi in enumerate(ps):
            out[i] = brentq(
                lambda x: float(np.atleast_1d(self.prob(np.array([x])))[0]) - pi,
                self._search_lo, self._search_hi, xtol=1e-12, rtol=1e-14,
            )
        return float(out[0]) if scalar else out


@dataclass(frozen=True)
class MatchedMixtureCurve(_NumericQuantileMixin):
    """(1 - lambda) * probit  +  lambda * matched alternative."""

    family: str
    lam: float
    loc: float
    scale: float
    mu0: float = REF_MU
    sigma0: float = REF_SIGMA
    gap: float = 0.6
    name: str = ""

    def prob(self, x):
        x = np.asarray(x, dtype=float)
        base = norm.cdf((x - self.mu0) / self.sigma0)
        if self.lam == 0.0:
            return base
        kw = {"gap": self.gap} if self.family == "mixture" else {}
        alt = BASE_FAMILIES[self.family](x, self.loc, self.scale, **kw)
        return (1.0 - self.lam) * base + self.lam * alt

    def describe(self):
        return {"family": self.family, "lambda": self.lam, "loc": self.loc,
                "scale": self.scale, "mu0": self.mu0, "sigma0": self.sigma0,
                "gap": self.gap, "matched": True, "name": self.name}


@dataclass(frozen=True)
class TailPerturbedCurve(_NumericQuantileMixin):
    """Probit below p0; a controlled upper-tail shift above it.

    Defined by perturbing the *quantile* function,

        q(p) = mu0 + sigma0 z_p + kappa * sigma0 * max(0, (p - p0)/(1 - p0))^2,

    which is strictly increasing in p, so the response curve q^{-1} is a valid
    monotone curve, and is *identical to the probit* for p <= p0.  With
    p0 = 0.85 that means the curve is exactly probit for x below roughly the
    0.85 quantile -- the region the adaptive designs actually visit -- and
    diverges only near q_0.95 and q_0.99.

    This is the family designed to defeat the mechanism found in Phase 2, where
    the Beta-CDF's global shape mismatch inflated the posterior width enough to
    absorb its own bias.  Here the likelihood sees almost nothing wrong.

    `shift95` is the shift at p = 0.95 in units of sigma0, which fixes kappa.
    """

    shift95: float
    mu0: float = REF_MU
    sigma0: float = REF_SIGMA
    p0: float = 0.85
    name: str = ""

    @property
    def kappa(self) -> float:
        u = (0.95 - self.p0) / (1.0 - self.p0)
        return float(self.shift95 / u**2)

    def q_of_p(self, p):
        p = np.asarray(p, dtype=float)
        u = np.clip((p - self.p0) / (1.0 - self.p0), 0.0, None)
        return self.mu0 + self.sigma0 * norm.ppf(p) + self.kappa * self.sigma0 * u**2

    def prob(self, x):
        """Vectorised inverse of the perturbed quantile function.

        Below p0 the curve is exactly probit, so that branch is closed form.
        Above p0 the inverse is obtained from a monotone lookup table built
        once per parameter set: q is strictly increasing, so interpolating p
        against q *is* the inverse.  A root solve per evaluation was the
        original implementation and is far too slow to sit inside a simulator
        loop; it also cannot bracket x beyond q(1 - eps), which is exactly what
        a wide verification grid asks for.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        q_grid, p_grid = _tail_table(self.shift95, self.mu0, self.sigma0, self.p0)
        base = norm.cdf((x - self.mu0) / self.sigma0)
        upper = np.interp(x, q_grid, p_grid, left=p_grid[0], right=p_grid[-1])
        return np.where(x <= q_grid[0], base, upper)

    def quantile(self, p):
        scalar = np.isscalar(p)
        out = self.q_of_p(np.atleast_1d(p))
        return float(out[0]) if scalar else out

    def describe(self):
        return {"family": "tail_perturbed", "shift95_in_sigma0": self.shift95,
                "kappa": self.kappa, "p0": self.p0, "mu0": self.mu0,
                "sigma0": self.sigma0, "matched": "exact below p0",
                "name": self.name}


@lru_cache(maxsize=64)
def _tail_table(shift95, mu0, sigma0, p0, n_nodes=200_001, u_max=8.5):
    """Monotone (q, p) table for TailPerturbedCurve above p0.

    Nodes are equally spaced in the probit scale u = Phi^{-1}(p), which places
    resolution where the curve changes fastest and keeps the table finite.
    """
    u_lo = norm.ppf(p0)
    u = np.linspace(u_lo, u_max, n_nodes)
    p = norm.cdf(u)
    frac = np.clip((p - p0) / (1.0 - p0), 0.0, None)
    kappa = shift95 / ((0.95 - p0) / (1.0 - p0)) ** 2
    q = mu0 + sigma0 * u + kappa * sigma0 * frac**2
    if np.any(np.diff(q) <= 0):
        raise RuntimeError("tail-perturbed quantile function is not increasing")
    return q, p


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def build_curve_family(family: str, lam: float, mu0=REF_MU, sigma0=REF_SIGMA,
                       gap: float = 0.6, name: str = ""):
    """Matched alternative at severity `lam` against probit(mu0, sigma0)."""
    if family == "probit" or lam == 0.0:
        return MatchedMixtureCurve("logistic", 0.0, mu0, 1.0, mu0, sigma0,
                                   name=name or "probit")
    loc, scale = match_median_and_slope(family, mu0, probit_slope_at_median(sigma0),
                                        gap=gap)
    return MatchedMixtureCurve(family, lam, loc, scale, mu0, sigma0, gap,
                               name=name or f"{family}_lam{lam}")


def verify_curve(curve, lo=-30.0, hi=100.0, n=60_001, tol=1e-10) -> dict:
    """Monotonicity and quantile self-consistency.  Run before any experiment.

    Checks (a) the curve is non-decreasing on a dense grid, (b) it spans
    [0, 1], and (c) `quantile(p)` really inverts `prob`, by evaluating the
    curve at the returned quantile and comparing with p.
    """
    xs = np.linspace(lo, hi, n)
    p = np.atleast_1d(curve.prob(xs))
    diffs = np.diff(p)
    # Whether the curve spans [0, 1] is a statement about its *limits*, not
    # about any finite window: a heavy-tailed robit link is still only at
    # p = 1e-4 by x = -30.  Evaluate the limits far outside the window.
    far = np.atleast_1d(curve.prob(np.array([-1e5, 1e5])))
    report = {
        "monotone": bool(diffs.min() >= -tol),
        "min_diff": float(diffs.min()),
        "p_min": float(p.min()),
        "p_max": float(p.max()),
        "p_at_minus_inf": float(far[0]),
        "p_at_plus_inf": float(far[1]),
        "spans_unit_interval": bool(far[0] < 1e-8 and far[1] > 1 - 1e-8),
    }
    errs = []
    for pp in (0.01, 0.05, 0.5, 0.95, 0.99):
        q = float(np.atleast_1d(curve.quantile(pp))[0])
        back = float(np.atleast_1d(curve.prob(np.array([q])))[0])
        errs.append(abs(back - pp))
        report[f"q{pp}"] = q
    report["max_quantile_inversion_error"] = float(max(errs))
    report["quantiles_consistent"] = bool(max(errs) < 1e-8)
    report["median"] = report["q0.5"]

    h = 1e-4
    m = report["q0.5"]
    slope = float(np.atleast_1d(curve.prob(np.array([m + h])))[0]
                  - np.atleast_1d(curve.prob(np.array([m - h])))[0]) / (2 * h)
    report["slope_at_median"] = slope
    report["ok"] = bool(report["monotone"] and report["quantiles_consistent"])
    return report


def curve_true_sd(curve, lo=-40.0, hi=120.0, n=40_001) -> float:
    """SD of the latent tolerance distribution implied by the response curve.

    The response curve *is* a CDF, so its standard deviation is well defined
    and computed by numerical integration of the implied density.  Used only as
    a descriptive scale, never as an estimand.
    """
    xs = np.linspace(lo, hi, n)
    p = np.atleast_1d(curve.prob(xs))
    dens = np.gradient(p, xs)
    dens = np.clip(dens, 0, None)
    mass = np.trapezoid(dens, xs)
    if mass <= 0:
        return float("nan")
    dens = dens / mass
    m = np.trapezoid(xs * dens, xs)
    return float(np.sqrt(max(np.trapezoid((xs - m) ** 2 * dens, xs), 0.0)))
