"""Can the collected data tell that the probit model is wrong?

Phase 3 established *that* tail-located misspecification destroys coverage
under an adaptive design (`docs/claims.md` C5).  This module supplies the
machinery for the companion question, which is the one that decides whether a
safeguard is optional: **is the failure visible in the data the design
collects?**

Three layers, in increasing order of what they assume the analyst knows.

1. `posterior_predictive_check` -- what a careful analyst does by default: a
   posterior predictive p-value under the fitted probit, with an omnibus
   discrepancy and a tail-directed one.
2. `log_evidence` -- what a good analyst does: marginal likelihoods for a set
   of competing link families under a common prior, hence Bayes factors.
   Every alternative is matched to the fitted probit on median and slope at
   the median (the same convention `curve_families` uses to build the DGPs),
   so a Bayes factor here compares *shape* and nothing else.
3. `lr_statistic` -- the Neyman-Pearson upper bound: a likelihood ratio
   against the exact alternative that generated the data, with its shape
   parameter free.  An oracle who knows the direction of the perturbation
   cannot do better than this, so if *this* has no power, no test does.

All three condition on the realised design ``x_{1:n}``.  That is legitimate
here and not an approximation: under an ignorable policy the design
contributes a factor constant in the parameter (`docs/claims.md` C1), so it
cancels from every likelihood ratio, every Bayes factor and every posterior
predictive draw taken at fixed ``x``.

Nothing in this module fits a model *to* the true curve's identity.  The
alternatives are named families; the fact that one of them happens to be the
truth is the point of layer 3, and is always labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp
from scipy.stats import norm
from scipy.stats import t as t_dist

from .curve_families import (
    REF_MU,
    REF_SIGMA,
    match_median_and_slope,
    probit_slope_at_median,
)
from .posterior_grid import GridSpec

# The tail perturbation is exactly probit below this response probability.
TAIL_P0 = 0.85

# Above this latent value Phi(u) is 1 to machine precision, so the perturbed
# quantile function is exactly u + kappa and needs no table.
_U_SATURATED = 9.0

SIGMA_LOG_BOUNDS = (np.log(1e-2), np.log(1e2))


# ---------------------------------------------------------------------------
# Fitted families
# ---------------------------------------------------------------------------
#
# Every family is parameterised by (m, s): the median and the *probit-
# equivalent scale*, i.e. the scale of the probit with the same slope at the
# median.  Two consequences the layers above rely on:
#
#   * one prior over (m, s) -- Rotem's -- is meaningful for every model, so a
#     Bayes factor is not contaminated by incomparable parameterisations;
#   * at (m, s) fixed, the families differ only in shape, which is the same
#     matching convention `curve_families` applies to the data-generating
#     curves.  A comparison here therefore asks exactly the question the DGPs
#     were built to pose.


@lru_cache(maxsize=16)
def _matched_reference(family: str) -> tuple[float, float]:
    """(loc, scale) matching `family` to probit(REF_MU, REF_SIGMA)."""
    return match_median_and_slope(family, REF_MU, probit_slope_at_median(REF_SIGMA))


def _loc_scale(family: str, m, s):
    """Map (median, probit-equivalent scale) to the family's own (loc, scale).

    Location-scale equivariance: matching at (REF_MU, REF_SIGMA) determines the
    match at every other (m, s) through the affine map that carries one to the
    other, so the root solve runs once per family rather than once per
    likelihood evaluation.
    """
    loc0, scale0 = _matched_reference(family)
    r = np.asarray(s, dtype=float) / REF_SIGMA
    return np.asarray(m, dtype=float) + (loc0 - REF_MU) * r, scale0 * r


def _probit_logprobs(x, m, s, shape=None):
    z = (x - m) / s
    return log_ndtr(z), log_ndtr(-z)


def _logistic_logprobs(x, m, s, shape=None):
    loc, scale = _loc_scale("logistic", m, s)
    u = (x - loc) / scale
    return -np.logaddexp(0.0, -u), -np.logaddexp(0.0, u)


def _cloglog_logprobs(x, m, s, shape=None):
    loc, scale = _loc_scale("cloglog", m, s)
    u = np.clip((x - loc) / scale, -700.0, 700.0)
    e = np.exp(u)
    # p = 1 - exp(-e).  log(1 - p) = -e is exact.  For log p the direct form
    # loses everything as e -> 0, where the series log(-expm1(-e)) = log e -
    # e/2 + O(e^2) is exact to double precision and stays finite; both branches
    # of the `where` must be evaluable on the whole input, hence the clamp.
    small = e < 1e-8
    log_p = np.where(small, u - 0.5 * e,
                     np.log(-np.expm1(-np.maximum(e, 1e-8))))
    return log_p, -e


def _robit_logprobs(x, m, s, shape=None):
    df = 3.0 if shape is None else float(shape)
    loc, scale = _loc_scale("robit", m, s)
    u = (x - loc) / scale
    return t_dist.logcdf(u, df), t_dist.logcdf(-u, df)


@lru_cache(maxsize=256)
def _tail_inverse_table(kappa: float, p0: float, n_nodes: int = 40_001):
    """Nodes for inverting g(u) = u + kappa * ((Phi(u) - p0)/(1 - p0))_+^2.

    `g` is the perturbed quantile function in standardised units: the true
    curve has q(p) = m + s * g(Phi^{-1}(p)).  It is strictly increasing and --
    crucially -- does **not** depend on (m, s), so one table per shape
    parameter inverts the family at every location and scale.  That is what
    makes evaluating the likelihood on a grid of 10^5 parameter cells
    affordable, and it is why this module does not reuse
    `curve_families._tail_table`, which is keyed on (m, s) as well.

    Only [Phi^{-1}(p0), _U_SATURATED] needs a table: below it g is the identity
    and above it g(u) = u + kappa, both exactly.
    """
    u = np.linspace(float(norm.ppf(p0)), _U_SATURATED, n_nodes)
    frac = (norm.cdf(u) - p0) / (1.0 - p0)
    g = u + kappa * np.clip(frac, 0.0, None) ** 2
    return g, u


def _g(u, kappa: float, p0: float):
    return u + kappa * np.clip((norm.cdf(u) - p0) / (1.0 - p0), 0.0, None) ** 2


def _tail_invert_exact(t, kappa: float, p0: float, n_iter: int = 60):
    """g^{-1}(t) by vectorised bisection, to machine precision.

    `g` is increasing with u <= g(u) <= u + kappa, so [t - kappa, t] brackets
    the root for every t and 60 halvings of an interval of width kappa land far
    inside double precision.
    """
    lo = t - kappa
    hi = np.array(t, dtype=float, copy=True)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        go_up = _g(mid, kappa, p0) < t
        lo = np.where(go_up, mid, lo)
        hi = np.where(go_up, hi, mid)
    return 0.5 * (lo + hi)


# Above this many parameter-by-stimulus evaluations the exact bisection is no
# longer affordable and the interpolation table takes over.  A 385^2 evidence
# grid is ~10^7 evaluations; an MLE step is ~10^2.
_EXACT_MAX_ELEMENTS = 200_000


def _tail_invert(t, kappa: float, p0: float = TAIL_P0):
    """g^{-1}(t), vectorised over any shape of `t`.

    Two implementations, selected on size, agreeing to ~1e-9 in probability
    (`tests/test_model_check.py::test_tail_inversion_agrees`).  The exact one
    is used wherever it is affordable because the table's O(1e-9) interpolation
    error is *not* negligible against a finite-difference derivative: with a
    step of 1e-8 in the shape parameter the two are the same size, so an
    optimiser handed the table sees pure noise for a gradient and stops at its
    starting point.  That is not a hypothetical -- it is why this function does
    not simply always interpolate.
    """
    t = np.asarray(t, dtype=float)
    if kappa == 0.0:
        return t
    if t.size <= _EXACT_MAX_ELEMENTS:
        return _tail_invert_exact(t, kappa, p0)
    u0 = float(norm.ppf(p0))
    g_nodes, u_nodes = _tail_inverse_table(float(kappa), float(p0))
    out = np.where(t <= u0, t, t - kappa)          # the two closed-form branches
    mid = (t > u0) & (t < _U_SATURATED + kappa)
    if np.any(mid):
        out = np.where(mid, np.interp(t, g_nodes, u_nodes), out)
    return out


def shift_to_kappa(shift95: float) -> float:
    return float(shift95) / ((0.95 - TAIL_P0) / (1.0 - TAIL_P0)) ** 2


def _tail_logprobs(x, m, s, shape=None):
    """The tail-perturbed family, with the shift at p = 0.95 as shape parameter.

    `shape` is `shift95` in units of s, the same parameterisation
    `curve_families.TailPerturbedCurve` uses, and shape = 0 is *exactly* the
    probit -- so the probit is nested in this family and `lr_statistic` is a
    genuine one-degree-of-freedom test.
    """
    u = _tail_invert((x - m) / s, shift_to_kappa(0.0 if shape is None else shape))
    return log_ndtr(u), log_ndtr(-u)


@dataclass(frozen=True)
class FittedFamily:
    name: str
    logprobs: object
    shape_name: str | None = None
    shape_start: float = 0.0
    shape_bounds: tuple | None = None

    @property
    def n_shape(self) -> int:
        return 0 if self.shape_name is None else 1


PROBIT = FittedFamily("probit", _probit_logprobs)
LOGISTIC = FittedFamily("logistic", _logistic_logprobs)
CLOGLOG = FittedFamily("cloglog", _cloglog_logprobs)
ROBIT = FittedFamily("robit3", _robit_logprobs)
TAIL_FREE = FittedFamily("tail_free", _tail_logprobs, "shift95", 0.3, (0.0, 4.0))

FAMILIES = {f.name: f for f in (PROBIT, LOGISTIC, CLOGLOG, ROBIT, TAIL_FREE)}


def tail_fixed(shift95: float) -> FittedFamily:
    """The tail-perturbed family with its shape *frozen* at a known value.

    Used as the oracle alternative in the evidence layer: a two-parameter model
    carrying the same prior as the probit, so the Bayes factor against it needs
    neither a prior over the shape nor an extra grid dimension.  Frozen at the
    truth it is the most favourable alternative any analyst could have
    proposed, and is always reported as such.
    """

    def _f(x, m, s, shape=None):
        return _tail_logprobs(x, m, s, shift95)

    return FittedFamily(f"tail_fixed{shift95:g}", _f)


# ---------------------------------------------------------------------------
# Log-likelihood, collapsed over repeated stimuli
# ---------------------------------------------------------------------------


def collapse(x, y):
    """(distinct stimuli, successes, failures), sorted ascending in x.

    Adaptive designs revisit the same grid point many times, so this is the
    difference between 50 and ~15 curve evaluations per likelihood call.  It is
    the same reduction `posterior_grid.aggregated_loglik` performs, and it is
    exact.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y).astype(int)
    xu, inv = np.unique(x, return_inverse=True)
    n1 = np.bincount(inv, weights=(y == 1), minlength=len(xu))
    n0 = np.bincount(inv, weights=(y == 0), minlength=len(xu))
    return xu, n1, n0


def loglik(family: FittedFamily, xu, n1, n0, m, s, shape=None):
    """Total log-likelihood, broadcast over K parameter values; returns (K,)."""
    m = np.atleast_1d(np.asarray(m, dtype=float))
    s = np.atleast_1d(np.asarray(s, dtype=float))
    lp, lq = family.logprobs(xu[None, :], m[:, None], s[:, None], shape)
    return lp @ n1 + lq @ n0


def _pack(family, m, s, shape):
    p = [m, np.log(s)]
    if family.n_shape:
        p.append(shape)
    return np.asarray(p, dtype=float)


def _unpack(family, p):
    m, s = float(p[0]), float(np.exp(p[1]))
    shape = float(p[2]) if family.n_shape else None
    return m, s, shape


@dataclass
class Fit:
    family: str
    m: float
    s: float
    shape: float | None
    loglik: float
    converged: bool

    def as_dict(self, prefix=""):
        out = {f"{prefix}{self.family}_m": self.m,
               f"{prefix}{self.family}_s": self.s,
               f"{prefix}{self.family}_loglik": self.loglik,
               f"{prefix}{self.family}_converged": self.converged}
        if self.shape is not None:
            out[f"{prefix}{self.family}_shape"] = self.shape
        return out


def fit_mle(family: FittedFamily, x, y, start=None) -> Fit:
    """Maximum likelihood for one fitted family.

    Separation (no overlapping response pattern) leaves the likelihood without
    an interior maximum; the optimiser still runs and the result is reported
    with `converged=False` rather than dropped, per `CLAUDE.md` section 4.
    """
    xu, n1, n0 = collapse(x, y)
    if start is None:
        start = (float(np.mean(x)), max(float(np.std(x)), 0.5))
    m0, s0 = start

    def neg(p):
        m, s, shape = _unpack(family, p)
        return -float(loglik(family, xu, n1, n0, m, s, shape)[0])

    bounds = [(None, None), SIGMA_LOG_BOUNDS]
    if family.n_shape:
        bounds.append(family.shape_bounds)
    res = minimize(neg, _pack(family, m0, s0, family.shape_start),
                   method="L-BFGS-B", bounds=bounds)
    m, s, shape = _unpack(family, res.x)
    return Fit(family.name, m, s, shape, -float(res.fun), bool(res.success))


# ---------------------------------------------------------------------------
# Layer 0: how much evidence the design collected at all
# ---------------------------------------------------------------------------


def design_information(curve, reference, x, u=None) -> dict:
    """What the realised design could possibly reveal about `curve` vs `reference`.

    This layer is prior to every test.  Conditional on the stimuli actually
    applied, the responses are independent Bernoulli draws, so the whole
    discriminating content of the sample is

        KL = sum_i KL( Bern(p*(x_i)) || Bern(p_0(x_i)) ),

    with `p*` the true curve and `p_0` the probit it was matched to.  By
    Pinsker, the total variation between the two response distributions is at
    most sqrt(KL / 2), and by Neyman-Pearson **no** test at level alpha -- of
    any form, against a known or an unknown alternative, frequentist or
    Bayesian -- can exceed power alpha + TV.  `max_power_bound` is that ceiling.

    `expected_flips` is the same fact in units an experimenter can feel: the
    expected number of the n trials whose *observed response* would have come
    out differently had the truth been the probit, given the same latent
    draws.  When the design never visits the region where the curves differ it
    is ~0, and then the two hypotheses did not merely go undetected -- they
    generated the same data.

    Passing the realised latent uniforms `u` adds `realized_flips`, the
    counterfactual actually drawn.  Everything here conditions on the realised
    x, which for an adaptive policy is itself a function of the responses; the
    quantity is therefore a property of the run, not an average over designs,
    and is summarised across replicates rather than interpreted alone.
    """
    x = np.asarray(x, dtype=float)
    p1 = np.clip(np.atleast_1d(curve.prob(x)), 1e-15, 1 - 1e-15)
    p0 = np.clip(np.atleast_1d(reference.prob(x)), 1e-15, 1 - 1e-15)
    kl = float(np.sum(p1 * np.log(p1 / p0) + (1 - p1) * np.log((1 - p1) / (1 - p0))))
    tv = float(min(np.sqrt(max(kl, 0.0) / 2.0), 1.0))
    out = {
        "design_kl_nats": kl,
        "design_tv_bound": tv,
        "max_power_bound": float(min(0.05 + tv, 1.0)),
        "expected_flips": float(np.sum(np.abs(p1 - p0))),
        "max_abs_prob_gap": float(np.max(np.abs(p1 - p0))),
    }
    if u is not None:
        u = np.asarray(u, dtype=float)[: len(x)]
        lo, hi = np.minimum(p0, p1), np.maximum(p0, p1)
        out["realized_flips"] = int(np.sum((u >= lo) & (u < hi)))
    return out


# ---------------------------------------------------------------------------
# Layer 3: the oracle likelihood ratio
# ---------------------------------------------------------------------------


def lr_statistic(x, y, alternative: FittedFamily = TAIL_FREE,
                 probit_fit: Fit | None = None) -> dict:
    """2 * (max log-lik under `alternative` - max log-lik under probit).

    The probit is nested in `alternative` at shape = 0, so this is a
    one-parameter likelihood ratio, non-negative up to optimiser error.  With
    `alternative = TAIL_FREE` the test knows the exact functional form of the
    perturbation and has only to estimate its size: no procedure ignorant of
    the truth can have more power against this alternative, so the power of
    this test is an **upper bound on detectability**.

    The shape parameter is bounded below at 0, so the asymptotic null is a
    50:50 mixture of chi^2_0 and chi^2_1 rather than chi^2_1.  Nothing here
    relies on that: the null is calibrated by simulation
    (`experiments/phase4_undetectability.py`), which also absorbs the fact that
    n = 50 under a sequentially chosen design is not an asymptotic regime.
    """
    p = probit_fit or fit_mle(PROBIT, x, y)
    a = fit_mle(alternative, x, y, start=(p.m, p.s))
    stat = 2.0 * (a.loglik - p.loglik)
    return {"lr": float(max(stat, 0.0)), "lr_raw": float(stat),
            "lr_shape_hat": a.shape,
            "lr_converged": bool(p.converged and a.converged),
            "probit_loglik": p.loglik, "alt_loglik": a.loglik}


# ---------------------------------------------------------------------------
# Layer 2: marginal likelihood on a shared grid
# ---------------------------------------------------------------------------


def shared_box(prior, x, y, *, pad: float = 0.25, n_localise: int = 129,
               n_grid: int = 385) -> GridSpec:
    """One (m, eta) box for every model, localised on the probit posterior.

    A Bayes factor is a ratio of integrals, so the two integrals must be taken
    over the same region or the ratio means nothing.  The box is found from the
    probit posterior -- the model in hand -- then inflated by `pad` times its
    half-width in each direction so that an alternative whose posterior sits
    elsewhere is still contained.  Containment is not assumed: `log_evidence`
    returns each model's boundary mass and the caller records it.

    Localisation is delegated to `build_reference_posterior`, which iterates
    shrink-and-rebuild until the box *stops moving* and the boundary mass is
    below tolerance.  Running `_shrink_to_mass` a fixed number of times instead
    does not converge: each pass pads by 25%, so on a coarse grid whose extreme
    marginal quantiles sit in the outermost cell the box grows without bound.

    `pad` is deliberately small.  The localised box already holds all but
    1e-10 of the probit posterior in each margin -- tens of posterior standard
    deviations wide, because the sigma margin is heavy tailed -- so inflating
    it further only costs resolution.  Measured on Phase 3B histories, the log
    Bayes factors agree to 4 decimal places across pad in {0, 0.25, 1} and to
    ~1e-7 between n_grid 385 and 1537.
    """
    from .posterior_grid import build_reference_posterior

    spec = build_reference_posterior(prior, x, y, fixed_n=n_localise).spec
    mc, ec = 0.5 * (spec.mu_lo + spec.mu_hi), 0.5 * (spec.eta_lo + spec.eta_hi)
    mh, eh = 0.5 * (spec.mu_hi - spec.mu_lo), 0.5 * (spec.eta_hi - spec.eta_lo)
    return GridSpec(max(1e-6, mc - (1 + pad) * mh), mc + (1 + pad) * mh,
                    ec - (1 + pad) * eh, ec + (1 + pad) * eh, n_grid, n_grid)


def log_evidence(family: FittedFamily, prior, x, y, spec: GridSpec) -> tuple[float, float]:
    """(log p(y | x, model), boundary mass), by midpoint quadrature on `spec`.

    The prior is Rotem's on (m, s) for *every* family -- see the
    parameterisation note above -- so the returned evidences are directly
    comparable and their differences are log Bayes factors under equal prior
    model odds.
    """
    mu_c = np.linspace(spec.mu_lo, spec.mu_hi, spec.n_mu)
    eta_c = np.linspace(spec.eta_lo, spec.eta_hi, spec.n_eta)
    MU, ETA = np.meshgrid(mu_c, eta_c, indexing="ij")
    m = MU.ravel()
    s = np.exp(ETA.ravel())

    logpri = prior.logpdf(m, s) + np.log(s)        # Jacobian d sigma / d eta
    finite = np.isfinite(logpri)
    ll = np.zeros_like(logpri)
    xu, n1, n0 = collapse(x, y)
    ll[finite] = loglik(family, xu, n1, n0, m[finite], s[finite])

    logpost = np.where(finite, logpri + ll, -np.inf)
    d_mu = (spec.mu_hi - spec.mu_lo) / (spec.n_mu - 1)
    d_eta = (spec.eta_hi - spec.eta_lo) / (spec.n_eta - 1)
    logZ = logsumexp(logpost) + np.log(d_mu * d_eta)

    w = np.exp(logpost - logsumexp(logpost)).reshape(spec.n_mu, spec.n_eta)
    edge = (w[0, :].sum() + w[-1, :].sum() + w[:, 0].sum() + w[:, -1].sum()
            - w[0, 0] - w[0, -1] - w[-1, 0] - w[-1, -1])
    return float(logZ), float(edge)


# ---------------------------------------------------------------------------
# Layer 1: posterior predictive checks under the fitted probit
# ---------------------------------------------------------------------------


def _discrepancies(counts, n_tot, p, tail_mask):
    """Discrepancies for one (data, parameter) pair.

    `counts` are successes at each distinct stimulus and `p` the fitted
    probabilities there.  Both statistics depend on the parameter, so the
    calibrating quantity is Gelman's posterior predictive p-value
    P(T(y_rep, theta) >= T(y, theta) | y) over *joint* draws of (theta, y_rep).
    """
    exp = n_tot * p
    var = np.maximum(n_tot * p * (1.0 - p), 1e-12)
    chi2 = float(np.sum((counts - exp) ** 2 / var))
    tail = float(np.sum((counts - exp)[tail_mask]) / np.sqrt(np.sum(var[tail_mask])))
    return chi2, tail


def _midp(rep, obs, side: str) -> float:
    """Predictive tail probability with ties counted half, not whole.

    The discrepancies are functions of integer counts, so `T_rep == T_obs`
    happens constantly -- and in the upper stimuli of an adaptive design, where
    the fitted probability is near one and every replicate reproduces the
    observed successes exactly, it is the *typical* case.  Counting those ties
    as evidence against the model drives the raw p-value to 0 and the resulting
    check rejects the correct model on the majority of null replicates.  The
    mid-p convention restores an approximately uniform null, which is what a
    p-value has to have before a detection rate computed from it means
    anything.
    """
    strict = np.mean(rep > obs) if side == "hi" else np.mean(rep < obs)
    return float(strict + 0.5 * np.mean(rep == obs))


def posterior_predictive_check(prior, x, y, rng, *, n_draws: int = 200,
                               fixed_n: int = 513, tail_frac: float = 0.25,
                               posterior=None) -> dict:
    """Posterior predictive p-values for the fitted probit at the realised design.

    Two discrepancies:

    * ``chi2`` -- a grouped Pearson statistic over the distinct stimuli: the
      omnibus check an analyst runs with no hypothesis in mind.
    * ``tail`` -- the standardised sum of residuals over the upper `tail_frac`
      of visited stimuli, signed, so a curve that is *harder to ignite* than
      the probit at high stress appears as a negative value.  This is the check
      an analyst runs who already suspects the tail, and so is the more
      favourable of the two to the detection hypothesis.

    Returns a one-sided p-value for ``chi2`` and a two-sided one for ``tail``.
    """
    from .posterior_grid import build_reference_posterior

    post = posterior or build_reference_posterior(prior, x, y, fixed_n=fixed_n)
    xu, n1, n0 = collapse(x, y)
    n_tot = (n1 + n0).astype(int)
    k = max(1, int(np.ceil(tail_frac * len(xu))))
    tail_mask = np.zeros(len(xu), dtype=bool)
    tail_mask[-k:] = True                      # collapse() returns xu ascending

    m_draw, s_draw = post.sample(n_draws, rng)
    p = norm.cdf((xu[None, :] - m_draw[:, None]) / s_draw[:, None])
    y_rep = rng.binomial(n_tot[None, :], p)

    obs = np.empty((n_draws, 2))
    rep = np.empty((n_draws, 2))
    for d in range(n_draws):
        obs[d] = _discrepancies(n1, n_tot, p[d], tail_mask)
        rep[d] = _discrepancies(y_rep[d], n_tot, p[d], tail_mask)

    lower = _midp(rep[:, 1], obs[:, 1], "lo")
    return {
        "ppp_chi2": _midp(rep[:, 0], obs[:, 0], "hi"),
        "ppp_tail": float(2.0 * min(lower, 1.0 - lower)),
        "ppp_tail_lower": lower,
        "ppc_n_distinct": int(len(xu)),
        "ppc_n_tail_bins": int(k),
        "ppc_obs_tail_mean": float(np.mean(obs[:, 1])),
    }
