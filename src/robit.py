"""The robit fitted family: a Student-t link carrying a free tail parameter.

**Additive.**  Nothing here changes the two-parameter probit path.  Every
existing module, test and result continues to run against
`response_models.probit_*` and `posterior_grid.GridPosterior` unmodified; the
robit path is a parallel one, entered only by code that asks for it.

Why a free tail parameter at all
--------------------------------
The Phase 3/4A failure is not that the probit's tail shape is wrong.  It is
that the tail shape is **assumed with certainty**.  Under
``p(x) = Phi((x - mu)/sigma)`` the quantile ``q_{0.99} = mu + sigma z_{0.99}``
is a fixed affine function of two parameters that the design pins down near
the median, so the posterior for ``q_{0.99}`` inherits the precision of data
collected nowhere near it.  Phase 4A showed the resulting error is not merely
unnoticed: in 65.5% of adaptive-MI runs the misspecified and correct curves
generated bit-identical data, so no check could have seen it
(`manuscript/phase4_undetectability_note.md`).

Enlarging the fitted family to

    p(x) = T_nu((x - mu) / sigma),      nu = 1 / r,

cuts that transfer.  ``z_{0.99}`` is no longer a constant but ``t^{-1}_{1/r}
(0.99)``, which ranges from 2.33 at ``r = 0`` to 6.96 at ``r = 0.5``.  Data at
the median constrain ``(mu, sigma)`` and say very little about ``r``, so the
``q_{0.99}`` posterior should *widen*.  Phase 4A predicts ``r`` is nearly
unidentified at n = 50; that is the mechanism, not a defect.

Why r = 1/nu on [0, 0.5] and not nu or log nu
---------------------------------------------
Because ``r = 0`` is **exactly** the probit, and it is an interior-reachable
endpoint of a bounded interval rather than a limit at infinity.  The
correct-model control is therefore *nested in the fitted family*, mirroring
`curve_families`, where every alternative reduces to the probit exactly at
``lambda = 0``.  Two consequences the rest of the phase relies on:

* a prior degenerate at ``r = 0`` reproduces the existing two-parameter
  posterior exactly, which is the hard invariant in
  `tests/test_robit_nesting.py`;
* the r-axis of a grid can carry ``r = 0`` as a node, so the nesting is exact
  in floating point and not merely a limit.

``r_max = 0.5`` is ``nu = 2``: the smallest degrees of freedom at which the
t distribution still has a finite variance.  Below it the implied tolerance
distribution has no scale, and "sigma" would stop meaning anything.

Numerics
--------
``r = 0`` is dispatched to `log_ndtr`, never to ``t`` with a huge ``df``.  For
``r > 0`` the smallest positive node on an n_r-point grid over [0, 0.5] is
``0.5/(n_r - 1)``, i.e. ``nu = 64`` at the Phase 6 production resolution
n_r = 33, where `scipy.stats.t` is well conditioned.  No probability floor is
applied anywhere in a log-likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import log_ndtr, ndtr, stdtr
from scipy.stats import beta as beta_dist
from scipy.stats import norm
from scipy.stats import t as t_dist

# nu = 2 is the heaviest tail with a finite variance.  See the module note.
R_MAX = 0.5


# ---------------------------------------------------------------------------
# The link
# ---------------------------------------------------------------------------


def nu_of_r(r):
    """Degrees of freedom implied by the tail parameter; r = 0 gives inf."""
    r = np.asarray(r, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(r > 0, 1.0 / np.where(r > 0, r, 1.0), np.inf)


def robit_prob(x, mu, sigma, r):
    """T_nu((x - mu)/sigma) with nu = 1/r.  Scalar `r`, broadcast x/mu/sigma."""
    z = (np.asarray(x, dtype=float) - np.asarray(mu, dtype=float)) / np.asarray(
        sigma, dtype=float
    )
    return _cdf_z(z, float(r))


def _cdf_z(z, r: float):
    return ndtr(z) if r <= 0.0 else t_dist.cdf(z, 1.0 / r)


def robit_logprobs_z(z, r: float):
    """(log T_nu(z), log T_nu(-z)).  Stable in both tails; no floor applied."""
    return _logcdf(z, r), _logcdf(-z, r)


def _logcdf(z, r: float):
    """log T_{1/r}(z), or log Phi(z) at r = 0.

    `scipy.special.stdtr` is the raw quadrature underneath `scipy.stats.t`, and
    calling it directly skips the `rv_continuous` dispatch that otherwise
    dominates a call on a large array -- worth 1.3-3x on this workload, which is
    the innermost loop of every three-parameter grid build.  It returns the CDF,
    not its log, so it underflows to exactly 0 in the far left tail (around
    z = -400 at nu = 32).  Those entries -- and only those -- are recomputed
    with `t.logcdf`, which is exact there.  Nothing is floored: the two paths
    agree to 1.1e-16 wherever both are defined.
    """
    z = np.asarray(z, dtype=float)
    if r <= 0.0:
        return log_ndtr(z)
    df = 1.0 / r
    v = stdtr(df, z)
    bad = v <= 0.0
    with np.errstate(divide="ignore"):
        out = np.log(v)
    if np.any(bad):
        out = np.where(bad, t_dist.logcdf(np.where(bad, z, 0.0), df), out)
    return out


def robit_z(p: float, r):
    """t^{-1}_{1/r}(p), vectorised over r, with r = 0 giving Phi^{-1}(p).

    This is the whole of the tail parameter's effect on a quantile estimand:
    ``q_p = mu + sigma * robit_z(p, r)``.  At p = 0.99 it runs from 2.326 at
    r = 0 to 6.965 at r = 0.5, which is why an unidentified r widens q_0.99
    while barely touching q_0.5 (where it is exactly 0 for every r).
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    out = np.empty(r.shape, dtype=float)
    zero = r <= 0.0
    out[zero] = float(norm.ppf(p))
    if np.any(~zero):
        out[~zero] = t_dist.ppf(p, 1.0 / r[~zero])
    return out


def robit_prob_matrix_T(x_grid, mu, sigma, r, dtype=np.float64):
    """(L, N) response probabilities for a particle cloud over (mu, sigma, r).

    Same layout convention as `utilities.response_probability_matrix_T`, so the
    existing mutual-information reductions run over it unchanged.  Particles are
    grouped by their (discrete) r value, so one `t.cdf` call serves every
    particle sharing a degrees-of-freedom, which is what makes a robit design
    affordable.
    """
    x = np.asarray(x_grid, dtype=dtype)
    mu = np.asarray(mu, dtype=dtype)
    sigma = np.asarray(sigma, dtype=dtype)
    r = np.asarray(r, dtype=dtype)
    z = (x[:, None] - mu[None, :]) / sigma[None, :]
    out = np.empty_like(z)
    for rv in np.unique(r):
        col = r == rv
        out[:, col] = _cdf_z(z[:, col], float(rv))
    return np.ascontiguousarray(out, dtype=dtype)


def collapse(x, y):
    """(distinct stimuli ascending, successes, failures).

    The same exact reduction `posterior_grid.aggregated_loglik` and
    `model_check.collapse` perform.  Adaptive designs revisit grid points, so
    this is the difference between n and ~n/3 link evaluations per cell.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y).astype(int)
    xu, inv = np.unique(x, return_inverse=True)
    n1 = np.bincount(inv, weights=(y == 1), minlength=len(xu))
    n0 = np.bincount(inv, weights=(y == 0), minlength=len(xu))
    return xu, n1, n0


# The z matrix is shared across every r node, so the working set is
# (block cells) x (distinct stimuli).  Bounded for the same reason
# `posterior_grid.aggregated_loglik` blocks: a 513^2 grid against 30 distinct
# stimuli is 8 x 10^6 doubles per temporary, and thrashing costs more than the
# arithmetic.
LOGLIK_BLOCK_CELLS = 4_000_000


def robit_aggregated_loglik(xu, n1, n0, mu, sigma, r_nodes, block=LOGLIK_BLOCK_CELLS):
    """(K, n_r) total log-likelihood over K (mu, sigma) cells and n_r tail nodes.

    ``z = (x - mu)/sigma`` does not depend on r, so it is formed once per block
    and reused for every tail node.  That is the only structure worth
    exploiting here: the cost is n_r link evaluations of the same z matrix.
    """
    xu = np.asarray(xu, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    r_nodes = np.asarray(r_nodes, dtype=float)
    n1 = np.asarray(n1, dtype=float)
    n0 = np.asarray(n0, dtype=float)
    # Stimuli with a zero count are dropped rather than weighted by zero.  A
    # Student-t CDF *can* underflow to 0 in the far tail where `log_ndtr`
    # cannot, so `lq @ n0` would form -inf * 0 = NaN and poison an otherwise
    # perfectly finite cell.  This is not a floor: the surviving -inf entries,
    # where the count is genuinely positive, stay -inf.
    i1 = n1 > 0
    i0 = n0 > 0
    out = np.empty((mu.shape[0], r_nodes.shape[0]), dtype=float)
    step = max(1, int(block // max(len(xu), 1)))
    for s in range(0, len(mu), step):
        e = min(s + step, len(mu))
        z = (xu[None, :] - mu[s:e, None]) / sigma[s:e, None]
        for j, rv in enumerate(r_nodes):
            acc = np.zeros(e - s)
            if np.any(i1):
                acc = acc + _logcdf(z[:, i1], float(rv)) @ n1[i1]
            if np.any(i0):
                acc = acc + _logcdf(-z[:, i0], float(rv)) @ n0[i0]
            out[s:e, j] = acc
    return out


# ---------------------------------------------------------------------------
# Priors on the tail parameter
# ---------------------------------------------------------------------------
#
# r is nearly unidentified at the horizons this project runs (Phase 4A), so the
# prior on it is not a nuisance detail -- it is doing most of the work.  A
# conclusion that holds under one r-prior and not the others is a statement
# about the prior, not about the safeguard.  Three are therefore implemented and
# **every** result is reported under all three; the choice is recorded as
# `docs/decisions.md` DEC-14 rather than left as a tuning knob.


@dataclass(frozen=True)
class RPrior:
    """Beta(a, b) on u = r / r_max, or a point mass at r = 0.

    `degenerate` is not a modelling option.  It exists so that the
    three-parameter machinery can be driven back onto the two-parameter one and
    checked against it exactly (`tests/test_robit_nesting.py`).
    """

    name: str
    a: float = 1.0
    b: float = 1.0
    r_max: float = R_MAX
    degenerate: bool = False

    def logpdf(self, r):
        r = np.asarray(r, dtype=float)
        if self.degenerate:
            return np.where(r == 0.0, 0.0, -np.inf)
        u = r / self.r_max
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = beta_dist.logpdf(u, self.a, self.b) - np.log(self.r_max)
        return np.where((r >= 0.0) & (r <= self.r_max), lp, -np.inf)

    def nodes(self, n_r: int):
        """Grid nodes on [0, r_max].

        r = 0 is always a node, so the probit is represented **exactly** rather
        than approached.  A degenerate prior collapses to the single node r = 0,
        which drives the whole grid back onto the two-parameter one with no
        residual r-axis at all -- that is what makes the nesting invariant an
        identity in floating point and not a limit.
        """
        if self.degenerate:
            return np.array([0.0])
        return np.linspace(0.0, self.r_max, n_r)

    def describe(self) -> dict:
        return {"r_prior": self.name, "a": self.a, "b": self.b,
                "r_max": self.r_max, "degenerate": self.degenerate,
                "prior_mean_r": (0.0 if self.degenerate
                                 else self.r_max * self.a / (self.a + self.b))}


# Spread-out reference: flat on the chosen parameterisation, so it makes no
# statement about the tail beyond the nu >= 2 bound.  Prior mean nu ~ 4.6.
R_REFERENCE = RPrior("reference_uniform", 1.0, 1.0)

# Concentrated near the probit: 90% of its mass below r = 0.115 (nu > 8.7).
# The sceptical setting -- an analyst who believes the probit but declines to
# assume it with certainty.  Density at r = 0 is finite and positive.
R_NEAR_PROBIT = RPrior("near_probit", 1.0, 9.0)

# Favouring heavy tails: prior mean r = 0.375 (nu ~ 2.7).  Density vanishes at
# r = 0, so this prior does not merely doubt the probit, it excludes it.
R_HEAVY = RPrior("heavy_tail", 3.0, 1.0)

# Not a modelling choice; see the dataclass note.
R_DEGENERATE = RPrior("degenerate_probit", degenerate=True)

R_PRIORS = {p.name: p for p in (R_REFERENCE, R_NEAR_PROBIT, R_HEAVY, R_DEGENERATE)}

# The three reported in every table, in the order they appear in the protocol.
R_PRIOR_PANEL = ("reference_uniform", "near_probit", "heavy_tail")
