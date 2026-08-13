"""High-accuracy deterministic reference posterior on (mu, eta = log sigma).

This is the *reference* the plan asks for.  It is deliberately not a Monte
Carlo method: for a two-parameter smooth model a refined tensor grid with
log-sum-exp normalisation reaches accuracy that no particle method with
N = 10^4 can match, which is exactly what makes it usable as ground truth when
auditing Rotem's weighted-particle posterior.

The class is only trusted after `refine()` has demonstrated convergence
against a denser grid; `GridPosterior.convergence` records the evidence and
`converged` is False if the tolerance was not met, so a caller can never
silently use an unconverged reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import log_ndtr, logsumexp
from scipy.stats import norm

from .response_models import probit_loglik


@dataclass
class GridSpec:
    mu_lo: float
    mu_hi: float
    eta_lo: float
    eta_hi: float
    n_mu: int = 385
    n_eta: int = 385

    def describe(self):
        return {
            "mu_lo": self.mu_lo,
            "mu_hi": self.mu_hi,
            "eta_lo": self.eta_lo,
            "eta_hi": self.eta_hi,
            "n_mu": self.n_mu,
            "n_eta": self.n_eta,
        }


class GridPosterior:
    """Normalised posterior over a regular (mu, eta) grid.

    Attributes
    ----------
    mu, sigma : (K,) flattened parameter values at cell centres
    logw      : (K,) normalised log posterior weights (log-sum-exp to 0)
    """

    def __init__(self, mu, sigma, logw, spec: GridSpec, log_evidence: float):
        self.mu = mu
        self.sigma = sigma
        self.logw = logw
        self.w = np.exp(logw)
        self.spec = spec
        self.log_evidence = log_evidence
        self.convergence: dict = {}
        self.converged: bool | None = None

    # -- construction -----------------------------------------------------
    @classmethod
    def build(cls, prior, x, y, spec: GridSpec):
        mu_c = np.linspace(spec.mu_lo, spec.mu_hi, spec.n_mu)
        eta_c = np.linspace(spec.eta_lo, spec.eta_hi, spec.n_eta)
        MU, ETA = np.meshgrid(mu_c, eta_c, indexing="ij")
        mu_flat = MU.ravel()
        sig_flat = np.exp(ETA.ravel())

        # prior density in (mu, eta): p(mu, sigma) * |d sigma / d eta| = p * sigma
        logpri = prior.logpdf(mu_flat, sig_flat) + np.log(sig_flat)

        loglik = np.zeros_like(logpri)
        if len(x) > 0:
            finite = np.isfinite(logpri)
            loglik[finite] = aggregated_loglik(x, y, mu_flat[finite], sig_flat[finite])
            loglik[~finite] = 0.0

        logpost = logpri + loglik
        d_mu = (spec.mu_hi - spec.mu_lo) / (spec.n_mu - 1)
        d_eta = (spec.eta_hi - spec.eta_lo) / (spec.n_eta - 1)
        log_evidence = logsumexp(logpost) + np.log(d_mu * d_eta)
        logw = logpost - logsumexp(logpost)
        return cls(mu_flat, sig_flat, logw, spec, float(log_evidence))

    # -- summaries --------------------------------------------------------
    def values(self, quantity: str):
        """Flattened values of a derived quantity at every grid cell."""
        if quantity == "mu":
            return self.mu
        if quantity == "sigma":
            return self.sigma
        if quantity.startswith("q"):
            p = float(quantity[1:])
            return self.mu + self.sigma * norm.ppf(p)
        raise ValueError(f"unknown quantity {quantity!r}")

    def mean(self, quantity: str) -> float:
        return float(np.dot(self.w, self.values(quantity)))

    def sd(self, quantity: str) -> float:
        v = self.values(quantity)
        m = float(np.dot(self.w, v))
        return float(np.sqrt(max(np.dot(self.w, (v - m) ** 2), 0.0)))

    # -- CDF and quantiles of a derived quantity ---------------------------
    #
    # A cell is treated as a uniform block of probability mass over its
    # rectangle in (mu, eta), which is the piecewise-constant density that the
    # midpoint rule already assumes.  Counting whole cells instead
    # (sum of w over cells whose *centre* falls below the threshold) leaves an
    # O(h) lattice error that dominates everything else on this grid, because
    # the posterior for sigma is heavy tailed and forces a wide box.  Resolving
    # the partial cell restores O(h^2) and is what lets a 513^2 grid be a
    # genuine reference.

    def _value_gradient(self, quantity: str):
        """d value / d(mu, eta) at each cell centre."""
        if quantity == "mu":
            return np.ones_like(self.mu), np.zeros_like(self.mu)
        if quantity == "sigma":
            return np.zeros_like(self.mu), self.sigma
        if quantity.startswith("q"):
            z = norm.ppf(float(quantity[1:]))
            return np.ones_like(self.mu), self.sigma * z
        raise ValueError(f"unknown quantity {quantity!r}")

    def _cell_extents(self):
        d_mu = (self.spec.mu_hi - self.spec.mu_lo) / (self.spec.n_mu - 1)
        d_eta = (self.spec.eta_hi - self.spec.eta_lo) / (self.spec.n_eta - 1)
        return d_mu, d_eta

    def cdf_at(self, quantity: str, value: float) -> float:
        """P(Q < value | data).  The SBC rank statistic when value is the truth."""
        v = self.values(quantity)
        gmu, geta = self._value_gradient(quantity)
        d_mu, d_eta = self._cell_extents()
        a = np.abs(gmu) * d_mu
        b = np.abs(geta) * d_eta
        return float(np.dot(self.w, _trapezoid_cdf(value - v, a, b)))

    def quantile(self, quantity: str, p) -> float | np.ndarray:
        """Inverse of the smoothed CDF, by bisection on a bracketing interval."""
        scalar = np.isscalar(p)
        ps = np.atleast_1d(p).astype(float)
        v = self.values(quantity)
        sd = self.sd(quantity)
        out = np.empty(len(ps))
        for i, pi in enumerate(ps):
            start = float(weighted_quantile(v, self.w, pi))
            out[i] = self._invert_cdf(quantity, pi, start, max(sd, 1e-9))
        return float(out[0]) if scalar else out

    def _invert_cdf(self, quantity, p, start, sd, n_iter=14, rtol=1e-3):
        """Bisection from the unsmoothed quantile.

        Each CDF evaluation is a weighted sum over every grid cell, so the
        iteration count is the whole cost of a quantile.  `rtol` is relative to
        the posterior sd: 1e-4 of a posterior standard deviation is far finer
        than anything downstream can resolve, and needs ~10 bisection steps
        from the default bracket instead of the ~40 a 1e-6 target would.
        """
        # The unsmoothed weighted quantile is already within a cell or two of
        # the answer, so a +/- 0.25 sd bracket almost always contains it on the
        # first try.  Starting at 0.05 sd instead made the widening loop, not
        # the bisection, the dominant cost of every interval.
        lo, hi = start - 0.25 * sd, start + 0.25 * sd
        for _ in range(20):                       # widen until bracketed
            if self.cdf_at(quantity, lo) <= p <= self.cdf_at(quantity, hi):
                break
            lo -= 0.5 * sd
            hi += 0.5 * sd
        tol = rtol * sd
        for _ in range(n_iter):
            if hi - lo < tol:
                break
            mid = 0.5 * (lo + hi)
            if self.cdf_at(quantity, mid) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def credible_interval(self, quantity: str, level: float = 0.95):
        a = (1.0 - level) / 2.0
        lo, hi = self.quantile(quantity, [a, 1.0 - a])
        return float(lo), float(hi)

    def covered(self, quantity: str, truth: float, level: float = 0.95) -> bool:
        """Is `truth` inside the central credible interval?

        Equivalent to a/2 < P(Q < truth | data) < 1 - a/2, so coverage is read
        straight off the rank statistic.  This costs one CDF evaluation instead
        of two root solves, and -- more importantly -- makes the coverage layer
        and the SBC layer exactly consistent by construction, so a discrepancy
        between them can never be an artefact of two different conventions.
        """
        a = (1.0 - level) / 2.0
        r = self.cdf_at(quantity, truth)
        return bool(a < r < 1.0 - a)

    def sample(self, n: int, rng: np.random.Generator):
        """Draw from the normalised grid, jittering uniformly within cells."""
        idx = rng.choice(len(self.w), size=n, p=self.w / self.w.sum())
        d_mu = (self.spec.mu_hi - self.spec.mu_lo) / (self.spec.n_mu - 1)
        d_eta = (self.spec.eta_hi - self.spec.eta_lo) / (self.spec.n_eta - 1)
        mu = self.mu[idx] + rng.uniform(-0.5, 0.5, n) * d_mu
        eta = np.log(self.sigma[idx]) + rng.uniform(-0.5, 0.5, n) * d_eta
        return mu, np.exp(eta)

    def boundary_mass(self) -> float:
        """Posterior mass sitting on the outermost ring of cells."""
        W = self.w.reshape(self.spec.n_mu, self.spec.n_eta)
        edge = W[0, :].sum() + W[-1, :].sum() + W[:, 0].sum() + W[:, -1].sum()
        edge -= W[0, 0] + W[0, -1] + W[-1, 0] + W[-1, -1]
        return float(edge)


def _trapezoid_cdf(d, a, b):
    """P(a*U1 + b*U2 < d) for independent U1, U2 ~ Uniform(-1/2, 1/2).

    The exact area fraction of a rectangular cell of side lengths (a, b) lying
    below a linear threshold at signed distance d from the cell centre.
    Degenerates correctly to a linear ramp when one side is zero and to a step
    when both are.
    """
    d = np.asarray(d, dtype=float)
    A = np.maximum(a, b) / 2.0
    B = np.minimum(a, b) / 2.0
    out = np.where(d > 0, 1.0, 0.0)

    ramp = A > 0
    if np.any(ramp):
        # B == 0: single uniform, linear ramp over [-A, A]
        lin = np.clip((d + A) / np.where(A > 0, 2 * A, 1.0), 0.0, 1.0)
        out = np.where(ramp, lin, out)

    trap = (B > 0) & (A > 0)
    if np.any(trap):
        denom = np.where(trap, 8.0 * A * B, 1.0)
        left = (d + A + B) ** 2 / denom
        right = 1.0 - (A + B - d) ** 2 / denom
        mid = (d + A) / np.where(A > 0, 2 * A, 1.0)
        val = np.where(d <= -(A - B), left, np.where(d >= (A - B), right, mid))
        val = np.where(d <= -(A + B), 0.0, np.where(d >= (A + B), 1.0, val))
        out = np.where(trap, val, out)
    return np.clip(out, 0.0, 1.0)


LOGLIK_BLOCK_CELLS = 4_000_000


def aggregated_loglik(x, y, mu, sigma, block=LOGLIK_BLOCK_CELLS):
    """Total log-likelihood, collapsing repeated stimuli into counts.

    Adaptive designs revisit the same grid point many times, so the distinct
    stimuli are typically far fewer than n.  Collapsing to
    (x_u, n_u1, n_u0) cuts the dominant cost of the reference posterior by an
    order of magnitude without changing a single digit of the result.

    The (cells x distinct-stimuli) intermediate is materialised in blocks of at
    most `block` entries.  A broad, non-adaptive design at n = 100 puts ~100
    distinct stimuli on a 1025^2 grid, which is 8 x 10^8 doubles -- several
    gigabytes of temporaries, and the resulting thrashing cost more than the
    arithmetic.  Blocking bounds the working set; the arithmetic is unchanged.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y).astype(int)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    xu, inv = np.unique(x, return_inverse=True)
    n1 = np.bincount(inv, weights=(y == 1), minlength=len(xu))
    n0 = np.bincount(inv, weights=(y == 0), minlength=len(xu))

    out = np.empty(mu.shape[0], dtype=float)
    step = max(1, int(block // max(len(xu), 1)))
    for s in range(0, len(mu), step):
        e = min(s + step, len(mu))
        z = (xu[None, :] - mu[s:e, None]) / sigma[s:e, None]
        out[s:e] = log_ndtr(z) @ n1
        np.negative(z, out=z)
        out[s:e] += log_ndtr(z) @ n0
    return out


def weighted_quantile(values, weights, probs):
    """Interpolated weighted quantile(s); `probs` may be scalar or array."""
    order = np.argsort(values, kind="stable")
    v = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    cw = np.cumsum(w)
    total = cw[-1]
    if total <= 0:
        raise ValueError("weights sum to zero")
    # mid-point cumulative probabilities give an unbiased, monotone rule
    cp = (cw - 0.5 * w) / total
    scalar = np.isscalar(probs)
    p = np.atleast_1d(probs).astype(float)
    out = np.interp(p, cp, v)
    return float(out[0]) if scalar else out


def lower_inverse_cdf(values, weights, probs):
    """Rotem's estimator convention: max{j : sum_{u<=j} w_(u) <= p}.

    Returns an actual particle/grid value rather than an interpolant.  Used
    wherever a number is compared against Rotem's published tables, so that a
    difference cannot be blamed on a different quantile convention.
    """
    order = np.argsort(values, kind="stable")
    v = np.asarray(values)[order]
    cw = np.cumsum(np.asarray(weights)[order])
    cw = cw / cw[-1]
    scalar = np.isscalar(probs)
    p = np.atleast_1d(probs).astype(float)
    idx = np.searchsorted(cw, p, side="right")
    idx = np.clip(idx, 0, len(v) - 1)
    out = v[idx]
    return float(out[0]) if scalar else out


# --------------------------------------------------------------------------
# Adaptive refinement
# --------------------------------------------------------------------------

DEFAULT_TRACKED = ("mu", "sigma", "q0.5", "q0.95", "q0.99")


def initial_spec(prior, n_mu=385, n_eta=385, width=6.0) -> GridSpec:
    """A box wide enough to contain essentially all prior mass."""
    s = prior.spec
    mu_lo = max(1e-6, s.mu0 - width * s.sigma_mu)
    mu_hi = s.mu0 + width * s.sigma_mu
    if s.dependent:
        eta_mid = np.log(s.mu0) + s.log_alpha0
        eta_sd = np.sqrt(s.tau_scale**2 + (s.sigma_mu / s.mu0) ** 2)
    else:
        eta_mid = s.sigma_log_location
        eta_sd = s.tau_scale
    return GridSpec(mu_lo, mu_hi, eta_mid - width * eta_sd, eta_mid + width * eta_sd,
                    n_mu, n_eta)


def build_reference_posterior(
    prior,
    x,
    y,
    *,
    tol: float = 1e-3,
    n_start: int = 257,
    n_max: int = 1025,
    n_localise: int = 129,
    boundary_tol: float = 1e-10,
    tracked=DEFAULT_TRACKED,
    return_history: bool = False,
    fixed_n: int | None = None,
):
    """Adaptively refined reference posterior.

    Strategy
    --------
    1. Localise on a deliberately coarse grid: shrink to the smallest box
       carrying all but `boundary_tol` of the mass in each margin, padded, and
       stop as soon as the box stops moving.  Localisation only needs to find
       the box, not resolve the posterior, so doing it at n_localise instead of
       the final resolution is roughly an order of magnitude cheaper.
    2. Refine: double the resolution until every tracked summary moves by less
       than `tol`, and until the log evidence moves by less than `tol`.

    Convergence is tracked on the mean, the sd, and the CDF at three fixed
    thresholds per quantity rather than on quantiles.  A CDF evaluation is a
    single weighted sum, whereas a quantile needs a bisection over many such
    sums; the two converge at the same rate, so this buys accuracy assessment
    at a fraction of the cost.  Thresholds are frozen after the first round so
    that successive rounds are compared at the same points.

    The default tol = 1e-3 means every reported reference summary is stable to
    a thousandth of a posterior standard deviation.  That is one to two orders
    of magnitude below the Monte Carlo error of any experiment in the plan, so
    the reference contributes negligibly to the reported uncertainty, while
    staying cheap enough to run inside a several-thousand-replicate SBC loop.
    `tests/test_posterior_grid.py::test_high_accuracy_reference_converges`
    exercises the slower tol = 1e-4 setting.
    """
    spec = initial_spec(prior, n_mu=n_localise, n_eta=n_localise)
    post = GridPosterior.build(prior, x, y, spec)
    for _ in range(4):
        new_spec = _shrink_to_mass(post, boundary_tol)
        stable = _box_is_stable(spec, new_spec)
        spec = new_spec
        post = GridPosterior.build(prior, x, y, spec)
        if stable and post.boundary_mass() < boundary_tol:
            break

    thresholds = _threshold_set(post, tracked)

    if fixed_n is not None:
        # Production mode: localise, then build once at a resolution whose
        # adequacy has been established separately on an audit sample (see
        # `experiments/phase3_reference_audit.py`).  Halves the cost of every
        # reference by skipping the doubling ladder.  `converged` is set to
        # None so that a caller can always tell a validated-resolution build
        # from one that demonstrated its own convergence.
        spec = GridSpec(spec.mu_lo, spec.mu_hi, spec.eta_lo, spec.eta_hi,
                        fixed_n, fixed_n)
        post = GridPosterior.build(prior, x, y, spec)
        post.converged = None
        post.convergence = {"tol": tol, "final_n": fixed_n, "mode": "fixed",
                            "boundary_mass": post.boundary_mass(),
                            "max_relative_delta": float("nan"), "rounds": 1}
        return (post, []) if return_history else post

    spec = GridSpec(spec.mu_lo, spec.mu_hi, spec.eta_lo, spec.eta_hi,
                    n_start, n_start)
    post = GridPosterior.build(prior, x, y, spec)
    history = [_summarise(post, tracked, thresholds)]
    converged = False
    n = n_start
    while n < n_max:
        n_next = min(2 * (n - 1) + 1, n_max)
        spec = GridSpec(spec.mu_lo, spec.mu_hi, spec.eta_lo, spec.eta_hi,
                        n_next, n_next)
        post_next = GridPosterior.build(prior, x, y, spec)
        summ_next = _summarise(post_next, tracked, thresholds)
        deltas = _deltas(history[-1], summ_next, tracked, thresholds)
        history.append(summ_next)
        post, n = post_next, n_next
        if max(deltas.values()) < tol:
            converged = True
            break

    post.converged = converged
    post.convergence = {
        "tol": tol,
        "final_n": n,
        "boundary_mass": post.boundary_mass(),
        "max_relative_delta": (
            max(_deltas(history[-2], history[-1], tracked, thresholds).values())
            if len(history) > 1 else float("nan")
        ),
        "rounds": len(history),
    }
    if return_history:
        return post, history
    return post


def _box_is_stable(a: GridSpec, b: GridSpec, rtol: float = 0.02) -> bool:
    wa = max(a.mu_hi - a.mu_lo, 1e-12)
    wb = max(a.eta_hi - a.eta_lo, 1e-12)
    return (
        abs(b.mu_lo - a.mu_lo) < rtol * wa
        and abs(b.mu_hi - a.mu_hi) < rtol * wa
        and abs(b.eta_lo - a.eta_lo) < rtol * wb
        and abs(b.eta_hi - a.eta_hi) < rtol * wb
    )


def _threshold_set(post: GridPosterior, tracked):
    """Three fixed probes per quantity, at the posterior mean and +/- 1.5 sd."""
    out = {}
    for q in tracked:
        m, s = post.mean(q), max(post.sd(q), 1e-9)
        out[q] = (m - 1.5 * s, m, m + 1.5 * s)
    return out


def _shrink_to_mass(post: GridPosterior, tail: float) -> GridSpec:
    """Smallest box holding 1 - tail of each marginal, padded by 25%."""
    W = post.w.reshape(post.spec.n_mu, post.spec.n_eta)
    mu_c = np.linspace(post.spec.mu_lo, post.spec.mu_hi, post.spec.n_mu)
    eta_c = np.linspace(post.spec.eta_lo, post.spec.eta_hi, post.spec.n_eta)

    def bounds(centres, marginal):
        cw = np.cumsum(marginal)
        cw /= cw[-1]
        lo = centres[max(int(np.searchsorted(cw, tail / 2)) - 1, 0)]
        hi = centres[min(int(np.searchsorted(cw, 1 - tail / 2)) + 1, len(centres) - 1)]
        if hi <= lo:
            lo, hi = centres[0], centres[-1]
        pad = 0.25 * (hi - lo)
        return lo - pad, hi + pad

    mu_lo, mu_hi = bounds(mu_c, W.sum(axis=1))
    eta_lo, eta_hi = bounds(eta_c, W.sum(axis=0))
    mu_lo = max(mu_lo, 1e-8)
    return GridSpec(mu_lo, mu_hi, eta_lo, eta_hi, post.spec.n_mu, post.spec.n_eta)


def _summarise(post: GridPosterior, tracked, thresholds):
    out = {"log_evidence": post.log_evidence}
    for q in tracked:
        out[f"{q}:mean"] = post.mean(q)
        out[f"{q}:sd"] = post.sd(q)
        for k, t in enumerate(thresholds[q]):
            out[f"{q}:cdf{k}"] = post.cdf_at(q, t)
    return out


def _deltas(a, b, tracked, thresholds):
    """Mean/sd changes scaled by the posterior sd; CDF changes in probability."""
    out = {"log_evidence": abs(b["log_evidence"] - a["log_evidence"])}
    for q in tracked:
        scale = max(b[f"{q}:sd"], 1e-12)
        out[f"{q}:mean"] = abs(b[f"{q}:mean"] - a[f"{q}:mean"]) / scale
        out[f"{q}:sd"] = abs(b[f"{q}:sd"] - a[f"{q}:sd"]) / scale
        for k in range(len(thresholds[q])):
            key = f"{q}:cdf{k}"
            out[key] = abs(b[key] - a[key])
    return out
