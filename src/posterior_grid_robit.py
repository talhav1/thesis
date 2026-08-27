"""Reference posterior on (mu, eta = log sigma, r) for the robit fitted family.

The three-parameter analogue of `posterior_grid`, and **strictly additive**:
`GridPosterior` is not imported for anything but its exact cell-smoothing
kernel, and no existing code path is altered.  Everything the two-parameter
reference guarantees is carried over -- normalisation by log-sum-exp, midpoint
quadrature on a regular grid, cell-resolved CDFs, a localisation pass, a
refinement ladder with recorded convergence evidence, and a boundary-mass
diagnostic -- with three changes forced by the extra dimension.

**1.  Cell smoothing is a three-fold uniform convolution.**  The two-parameter
code resolves the partial cell exactly, treating a cell as a uniform block of
mass over its rectangle, because counting whole cells leaves an O(h) lattice
error that dominates on a box wide enough for a heavy-tailed sigma margin.  The
same argument applies here with one more axis, so `_uniform_sum_cdf` implements
the generalised Irwin-Hall CDF for a sum of independent uniforms of unequal
widths.  It reduces to `posterior_grid._trapezoid_cdf` when one width is zero
and to a step when two are, and `tests/test_robit_grid.py` checks both
degenerations against the existing kernel.

**2.  The r boundaries are prior boundaries, not tails.**  Mass at r = 0 means
the data prefer the probit; mass at r = 0.5 means they prefer nu = 2.  Neither
is evidence that the box is too small, which is what boundary mass means on the
mu and eta axes.  `boundary_mass()` is therefore computed over the (mu, eta)
faces only -- the same quantity DEC-7 records -- and the r-edge mass is reported
separately as `r_edge_mass`, as information rather than as a failure.

**3.  Localisation runs on the three-parameter posterior itself.**  Localising
on the probit posterior and padding would be cheaper, but under a heavy-tail
prior the (mu, sigma) posterior genuinely moves, and a box chosen by a model
that is not the one being integrated is a box whose adequacy is assumed rather
than measured.

Cost.  A three-parameter grid at equal per-axis resolution is n_r times the
two-parameter one.  At the Phase 6 production setting (fixed_n = 257, n_r = 33,
chosen by `experiments/phase6_robit_pilot.resolution_audit` and recorded in
`manuscript/phase6_protocol.md` section 7) a fit costs about 12x the 513^2
two-parameter reference over the same three targets -- not the two orders of
magnitude equal per-axis resolution would imply.  Interval inversion dominates:
`_invert_cdf` calls `cdf_at` a few hundred times per replicate, which is why
`_smoothing_plan` hoists the per-cell widths out of that loop and keeps one
quantity's plan at a time.  The r-axis resolution is chosen from refinement
evidence, never asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .posterior_grid import _trapezoid_cdf, weighted_quantile
from .robit import R_MAX, RPrior, collapse, robit_aggregated_loglik, robit_z

DEFAULT_TRACKED = ("mu", "sigma", "q0.5", "q0.95", "q0.99")

# A width below this fraction of the largest is treated as exactly zero in the
# smoothing kernel.  The cubic branch divides a difference of cubes by the
# product of the three widths, so a width ratio of tau costs ~eps/tau in
# round-off while dropping it costs ~tau in probability; 1e-6 balances the two
# at ~1e-6, four orders below the 1e-3 refinement tolerance.  It is never
# reached on the nesting path, where the r-axis width is *exactly* zero.
WIDTH_EPS = 1e-6


@dataclass
class RobitGridSpec:
    mu_lo: float
    mu_hi: float
    eta_lo: float
    eta_hi: float
    n_mu: int = 257
    n_eta: int = 257
    n_r: int = 17
    r_max: float = R_MAX

    def describe(self):
        return {"mu_lo": self.mu_lo, "mu_hi": self.mu_hi,
                "eta_lo": self.eta_lo, "eta_hi": self.eta_hi,
                "n_mu": self.n_mu, "n_eta": self.n_eta,
                "n_r": self.n_r, "r_max": self.r_max}


class RobitGridPosterior:
    """Normalised posterior over a regular (mu, eta, r) grid.

    The public surface deliberately matches `posterior_grid.GridPosterior`
    -- `mean`, `sd`, `cdf_at`, `quantile`, `credible_interval`, `covered`,
    `sample`, `boundary_mass` -- so a caller can hold either object and every
    metric downstream keeps its definition.  In particular `covered` reads the
    rank statistic (DEC-3), so the robit and probit coverage numbers are formed
    the same way and a gap between them is a difference in the posterior, never
    in the convention.
    """

    def __init__(self, mu, sigma, r, logw, spec: RobitGridSpec, log_evidence: float,
                 r_prior_name: str = ""):
        self.mu = mu
        self.sigma = sigma
        self.r = r
        self.logw = logw
        self.w = np.exp(logw)
        self.spec = spec
        self.log_evidence = log_evidence
        self.r_prior_name = r_prior_name
        self.convergence: dict = {}
        self.converged: bool | None = None
        # Derived arrays (quantile maps, smoothing plans) for the quantity most
        # recently asked for.  The grid is immutable once built, so nothing here
        # can go stale; it is bounded rather than growing because the arrays are
        # grid-sized and callers walk the targets in sequence.
        self._cache: dict = {}

    # -- construction -----------------------------------------------------
    @classmethod
    def build(cls, prior, r_prior: RPrior, x, y, spec: RobitGridSpec):
        mu_c = np.linspace(spec.mu_lo, spec.mu_hi, spec.n_mu)
        eta_c = np.linspace(spec.eta_lo, spec.eta_hi, spec.n_eta)
        r_c = r_prior.nodes(spec.n_r)

        MU, ETA = np.meshgrid(mu_c, eta_c, indexing="ij")
        mu2 = MU.ravel()
        sig2 = np.exp(ETA.ravel())

        # prior in (mu, eta, r): p(mu, sigma) * |d sigma / d eta| * p(r)
        logpri2 = prior.logpdf(mu2, sig2) + np.log(sig2)
        logpri_r = r_prior.logpdf(r_c) + _r_quadrature_logweights(len(r_c))

        loglik = np.zeros((len(mu2), len(r_c)))
        if len(x) > 0:
            finite = np.isfinite(logpri2)
            xu, n1, n0 = collapse(x, y)
            loglik[finite] = robit_aggregated_loglik(
                xu, n1, n0, mu2[finite], sig2[finite], r_c
            )

        logpost = logpri2[:, None] + logpri_r[None, :] + loglik
        logpost = np.where(np.isfinite(logpost), logpost, -np.inf).ravel()

        d_mu, d_eta, d_r = _extents(spec, len(r_c))
        # A single r node is a point mass, not an integral: its volume element
        # is 1, not a spacing.  (Writing `max(d_r, 1.0)` here made the log
        # evidence drift by exactly log 2 per refinement round, which the
        # refinement ladder then reported as non-convergence.)
        log_evidence = logsumexp(logpost) + np.log(d_mu * d_eta * (d_r or 1.0))
        logw = logpost - logsumexp(logpost)

        mu_flat = np.repeat(mu2, len(r_c))
        sig_flat = np.repeat(sig2, len(r_c))
        r_flat = np.tile(r_c, len(mu2))
        spec = RobitGridSpec(spec.mu_lo, spec.mu_hi, spec.eta_lo, spec.eta_hi,
                             spec.n_mu, spec.n_eta, len(r_c), spec.r_max)
        return cls(mu_flat, sig_flat, r_flat, logw, spec, float(log_evidence),
                   r_prior.name)

    # -- summaries --------------------------------------------------------
    def values(self, quantity: str):
        """q_p = mu + sigma * t^{-1}_{1/r}(p); mu and sigma are the fitted pair.

        ``t^{-1}`` is evaluated on the ``n_r`` distinct nodes and tiled, not on
        all ``n_mu * n_eta * n_r`` cells.  The r axis is a *shared* set of nodes
        by construction, so this is an identity, not an approximation -- and it
        is the difference between 0.22 s and 0.01 s per call on a 257^2 x 17
        grid, which matters because the interval bisection calls `cdf_at` a few
        hundred times per replicate.
        """
        if quantity == "mu":
            return self.mu
        if quantity == "sigma":
            return self.sigma
        if quantity.startswith("q"):
            return self.mu + self.sigma * self._z_flat(float(quantity[1:]))
        raise ValueError(f"unknown quantity {quantity!r}")

    def _z_flat(self, p: float):
        """t^{-1}_{1/r}(p) at every cell, from the n_r distinct node values."""
        key = ("z", p)
        hit = self._cache.get(key)
        if hit is None:
            n_r = self.spec.n_r
            z = robit_z(p, self.r[:n_r])
            hit = np.tile(z, len(self.mu) // n_r)
            self._cache[key] = hit
        return hit

    def mean(self, quantity: str) -> float:
        return float(np.dot(self.w, self.values(quantity)))

    def sd(self, quantity: str) -> float:
        v = self.values(quantity)
        m = float(np.dot(self.w, v))
        return float(np.sqrt(max(np.dot(self.w, (v - m) ** 2), 0.0)))

    # -- CDF and quantiles -------------------------------------------------
    def _value_gradient(self, quantity: str):
        """d value / d(mu, eta, r) at each cell centre.

        The r-derivative of a quantile is taken by central difference over the
        r nodes.  ``t^{-1}_{1/r}(p)`` has no elementary derivative in r, and a
        finite difference on the node spacing is the consistent choice: it is
        exactly the slope the piecewise-linear cell model already assumes.
        """
        n_r = self.spec.n_r
        if quantity == "mu":
            return np.ones_like(self.mu), np.zeros_like(self.mu), np.zeros_like(self.mu)
        if quantity == "sigma":
            return np.zeros_like(self.mu), self.sigma, np.zeros_like(self.mu)
        if not quantity.startswith("q"):
            raise ValueError(f"unknown quantity {quantity!r}")
        p = float(quantity[1:])
        r_nodes = self.r[:n_r]
        z = robit_z(p, r_nodes)
        if n_r > 1:
            dz = np.gradient(z, r_nodes)
        else:
            dz = np.zeros_like(z)
        z_flat = np.tile(z, len(self.mu) // n_r)
        dz_flat = np.tile(dz, len(self.mu) // n_r)
        return np.ones_like(self.mu), self.sigma * z_flat, self.sigma * dz_flat

    def _cell_extents(self):
        return _extents(self.spec, self.spec.n_r)

    def cdf_at(self, quantity: str, value: float) -> float:
        """P(Q < value | data).  The rank statistic when `value` is the truth."""
        v, plan = self._smoothing_plan(quantity)
        return float(np.dot(self.w, _uniform_sum_cdf_planned(value - v, plan)))

    def _smoothing_plan(self, quantity: str):
        """Cell values and the per-cell smoothing widths, sorted and branched.

        Neither depends on the probe value, and the grid is immutable, so both
        are formed once per quantity.  The bisection in `_invert_cdf` calls
        `cdf_at` a few hundred times per replicate and each call was otherwise
        re-sorting a (3, n_cells) stack; hoisting it is the single largest cost
        in a three-parameter build.  Nothing about the arithmetic changes --
        `_uniform_sum_cdf` remains the tested entry point and delegates here.
        """
        key = ("plan", quantity)
        hit = self._cache.get(key)
        if hit is None:
            # One quantity's plan at a time.  The plan is eight arrays the size
            # of the grid -- ~90 MB at 257^2 x 33 -- and callers walk the targets
            # strictly in sequence, so holding all of them at once triples the
            # working set of every worker process for no reuse at all.  Keeping
            # the most recent is what a sequential access pattern wants.
            self._cache.clear()
            v = self.values(quantity)
            gmu, geta, gr = self._value_gradient(quantity)
            d_mu, d_eta, d_r = self._cell_extents()
            plan = _plan(np.abs(gmu) * d_mu, np.abs(geta) * d_eta, np.abs(gr) * d_r)
            hit = (v, plan)
            self._cache[key] = hit
        return hit

    def quantile(self, quantity: str, p) -> float | np.ndarray:
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

        Deliberately the *same* bracket, iteration count and relative tolerance
        as `posterior_grid.GridPosterior._invert_cdf`.  D19 records what it cost
        this project when those settings changed inside the commit that produced
        a set of results: the Phase 3A/3B interval widths are permanently
        non-comparable.  Widths are a primary Phase 6 outcome -- the prediction
        is that the robit q_0.99 interval *stops shrinking* -- so probit and
        robit widths have to be formed by an identical solver.
        """
        lo, hi = start - 0.25 * sd, start + 0.25 * sd
        for _ in range(20):
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
        a = (1.0 - level) / 2.0
        r = self.cdf_at(quantity, truth)
        return bool(a < r < 1.0 - a)

    def sample(self, n: int, rng: np.random.Generator):
        idx = rng.choice(len(self.w), size=n, p=self.w / self.w.sum())
        d_mu, d_eta, d_r = self._cell_extents()
        mu = self.mu[idx] + rng.uniform(-0.5, 0.5, n) * d_mu
        eta = np.log(self.sigma[idx]) + rng.uniform(-0.5, 0.5, n) * d_eta
        r = np.clip(self.r[idx] + rng.uniform(-0.5, 0.5, n) * d_r, 0.0, self.spec.r_max)
        return mu, np.exp(eta), r

    # -- diagnostics -------------------------------------------------------
    def _cube(self):
        return self.w.reshape(self.spec.n_mu, self.spec.n_eta, self.spec.n_r)

    def boundary_mass(self) -> float:
        """Mass on the outer (mu, eta) ring, summed over r.

        Deliberately *not* including the r faces: see the module note.  This is
        the same quantity `GridPosterior.boundary_mass` reports, so the DEC-7
        adequacy convention carries over unchanged.
        """
        W = self._cube().sum(axis=2)
        edge = W[0, :].sum() + W[-1, :].sum() + W[:, 0].sum() + W[:, -1].sum()
        edge -= W[0, 0] + W[0, -1] + W[-1, 0] + W[-1, -1]
        return float(edge)

    def r_edge_mass(self) -> tuple[float, float]:
        """(mass at r = 0, mass at r = r_max).

        Substantive, not diagnostic.  Mass piling at r = 0 says the data prefer
        the probit; mass at r_max says they prefer the heaviest tail allowed and
        that the r_max = 0.5 bound is binding, which the write-up must say
        rather than absorb.
        """
        W = self._cube()
        if self.spec.n_r == 1:
            return float(W.sum()), float(W.sum())
        return float(W[:, :, 0].sum()), float(W[:, :, -1].sum())

    def r_summary(self) -> dict:
        """Posterior for the tail parameter itself.

        Phase 4A predicts r is nearly unidentified at n = 50.  The check is
        whether the r posterior has moved from its prior at all; `r_mean` and
        `r_sd` beside the prior's own moments are how that is read.
        """
        lo, hi = self.r_edge_mass()
        return {"r_mean": float(np.dot(self.w, self.r)),
                "r_sd": float(np.sqrt(max(np.dot(self.w, (self.r - np.dot(self.w, self.r)) ** 2), 0.0))),
                "r_mass_at_zero": lo, "r_mass_at_max": hi,
                "r_prior": self.r_prior_name}


# ---------------------------------------------------------------------------
# Exact cell smoothing in three dimensions
# ---------------------------------------------------------------------------


def _uniform_sum_cdf(d, h_a, h_b, h_c):
    """P(sum_i U_i < d) for independent U_i ~ Uniform(-h_i/2, h_i/2).

    The exact volume fraction of a rectangular cell of side lengths (h_a, h_b,
    h_c) lying below a linear threshold at signed distance d from the centre --
    the three-dimensional version of the area fraction
    `posterior_grid._trapezoid_cdf` computes, and the reason the reference
    reaches O(h^2) instead of the O(h) a whole-cell count would leave.

    Implemented as the generalised Irwin-Hall CDF for uniforms of unequal
    widths.  Zero widths are not passed to it: the branch is selected on the
    number of non-negligible widths, so the two- and one-dimensional
    degenerations are taken exactly rather than as a limit of a formula with a
    vanishing denominator.  `WIDTH_EPS` documents the trade-off.
    """
    return _uniform_sum_cdf_planned(d, _plan(h_a, h_b, h_c))


def _plan(h_a, h_b, h_c):
    """Sort the three side lengths and decide each cell's branch, once."""
    H = np.sort(np.stack(np.broadcast_arrays(
        np.abs(np.asarray(h_a, dtype=float)),
        np.abs(np.asarray(h_b, dtype=float)),
        np.abs(np.asarray(h_c, dtype=float)))), axis=0)[::-1]
    h1, h2, h3 = H[0], H[1], H[2]
    big = np.maximum(h1, np.finfo(float).tiny)
    live2 = h2 > WIDTH_EPS * big
    live3 = h3 > WIDTH_EPS * big
    k = (h1 > 0).astype(int) + live2.astype(int) + live3.astype(int)
    # The cubic branch's total width and denominator do not depend on the probe
    # either, so they are formed here too.
    total = h1 + h2 + h3
    den = 6.0 * h1 * np.where(live2, h2, 1.0) * np.where(live3, h3, 1.0)
    return h1, h2, h3, live2, live3, k, total, den


def _uniform_sum_cdf_planned(d, plan):
    """`_uniform_sum_cdf` with the sort and the branch selection already done."""
    h1, h2, h3, live2, live3, k, total, den = plan
    d = np.asarray(d, dtype=float)

    # On a regular grid every cell usually takes the *same* branch, and the
    # masked-where form below then costs several full-array temporaries to
    # select a branch that was never in doubt.  Taking it directly is the same
    # arithmetic on the same values.
    if np.all(k == 3):
        return _cubic_branch(d, h1, h2, h3, total, den)

    out = np.where(d > 0.0, 1.0, 0.0)                        # k = 0: a step

    one = k == 1
    if np.any(one):
        lin = np.clip((d + 0.5 * h1) / np.where(h1 > 0, h1, 1.0), 0.0, 1.0)
        out = np.where(one, lin, out)

    two = k == 2
    if np.any(two):
        out = np.where(two, _trapezoid_cdf(d, h1, np.where(live2, h2, 0.0)), out)

    three = k == 3
    if np.any(three):
        out = np.where(three, _cubic_branch(d, h1, h2, h3, total, den), out)
    return np.clip(out, 0.0, 1.0)


def _cubic_branch(d, h1, h2, h3, total, den):
    """The three-live-width case of the generalised Irwin-Hall CDF."""
    s = d + 0.5 * total

    def cube(t):
        return np.clip(t, 0.0, None) ** 3

    val = (cube(s) - cube(s - h1) - cube(s - h2) - cube(s - h3)
           + cube(s - h1 - h2) + cube(s - h1 - h3) + cube(s - h2 - h3)
           - cube(s - h1 - h2 - h3)) / den
    # Outside the support the alternating sum is a difference of cubes of order
    # s^3 that must telescope to exactly 6 h1 h2 h3; for |d| large it cancels to
    # noise and clipping cannot recover it (the rank statistic at a far-away
    # probe came back as 0.32 rather than 1).  Both tails are known in closed
    # form, so they are taken directly.
    val = np.where(s <= 0.0, 0.0, np.where(s >= total, 1.0, val))
    return np.clip(val, 0.0, 1.0)


def _r_quadrature_logweights(n_r: int):
    """Trapezoid weights on the r axis: 1/2 at the two endpoints, 1 inside.

    The mu and eta axes use the midpoint/rectangle rule and are second-order
    there because the posterior density vanishes at the box edges by
    construction -- that is exactly what `boundary_mass` < 1e-10 certifies.  The
    r axis is different: [0, r_max] is a *hard* prior boundary and the density
    at r = 0 is finite and positive under the reference and near-probit priors,
    so a rectangle rule carries a first-order endpoint error.

    Measured: with rectangle weights the q_0.99 posterior mean moved 0.245
    stress units between n_r = 9 and n_r = 129, halving with each doubling --
    textbook O(h) -- so a 1e-3 tolerance needed n_r ~ 65 and a build cost to
    match.  The trapezoid correction restores O(h^2) and is one line.
    """
    if n_r <= 1:
        return np.zeros(1)
    w = np.ones(n_r)
    w[0] = w[-1] = 0.5
    return np.log(w)


def _extents(spec: RobitGridSpec, n_r: int):
    d_mu = (spec.mu_hi - spec.mu_lo) / (spec.n_mu - 1)
    d_eta = (spec.eta_hi - spec.eta_lo) / (spec.n_eta - 1)
    # A single r node carries no width, which is exactly what makes a degenerate
    # r prior collapse the smoothing kernel onto the two-parameter one.
    d_r = 0.0 if n_r <= 1 else spec.r_max / (n_r - 1)
    return d_mu, d_eta, d_r


# ---------------------------------------------------------------------------
# Localisation and refinement
# ---------------------------------------------------------------------------


def initial_spec(prior, n_mu=129, n_eta=129, n_r=9, width=6.0,
                 r_max=R_MAX) -> RobitGridSpec:
    """A (mu, eta) box wide enough to contain essentially all prior mass.

    Identical rule to `posterior_grid.initial_spec`, so the two references start
    from the same box and a difference between them is a difference in the
    model, not in where they were asked to look.
    """
    s = prior.spec
    mu_lo = max(1e-6, s.mu0 - width * s.sigma_mu)
    mu_hi = s.mu0 + width * s.sigma_mu
    if s.dependent:
        eta_mid = np.log(s.mu0) + s.log_alpha0
        eta_sd = np.sqrt(s.tau_scale**2 + (s.sigma_mu / s.mu0) ** 2)
    else:
        eta_mid = s.sigma_log_location
        eta_sd = s.tau_scale
    return RobitGridSpec(mu_lo, mu_hi, eta_mid - width * eta_sd,
                         eta_mid + width * eta_sd, n_mu, n_eta, n_r, r_max)


def _shrink_to_mass(post: RobitGridPosterior, tail: float) -> RobitGridSpec:
    """Smallest (mu, eta) box holding 1 - tail of each margin, padded by 25%."""
    W = post._cube().sum(axis=2)
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
    return RobitGridSpec(max(mu_lo, 1e-8), mu_hi, eta_lo, eta_hi,
                         post.spec.n_mu, post.spec.n_eta, post.spec.n_r,
                         post.spec.r_max)


def _box_is_stable(a: RobitGridSpec, b: RobitGridSpec, rtol: float = 0.02) -> bool:
    wa = max(a.mu_hi - a.mu_lo, 1e-12)
    wb = max(a.eta_hi - a.eta_lo, 1e-12)
    return (abs(b.mu_lo - a.mu_lo) < rtol * wa and abs(b.mu_hi - a.mu_hi) < rtol * wa
            and abs(b.eta_lo - a.eta_lo) < rtol * wb
            and abs(b.eta_hi - a.eta_hi) < rtol * wb)


def _threshold_set(post, tracked):
    out = {}
    for q in tracked:
        m, s = post.mean(q), max(post.sd(q), 1e-9)
        out[q] = (m - 1.5 * s, m, m + 1.5 * s)
    return out


def _summarise(post, tracked, thresholds):
    out = {"log_evidence": post.log_evidence}
    for q in tracked:
        out[f"{q}:mean"] = post.mean(q)
        out[f"{q}:sd"] = post.sd(q)
        for k, t in enumerate(thresholds[q]):
            out[f"{q}:cdf{k}"] = post.cdf_at(q, t)
    return out


def _deltas(a, b, tracked, thresholds):
    out = {"log_evidence": abs(b["log_evidence"] - a["log_evidence"])}
    for q in tracked:
        scale = max(b[f"{q}:sd"], 1e-12)
        out[f"{q}:mean"] = abs(b[f"{q}:mean"] - a[f"{q}:mean"]) / scale
        out[f"{q}:sd"] = abs(b[f"{q}:sd"] - a[f"{q}:sd"]) / scale
        for k in range(len(thresholds[q])):
            key = f"{q}:cdf{k}"
            out[key] = abs(b[key] - a[key])
    return out


def build_robit_reference_posterior(
    prior,
    r_prior: RPrior,
    x,
    y,
    *,
    tol: float = 1e-3,
    n_start: int = 129,
    n_max: int = 513,
    n_r_start: int = 9,
    n_r_max: int = 33,
    n_localise: int = 65,
    n_r_localise: int = 5,
    boundary_tol: float = 1e-10,
    tracked=DEFAULT_TRACKED,
    return_history: bool = False,
    fixed_n: int | None = None,
    fixed_n_r: int | None = None,
    spec: RobitGridSpec | None = None,
):
    """Adaptively refined three-parameter reference posterior.

    Same two-stage strategy as `posterior_grid.build_reference_posterior`:
    localise the (mu, eta) box on a coarse grid until it stops moving and its
    boundary mass is below tolerance, then refine until every tracked summary
    stops moving.  The refinement ladder doubles **all three** axes together, so
    a reported convergence is joint and never hides an unresolved r direction.

    `fixed_n` / `fixed_n_r` reproduce the DEC-7 production mode: localise, then
    build once at a resolution whose adequacy is established separately, and set
    `converged = None` so a caller can always tell a validated-resolution build
    from one that demonstrated its own convergence.

    `spec` bypasses localisation entirely and builds on exactly the box given.
    That is what the nesting invariant needs: the two- and three-parameter
    references must be compared on the *same* box, or the comparison measures
    two localisation passes rather than two models.
    """
    n_r_start = 1 if r_prior.degenerate else n_r_start

    if spec is not None:
        post = RobitGridPosterior.build(prior, r_prior, x, y, spec)
        post.converged = None
        post.convergence = {"tol": tol, "final_n": spec.n_mu, "final_n_r": post.spec.n_r,
                            "mode": "given_spec", "boundary_mass": post.boundary_mass(),
                            "max_relative_delta": float("nan"), "rounds": 1}
        return (post, []) if return_history else post

    cur = initial_spec(prior, n_mu=n_localise, n_eta=n_localise,
                       n_r=n_r_localise, r_max=r_prior.r_max)
    post = RobitGridPosterior.build(prior, r_prior, x, y, cur)
    for _ in range(4):
        new = _shrink_to_mass(post, boundary_tol)
        stable = _box_is_stable(cur, new)
        cur = new
        post = RobitGridPosterior.build(prior, r_prior, x, y, cur)
        if stable and post.boundary_mass() < boundary_tol:
            break

    thresholds = _threshold_set(post, tracked)

    if fixed_n is not None:
        n_r = 1 if r_prior.degenerate else (fixed_n_r or n_r_start)
        cur = RobitGridSpec(cur.mu_lo, cur.mu_hi, cur.eta_lo, cur.eta_hi,
                            fixed_n, fixed_n, n_r, r_prior.r_max)
        post = RobitGridPosterior.build(prior, r_prior, x, y, cur)
        post.converged = None
        post.convergence = {"tol": tol, "final_n": fixed_n, "final_n_r": post.spec.n_r,
                            "mode": "fixed", "boundary_mass": post.boundary_mass(),
                            "max_relative_delta": float("nan"), "rounds": 1}
        return (post, []) if return_history else post

    cur = RobitGridSpec(cur.mu_lo, cur.mu_hi, cur.eta_lo, cur.eta_hi,
                        n_start, n_start, n_r_start, r_prior.r_max)
    post = RobitGridPosterior.build(prior, r_prior, x, y, cur)
    history = [_summarise(post, tracked, thresholds)]
    converged = False
    n, n_r = n_start, post.spec.n_r
    while n < n_max or n_r < n_r_max:
        n_next = min(2 * (n - 1) + 1, n_max)
        n_r_next = n_r if r_prior.degenerate else min(2 * (n_r - 1) + 1, n_r_max)
        if n_next == n and n_r_next == n_r:
            break
        cur = RobitGridSpec(cur.mu_lo, cur.mu_hi, cur.eta_lo, cur.eta_hi,
                            n_next, n_next, n_r_next, r_prior.r_max)
        post_next = RobitGridPosterior.build(prior, r_prior, x, y, cur)
        summ_next = _summarise(post_next, tracked, thresholds)
        deltas = _deltas(history[-1], summ_next, tracked, thresholds)
        history.append(summ_next)
        post, n, n_r = post_next, n_next, post_next.spec.n_r
        if max(deltas.values()) < tol:
            converged = True
            break

    post.converged = converged
    post.convergence = {
        "tol": tol, "final_n": n, "final_n_r": n_r,
        "boundary_mass": post.boundary_mass(),
        "max_relative_delta": (
            max(_deltas(history[-2], history[-1], tracked, thresholds).values())
            if len(history) > 1 else float("nan")),
        "rounds": len(history),
    }
    return (post, history) if return_history else post
