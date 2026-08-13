"""Mutual-information design utilities.

Two estimators, matching Rotem sections 5.2 and 5.3:

* `mutual_information_vector` -- I((mu, sigma); Y_{n+1} | x).  Because the
  particle representation already carries the exact response probability for
  each draw, this is a pair of weighted averages and is exact up to the Monte
  Carlo error of the particle set.

* `mutual_information_scalar` -- I(theta; Y_{n+1} | x) for a scalar functional
  theta = g(mu, sigma) (mu, sigma, or a quantile).  Rotem's quadrature: R grid
  points equally spaced in posterior probability, rectangular kernel density
  estimates of pi(theta | y_{n+1}=1, x) and pi(theta | y_{n+1}=0, x), and a
  Bayes-rule inversion for P(y_{n+1}=1 | theta, x).

  The implementation exploits an algebraic simplification of her formulae.
  With A_r(x) = sum_{j in S_r} w_j p_j(x) and B_r(x) = sum_{j in S_r} w_j
  (1 - p_j(x)),

      P_{r,x} = pi^(1)_r Phibar / (pi^(1)_r Phibar + pi^(0)_r (1 - Phibar))
              = A_r(x) / (A_r(x) + B_r(x)) = A_r(x) / sum_{j in S_r} w_j,

  because the normalising constants of pi^(1) and pi^(0) cancel exactly
  against Phibar and 1 - Phibar.  Since every S_r is a contiguous block of the
  theta-sorted particles, A_r is a difference of prefix sums.  The cost drops
  from O(R N L) to O(N L), which is what makes the scalar-target policies
  affordable at Rotem's N = 10,000 and L = 200.  The identity is verified
  against a literal transcription of her formulae in
  `tests/test_mutual_information.py::test_scalar_mi_matches_literal_formulae`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .response_models import binary_entropy, derived_values, probit_prob_matrix

DEFAULT_R = 100
DEFAULT_C_FRACTION = 0.1


def response_probability_matrix(x_grid, mu, sigma, dtype=np.float64):
    """(N, L) response probabilities: rows are parameter draws."""
    return probit_prob_matrix(x_grid, mu, sigma, dtype=dtype)


def response_probability_matrix_T(x_grid, mu, sigma, dtype=np.float64):
    """(L, N) response probabilities: rows are *candidate stimuli*.

    This is the layout every hot loop uses.  Both reductions the designs need
    -- a weighted average over draws, and a segmented sum over draws -- then
    run along the contiguous axis.  Measured on the working configuration
    (N = 10,000, L = 200) the segmented sum is ~11x faster in this layout than
    the equivalent prefix scan in (N, L), which is what makes the scalar-target
    quadrature cheap enough to run at Rotem's settings.
    """
    z = (np.asarray(x_grid, dtype=dtype)[:, None] - np.asarray(mu, dtype=dtype)[None, :]) / \
        np.asarray(sigma, dtype=dtype)[None, :]
    from scipy.special import ndtr

    return np.ascontiguousarray(ndtr(z), dtype=dtype)


def mutual_information_vector_T(w, P_T, H_T=None):
    """I(theta_vector; Y | x) with P in (L, N) layout."""
    w64 = np.asarray(w, dtype=np.float64)
    P_T = np.asarray(P_T, dtype=np.float64)
    phibar = P_T @ w64
    cond = (binary_entropy(P_T) if H_T is None else np.asarray(H_T, dtype=np.float64)) @ w64
    return binary_entropy(phibar) - cond


def mutual_information_vector(w, P, H=None):
    """I(theta_vector; Y | x) for every column of P.

    Parameters
    ----------
    w : (N,) normalised posterior weights
    P : (N, L) response probabilities
    H : (N, L) optional precomputed entrywise binary entropies of P.
    """
    return mutual_information_vector_T(
        w, np.asarray(P).T, None if H is None else np.asarray(H).T
    )


@dataclass
class ScalarMIResult:
    mi: np.ndarray
    density_integral: float
    n_empty_windows: int
    bandwidth: float
    n_distinct_nodes: int
    min_unclipped: float = float("nan")
    n_clipped: int = 0
    clipped: bool = False

    def diagnostics(self) -> dict:
        return {
            "quadrature_density_integral": self.density_integral,
            "quadrature_empty_windows": self.n_empty_windows,
            "quadrature_bandwidth": self.bandwidth,
            "quadrature_distinct_nodes": self.n_distinct_nodes,
            "quadrature_min_unclipped": self.min_unclipped,
            "quadrature_n_clipped": self.n_clipped,
            "quadrature_clipped": self.clipped,
        }


def mutual_information_scalar(
    theta, w, P, R: int = DEFAULT_R, c_fraction: float = DEFAULT_C_FRACTION,
    clip_negative: bool = False
) -> ScalarMIResult:
    """I(theta; Y | x) for every column of P, by Rotem's quadrature.

    `density_integral` is sum_r pi_r (theta_{r+1} - theta_r), a Riemann sum for
    the posterior density of theta.  It should be close to 1; departures are a
    direct measure of quadrature failure (typically caused by posterior
    concentration collapsing the quantile nodes onto duplicate particle
    values), and are propagated into the run manifest rather than ignored.
    """
    theta = np.asarray(theta, dtype=np.float64)
    order = np.argsort(theta, kind="stable")
    return mutual_information_scalar_presorted(
        theta[order], np.asarray(w, dtype=np.float64)[order],
        np.ascontiguousarray(np.asarray(P)[order].T), R=R, c_fraction=c_fraction,
        clip_negative=clip_negative,
    )


def mutual_information_scalar_presorted(
    th, ws, Ps_T, R: int = DEFAULT_R, c_fraction: float = DEFAULT_C_FRACTION,
    clip_negative: bool = False
) -> ScalarMIResult:
    """`mutual_information_scalar` with the theta-sort already applied.

    `Ps_T` is (L, N) with columns in ascending theta order.  The sort order
    depends only on the particle values, so a caller holding a
    generation-keyed cache can hoist it out of the sequential loop.

    Every window S_r is a contiguous run of the theta-sorted draws, so the
    windowed sums are differences of prefix sums taken at only 2R breakpoints.
    `np.add.reduceat` computes the segment sums between consecutive
    breakpoints in one pass; the prefix scan is then over ~200 segments rather
    than N draws.
    """
    th = np.asarray(th, dtype=np.float64)
    ws = np.asarray(ws, dtype=np.float64)
    Ps_T = np.asarray(Ps_T, dtype=np.float64)
    L, N = Ps_T.shape

    mean = float(ws @ th)
    sd = float(np.sqrt(max(ws @ (th - mean) ** 2, 0.0)))
    c = c_fraction * sd
    if not np.isfinite(c) or c <= 0:
        # degenerate posterior on theta: no scalar information to resolve
        return ScalarMIResult(np.zeros(L), 0.0, R, 0.0, 0, clipped=clip_negative)

    # quadrature nodes: equally spaced in posterior probability
    cw_full = np.cumsum(ws)
    cw = cw_full / cw_full[-1]
    probs = (np.arange(R) + 0.5) / R
    node_idx = np.clip(np.searchsorted(cw, probs, side="right"), 0, N - 1)
    nodes = th[node_idx]

    lo = np.searchsorted(th, nodes - c, side="right")
    hi = np.searchsorted(th, nodes + c, side="left")

    cw0 = np.concatenate([[0.0], cw_full])
    denom = cw0[hi] - cw0[lo]                      # sum of w_j over S_r
    pi_r = denom / (2.0 * c)

    # segmented sums of w_j p_jl over the breakpoints
    breaks = np.unique(np.concatenate([[0], lo, hi]))
    breaks = breaks[breaks < N]
    wP = Ps_T * ws[None, :]
    seg = np.add.reduceat(wP, breaks, axis=1)      # (L, n_seg)
    prefix = np.empty((L, len(breaks) + 1))
    prefix[:, 0] = 0.0
    np.cumsum(seg, axis=1, out=prefix[:, 1:])
    total = prefix[:, -1]

    def at(idx):
        # every lo/hi is itself a breakpoint, except the value N, which
        # searchsorted maps to the final column holding the full sum
        return prefix[:, np.searchsorted(breaks, idx, side="left")]

    A = at(hi) - at(lo)                            # (L, R)

    empty = denom <= 0
    safe_denom = np.where(empty, 1.0, denom)
    P_rx = np.clip(A / safe_denom[None, :], 0.0, 1.0)
    H_r = binary_entropy(P_rx)
    H_r[:, empty] = 0.0

    dtheta = np.diff(nodes)
    weight_r = pi_r[:-1] * dtheta                  # (R-1,)
    cond = H_r[:, :-1] @ weight_r

    mi = binary_entropy(total) - cond

    # Rotem's estimator is biased low and can return negative values, which no
    # mutual information can be (Phase 1, D14).  Clipping at zero is the Phase 3
    # default; the unclipped minimum and the number of clipped entries are
    # always recorded so the numerical error stays visible, and
    # `clip_negative=False` reproduces the original convention exactly.
    min_unclipped = float(mi.min())
    n_clipped = int(np.sum(mi < 0.0))
    if clip_negative:
        mi = np.maximum(mi, 0.0)

    return ScalarMIResult(
        mi=mi,
        density_integral=float(np.sum(weight_r)),
        n_empty_windows=int(np.sum(empty)),
        bandwidth=float(c),
        n_distinct_nodes=int(len(np.unique(nodes))),
        min_unclipped=min_unclipped,
        n_clipped=n_clipped,
        clipped=bool(clip_negative),
    )


def mutual_information_scalar_literal(
    theta, w, P, R: int = DEFAULT_R, c_fraction: float = DEFAULT_C_FRACTION
):
    """Direct transcription of Rotem's section 5.3 formulae.

    Deliberately slow and unoptimised.  Exists only so the fast path can be
    checked against the equations as written; never used in experiments.
    """
    theta = np.asarray(theta, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    L = P.shape[1]

    mean = float(w @ theta)
    sd = float(np.sqrt(max(w @ (theta - mean) ** 2, 0.0)))
    c = c_fraction * sd

    order = np.argsort(theta, kind="stable")
    cw = np.cumsum(w[order])
    cw /= cw[-1]
    probs = (np.arange(R) + 0.5) / R
    nodes = theta[order][np.clip(np.searchsorted(cw, probs, side="right"), 0, len(theta) - 1)]

    S = [np.abs(theta - node) < c for node in nodes]
    out = np.zeros(L)
    for l in range(L):
        p = P[:, l]
        L1 = w * p
        L0 = w * (1.0 - p)
        w1 = L1 / L1.sum()
        w0 = L0 / L0.sum()
        phibar = float(w @ p)

        pi1 = np.array([w1[s].sum() / (2 * c) for s in S])
        pi0 = np.array([w0[s].sum() / (2 * c) for s in S])
        pir = np.array([w[s].sum() / (2 * c) for s in S])

        num = pi1 * phibar
        den = pi1 * phibar + pi0 * (1 - phibar)
        Prx = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
        Hr = binary_entropy(Prx)
        Hr[den <= 0] = 0.0
        cond = float(np.sum(Hr[:-1] * pir[:-1] * np.diff(nodes)))
        out[l] = binary_entropy(np.array([phibar]))[0] - cond
    return out


# --------------------------------------------------------------------------
# Quantile-indexed utility surface  U_p(r; H_n)
# --------------------------------------------------------------------------


def stimulus_at_response_level(posterior, r_levels, summary: str = "median"):
    """tilde q_r(H_n): posterior summary of the stimulus where P(Y=1) = r.

    Under the fitted probit family the stimulus achieving response probability
    r is itself the r-quantile of the sensitivity curve, q_r = mu + sigma z_r,
    which is a random variable under the posterior.  `summary` selects the
    posterior functional used to turn it into a single candidate stimulus.
    """
    out = []
    for r in np.atleast_1d(r_levels):
        vals = derived_values(posterior.mu, posterior.sigma, f"q{r}")
        if summary == "median":
            out.append(float(np.interp(0.5, _cumprob(posterior.w, vals), np.sort(vals))))
        elif summary == "mean":
            out.append(float(posterior.w @ vals))
        else:
            raise ValueError(f"unknown summary {summary!r}")
    return np.asarray(out)


def _cumprob(w, values):
    order = np.argsort(values, kind="stable")
    ws = np.asarray(w)[order]
    cw = np.cumsum(ws)
    return (cw - 0.5 * ws) / cw[-1]


def utility_surface(posterior, target_ps, stimulus_rs, R: int = DEFAULT_R,
                    c_fraction: float = DEFAULT_C_FRACTION):
    """U_p(r; H_n) over a grid of target quantiles p and stimulus levels r.

    Returns
    -------
    dict with 'x_levels' (the stimuli tilde q_r), 'surface' of shape
    (len(target_ps), len(stimulus_rs)), 'vector_mi' -- the full-parameter MI at
    the same stimuli, for reference -- and per-row quadrature diagnostics.
    """
    x_levels = stimulus_at_response_level(posterior, stimulus_rs)
    P = response_probability_matrix(x_levels, posterior.mu, posterior.sigma)

    surface = np.zeros((len(target_ps), len(x_levels)))
    diagnostics = []
    for i, p in enumerate(target_ps):
        theta = derived_values(posterior.mu, posterior.sigma, f"q{p}")
        res = mutual_information_scalar(theta, posterior.w, P, R=R, c_fraction=c_fraction)
        surface[i] = res.mi
        diagnostics.append(res.diagnostics())

    return {
        "target_ps": np.asarray(target_ps),
        "stimulus_rs": np.asarray(stimulus_rs),
        "x_levels": x_levels,
        "surface": surface,
        "vector_mi": mutual_information_vector(posterior.w, P),
        "quadrature_diagnostics": diagnostics,
    }
