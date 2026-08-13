"""Response curves.

Two distinct roles are kept strictly separate throughout the codebase:

* the **fitted family** -- always the two-parameter probit location-scale model
  ``P(Y=1|x) = Phi((x - mu)/sigma)``.  This is what every inference routine
  assumes.
* the **data-generating curve** (`ResponseCurve`) -- the physical truth used by
  the simulator.  It may or may not lie in the fitted family.

Keeping these apart is what makes "misspecification" a well-defined object
later (Phase 3) and prevents the confusion the plan warns about between
computational posterior error and response-model misspecification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.special import log_ndtr, ndtr
from scipy.stats import beta as beta_dist
from scipy.stats import norm

# Probabilities are clamped away from {0, 1} only where an entropy or a plain
# (non-log) probability is required.  Log-likelihoods use `log_ndtr` and never
# need a floor, so the floor never silently changes an inference result.
PROB_FLOOR = 1e-12


class ResponseCurve(Protocol):
    """A true response curve p*(x) = P(Y = 1 | x)."""

    name: str

    def prob(self, x: np.ndarray) -> np.ndarray: ...

    def quantile(self, p: float | np.ndarray) -> float | np.ndarray: ...

    def describe(self) -> dict: ...


@dataclass(frozen=True)
class ProbitCurve:
    """P(Y=1|x) = Phi((x - mu) / sigma).  Lies inside the fitted family."""

    mu: float
    sigma: float
    name: str = "probit"

    def prob(self, x):
        return ndtr((np.asarray(x, dtype=float) - self.mu) / self.sigma)

    def quantile(self, p):
        return self.mu + self.sigma * norm.ppf(p)

    def describe(self):
        return {"family": "probit", "mu": self.mu, "sigma": self.sigma}


@dataclass(frozen=True)
class BetaCDFCurve:
    """Rotem's asymmetric misspecification case.

    The stress is mapped to [0, 1] by ``u = (x - lo) / (hi - lo)`` and the
    response probability is the Beta(a, b) CDF of u.  The curve is *defined* by
    ``(lo, hi)`` -- these are physical constants of the true system and are
    deliberately NOT tied to the experimenter's design grid, which depends on
    the prior calibration.  See `docs/discrepancies.md`, item D3.
    """

    a: float = 5.0
    b: float = 8.0
    lo: float = 18.0
    hi: float = 48.0
    name: str = "beta_cdf"

    def prob(self, x):
        u = (np.asarray(x, dtype=float) - self.lo) / (self.hi - self.lo)
        return beta_dist.cdf(np.clip(u, 0.0, 1.0), self.a, self.b)

    def quantile(self, p):
        return self.lo + (self.hi - self.lo) * beta_dist.ppf(p, self.a, self.b)

    @property
    def implied_median(self) -> float:
        return float(self.quantile(0.5))

    @property
    def implied_sd(self) -> float:
        a, b = self.a, self.b
        var_u = a * b / ((a + b) ** 2 * (a + b + 1))
        return float((self.hi - self.lo) * np.sqrt(var_u))

    def describe(self):
        return {
            "family": "beta_cdf",
            "a": self.a,
            "b": self.b,
            "lo": self.lo,
            "hi": self.hi,
            "median": self.implied_median,
            "sd": self.implied_sd,
        }


# --------------------------------------------------------------------------
# Fitted family: probit helpers, vectorised over parameter draws x stimuli
# --------------------------------------------------------------------------


def probit_prob(x, mu, sigma, dtype=np.float64):
    """Phi((x - mu)/sigma), broadcasting over any compatible shapes."""
    z = (np.asarray(x, dtype=dtype) - np.asarray(mu, dtype=dtype)) / np.asarray(
        sigma, dtype=dtype
    )
    return ndtr(z)


def probit_prob_matrix(x_grid, mu, sigma, dtype=np.float64):
    """(N, L) matrix of response probabilities.

    Rows index parameter draws, columns index candidate stimuli.  This matrix
    dominates the cost of every mutual-information evaluation, but it depends
    only on the particle values and the candidate grid -- not on the weights.
    It is therefore cached by `PolicyState` and rebuilt only when the particle
    set is rejuvenated, which is what makes float64 affordable throughout.
    `tests/test_numerics.py::test_float32_matrix_matches_float64` records the
    error that a float32 variant would have introduced, for the record.
    """
    z = (np.asarray(x_grid, dtype=dtype)[None, :] - np.asarray(mu, dtype=dtype)[:, None]) / np.asarray(
        sigma, dtype=dtype
    )[:, None]
    return ndtr(z).astype(dtype, copy=False)


def probit_loglik(x, y, mu, sigma):
    """Stable per-observation log-likelihood of the fitted probit family.

    Uses `log_ndtr`, so no probability floor is applied and no underflow
    occurs even for |z| ~ 40.  Broadcasts over parameter arrays.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = (x - np.asarray(mu)[..., None]) / np.asarray(sigma)[..., None]
    return np.where(y > 0.5, log_ndtr(z), log_ndtr(-z))


def binary_entropy(p):
    """h(p) = -p log p - (1-p) log(1-p), in nats, with h(0) = h(1) = 0.

    The input is promoted to float64 *before* clipping.  This is not cosmetic:
    in float32, ``1.0 - 1e-12`` rounds to exactly 1.0, so a float32 clip leaves
    saturated probabilities at 1, and ``log1p(-1) = -inf`` then poisons the
    entropy matrix with NaN.  Promoting first makes the floor effective for any
    input dtype.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), PROB_FLOOR, 1.0 - PROB_FLOOR)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def quantile_from_params(mu, sigma, p):
    """q_p = mu + sigma * z_p under the fitted probit family."""
    return np.asarray(mu) + np.asarray(sigma) * norm.ppf(p)


def derived_values(mu, sigma, quantity: str):
    """Values of a named derived quantity at each (mu, sigma) pair.

    Recognised names: ``"mu"``, ``"sigma"``, and ``"q<p>"`` for the p-quantile
    of the fitted sensitivity curve (e.g. ``"q0.95"``).
    """
    if quantity == "mu":
        return np.asarray(mu)
    if quantity == "sigma":
        return np.asarray(sigma)
    if quantity.startswith("q"):
        return np.asarray(mu) + np.asarray(sigma) * norm.ppf(float(quantity[1:]))
    raise ValueError(f"unknown quantity {quantity!r}")


def params_from_two_quantiles(q1, q2, p1, p2):
    """Invert (q_p1, q_p2) -> (mu, sigma).

    This is the constructive half of the two-quantile reparameterisation
    proposition in the Phase 0 note: for p1 != p2 the map is a bijection, so
    any two distinct quantiles carry exactly the information in (mu, sigma).
    """
    if np.isclose(p1, p2):
        raise ValueError("p1 and p2 must be distinct for the map to be invertible")
    z1, z2 = norm.ppf(p1), norm.ppf(p2)
    sigma = (np.asarray(q2) - np.asarray(q1)) / (z2 - z1)
    mu = np.asarray(q1) - sigma * z1
    return mu, sigma


def observed_information(x, y, mu, sigma):
    """Observed (= expected, for canonical-free probit here) information matrix.

    Returns the 2x2 Fisher information for (mu, sigma) accumulated over the
    design points ``x``.  The responses ``y`` are unused because the *expected*
    information for a probit GLM does not depend on the realised outcomes; the
    argument is kept so callers can pass a full history uniformly.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.zeros((2, 2))
    z = (x - mu) / sigma
    p = np.clip(ndtr(z), PROB_FLOOR, 1 - PROB_FLOOR)
    w = norm.pdf(z) ** 2 / (p * (1.0 - p)) / sigma**2
    i11 = np.sum(w)
    i12 = np.sum(w * z)
    i22 = np.sum(w * z * z)
    return np.array([[i11, i12], [i12, i22]])
