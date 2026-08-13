"""Maximum-likelihood estimation for the fitted probit family.

Needed because several of Rotem's tables report MLEs rather than Bayes
estimates.  The MLE is also the cleanest place where weak identification
becomes visible: without an overlapping pattern the likelihood has no interior
maximum and sigma_hat drifts to 0.  Such runs are flagged (`finite=False`) and
kept, never dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import norm

SIGMA_LOG_BOUNDS = (np.log(1e-3), np.log(1e3))


@dataclass
class MLEResult:
    mu: float
    sigma: float
    converged: bool
    finite: bool          # False when the design has no overlapping pattern
    n_iter: int
    message: str

    def quantile(self, p: float) -> float:
        return self.mu + self.sigma * norm.ppf(p)

    def as_dict(self, prefix="mle_", quantiles=(0.5, 0.95, 0.99)):
        out = {
            f"{prefix}mu": self.mu,
            f"{prefix}sigma": self.sigma,
            f"{prefix}converged": self.converged,
            f"{prefix}finite": self.finite,
        }
        for p in quantiles:
            out[f"{prefix}q{p}"] = self.quantile(p)
        return out


def _neg_loglik(params, x, y):
    mu, eta = params
    sigma = np.exp(eta)
    z = (x - mu) / sigma
    return -np.sum(np.where(y > 0.5, log_ndtr(z), log_ndtr(-z)))


def fit_probit_mle(x, y, start=None) -> MLEResult:
    """Probit MLE in (mu, log sigma), with explicit separation handling."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y).astype(int)

    from .diagnostics import overlapping_pattern

    has_overlap = overlapping_pattern(x, y)

    if start is None:
        x1 = x[y == 1]
        x0 = x[y == 0]
        mu0 = float(np.mean(x))
        if len(x1) and len(x0):
            mu0 = 0.5 * (float(x1.min()) + float(x0.max()))
        sd0 = max(float(np.std(x)), 1e-2)
        start = np.array([mu0, np.log(sd0)])

    res = minimize(
        _neg_loglik, start, args=(x, y), method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-10},
    )
    mu = float(res.x[0])
    eta = float(np.clip(res.x[1], *SIGMA_LOG_BOUNDS))
    return MLEResult(
        mu=mu,
        sigma=float(np.exp(eta)),
        converged=bool(res.success),
        finite=bool(has_overlap),
        n_iter=int(res.nit),
        message=str(res.message),
    )


def dixon_mood_estimates(x, y, step: float) -> dict:
    """Dixon & Mood's (1948) closed-form Bruceton estimators.

    Kept for fidelity to the historical method that Rotem's Bruceton column
    describes, alongside the probit MLE she actually reports.  The validity
    condition (N*B - A^2)/N^2 > 0.3 is returned rather than enforced.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y).astype(int)
    n0 = int(np.sum(y == 0))
    n1 = int(np.sum(y == 1))
    N = min(n0, n1)
    if N == 0:
        return {"dm_mu": float("nan"), "dm_sigma": float("nan"),
                "dm_condition": float("nan"), "dm_valid": False}

    minority = 0 if n0 < n1 else 1
    sel = x[y == minority]
    x0 = float(np.min(x))
    levels = np.round((sel - x0) / step).astype(int)
    A = float(np.sum(levels))
    B = float(np.sum(levels.astype(float) ** 2))
    cond = (N * B - A**2) / N**2
    mu = x0 + step * (A / N + (0.5 if minority == 0 else -0.5))
    sigma = 1.62 * step * (cond + 0.029)
    return {"dm_mu": mu, "dm_sigma": sigma, "dm_condition": cond,
            "dm_valid": bool(cond > 0.3)}
