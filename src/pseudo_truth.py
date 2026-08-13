"""Policy-dependent pseudo-true parameters under misspecification.

For a fixed design distribution nu, the KL projection of the true response
curve p* onto the fitted probit family is

    theta_nu = argmax_theta  E_nu[ p*(X) log p_theta(X)
                                 + (1 - p*(X)) log(1 - p_theta(X)) ].

Under an *adaptive* design the empirical design distribution nu_n is itself
produced by the fitted model and the observed responses, so its limit -- and
hence the projection -- can depend on the acquisition policy.  That is the
central mechanism the plan asks to test: the same physical response curve can
imply different pseudo-true probit curves, and therefore different fitted
quantiles, under different policies.

Nothing here is an estimand.  The pseudo-true quantile is a property of the
(curve, policy) pair used to explain *where the posterior is heading*; the
estimand is always the physical quantile.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

from .response_models import derived_values


def empirical_design_distribution(x_all, grid=None, bins: int = 200):
    """Weights of the pooled empirical stimulus distribution.

    `x_all` is every stimulus applied across replicates of one (curve, policy)
    cell.  Returns (support, weights) with weights summing to one.
    """
    x = np.asarray(x_all, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if grid is not None:
        grid = np.asarray(grid, dtype=float)
        idx = np.abs(x[:, None] - grid[None, :]).argmin(axis=1)
        counts = np.bincount(idx, minlength=len(grid)).astype(float)
        keep = counts > 0
        return grid[keep], counts[keep] / counts.sum()
    counts, edges = np.histogram(x, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    keep = counts > 0
    return centres[keep], counts[keep] / counts.sum()


def kl_projection(curve, support, weights, start=(30.0, np.log(3.0)),
                  bounds=((1e-3, 200.0), (np.log(1e-3), np.log(1e3)))):
    """Probit KL projection of `curve` under the design distribution.

    Maximising the expected log-likelihood is exactly minimising the
    nu-weighted KL divergence from the true Bernoulli law to the fitted one,
    since the entropy term does not involve theta.
    """
    support = np.asarray(support, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    p_star = np.clip(np.atleast_1d(curve.prob(support)), 0.0, 1.0)

    def neg_obj(theta):
        mu, eta = theta
        z = (support - mu) / np.exp(eta)
        return -float(np.sum(weights * (p_star * log_ndtr(z)
                                        + (1 - p_star) * log_ndtr(-z))))

    best = None
    for s in (start, (float(np.average(support, weights=weights)), np.log(3.0)),
              (30.0, np.log(1.0)), (30.0, np.log(8.0))):
        res = minimize(neg_obj, np.asarray(s, dtype=float), method="Nelder-Mead",
                       options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res
    mu = float(best.x[0])
    sigma = float(np.exp(np.clip(best.x[1], *bounds[1])))
    return {
        "pseudo_mu": mu,
        "pseudo_sigma": sigma,
        "expected_loglik": float(-best.fun),
        "converged": bool(best.success),
        "n_support": int(len(support)),
    }


def pseudo_true_quantiles(projection: dict, targets=("q0.5", "q0.95", "q0.99")):
    return {t: float(derived_values(projection["pseudo_mu"],
                                    projection["pseudo_sigma"], t))
            for t in targets}


def projection_with_bootstrap(curve, x_by_replicate, grid=None, n_boot: int = 200,
                              seed: int = 0, targets=("q0.5", "q0.95", "q0.99")):
    """KL projection plus a replicate-level bootstrap for its Monte Carlo error.

    Resampling *replicates* (not individual stimuli) is the right unit: the
    design distribution is generated one whole experiment at a time, and
    stimuli within an experiment are strongly dependent.
    """
    rng = np.random.default_rng(seed)
    reps = [np.asarray(r, dtype=float) for r in x_by_replicate if len(r)]
    support, weights = empirical_design_distribution(np.concatenate(reps), grid)
    point = kl_projection(curve, support, weights)
    point_q = pseudo_true_quantiles(point, targets)

    boot = {t: [] for t in targets}
    boot_mu, boot_sigma = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(reps), len(reps))
        s, w = empirical_design_distribution(np.concatenate([reps[i] for i in idx]), grid)
        pr = kl_projection(curve, s, w, start=(point["pseudo_mu"],
                                               np.log(point["pseudo_sigma"])))
        q = pseudo_true_quantiles(pr, targets)
        for t in targets:
            boot[t].append(q[t])
        boot_mu.append(pr["pseudo_mu"])
        boot_sigma.append(pr["pseudo_sigma"])

    out = dict(point)
    out["pseudo_mu_se"] = float(np.std(boot_mu, ddof=1)) if n_boot > 1 else np.nan
    out["pseudo_sigma_se"] = float(np.std(boot_sigma, ddof=1)) if n_boot > 1 else np.nan
    for t in targets:
        out[f"pseudo_{t}"] = point_q[t]
        out[f"pseudo_{t}_se"] = float(np.std(boot[t], ddof=1)) if n_boot > 1 else np.nan
        out[f"physical_{t}"] = float(np.atleast_1d(curve.quantile(float(t[1:])))[0])
        out[f"pseudo_minus_physical_{t}"] = out[f"pseudo_{t}"] - out[f"physical_{t}"]
    out["n_replicates"] = len(reps)
    out["n_boot"] = n_boot
    return out


def design_distribution_summary(x_all) -> dict:
    x = np.asarray(x_all, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    return {
        "design_mean": float(x.mean()),
        "design_sd": float(x.std()),
        "design_q05": float(np.quantile(x, 0.05)),
        "design_q50": float(np.quantile(x, 0.50)),
        "design_q95": float(np.quantile(x, 0.95)),
        "design_min": float(x.min()),
        "design_max": float(x.max()),
        "design_n": int(x.size),
    }
