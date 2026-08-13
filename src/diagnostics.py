"""Path-level diagnostics computable from the observable history alone.

Everything here is a function of H_n = (x_1, y_1, ..., x_n, y_n) plus the
numerical state of the inference, and never of the true parameter.  That
restriction is deliberate: these are the candidate inputs to the Phase 5
reliability map, and a diagnostic that peeks at the truth would be useless
online.  Phase 0-2 uses them for stratification only.
"""

from __future__ import annotations

import numpy as np

from .response_models import observed_information


def overlapping_pattern(x, y) -> bool:
    """True iff max{x_i : y_i = 0} > min{x_i : y_i = 1}.

    The classical condition for a finite probit MLE.  Without it the MLE for
    sigma is 0 and the likelihood surface is monotone in a direction: exactly
    the weak-identification regime the thesis wants to characterise.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    if not np.any(y == 0) or not np.any(y == 1):
        return False
    return float(x[y == 0].max()) > float(x[y == 1].min())


def stimulus_entropy(x, candidates, eps: float = 1e-12) -> float:
    """Shannon entropy (nats) of the empirical distribution of applied stresses.

    Computed over the candidate grid so that it is comparable across runs.
    Low values flag a design that has collapsed onto a single stimulus, the
    'design supplies growing information' assumption failure in the plan.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    idx = np.abs(x[:, None] - np.asarray(candidates)[None, :]).argmin(axis=1)
    counts = np.bincount(idx, minlength=len(candidates)).astype(float)
    p = counts / counts.sum()
    p = p[p > eps]
    return float(-np.sum(p * np.log(p)))


def path_summary(x, y, candidates=None) -> dict:
    """Observable summary of a completed (or partial) sequential path."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y).astype(int)
    n = len(x)
    n1 = int(y.sum())
    n0 = n - n1
    out = {
        "n": n,
        "n_positive": n1,
        "n_negative": n0,
        "all_zero": n > 0 and n1 == 0,
        "all_one": n > 0 and n0 == 0,
        "overlap": overlapping_pattern(x, y),
        "x_min": float(x.min()) if n else float("nan"),
        "x_max": float(x.max()) if n else float("nan"),
        "x_range": float(x.max() - x.min()) if n else float("nan"),
        "x_sd": float(x.std()) if n else float("nan"),
        "n_distinct_x": int(len(np.unique(np.round(x, 9)))),
    }
    out["response_balance"] = min(n1, n0) / n if n else float("nan")
    if candidates is not None and n:
        out["x_entropy"] = stimulus_entropy(x, candidates)
        out["x_at_grid_edge"] = bool(
            np.isclose(x, candidates[0]).any() or np.isclose(x, candidates[-1]).any()
        )
    return out


def information_summary(x, y, mu, sigma) -> dict:
    """Eigen-structure of the accumulated information at a plug-in parameter.

    `mu`, `sigma` should be a posterior summary (available online), not the
    truth.  A small minimum eigenvalue means some direction of the parameter
    space -- typically the scale, hence the tails -- has effectively not been
    measured, no matter how many trials were run.
    """
    I = observed_information(x, y, mu, sigma)
    evals = np.linalg.eigvalsh(I)
    evals = np.sort(evals)
    lam_min, lam_max = float(evals[0]), float(evals[-1])
    return {
        "info_min_eig": lam_min,
        "info_max_eig": lam_max,
        "info_logdet": float(np.log(max(np.linalg.det(I), 1e-300))),
        "info_condition": float(lam_max / lam_min) if lam_min > 0 else float("inf"),
    }


def target_vs_support(target_value: float, x, candidates) -> dict:
    """How far the estimated target sits from the region actually stimulated."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"target_outside_sampled": True, "target_support_gap": float("nan")}
    lo, hi = float(x.min()), float(x.max())
    if lo <= target_value <= hi:
        gap = 0.0
    else:
        gap = float(min(abs(target_value - lo), abs(target_value - hi)))
    return {
        "target_outside_sampled": not (lo <= target_value <= hi),
        "target_support_gap": gap,
        "target_outside_grid": bool(
            target_value < candidates[0] or target_value > candidates[-1]
        ),
    }


def degeneracy_flags(path: dict, particle_snapshot: dict, ess_floor: float = 100.0) -> dict:
    """Named failure conditions.  These are outcomes, never reasons to drop a run."""
    return {
        "degenerate_all_equal": bool(path["all_zero"] or path["all_one"]),
        "degenerate_no_overlap": not bool(path["overlap"]),
        "degenerate_low_ess": bool(particle_snapshot.get("ess", np.inf) < ess_floor),
        "degenerate_particle_invalid": bool(
            particle_snapshot.get("n_invalid_particles", 0) > 0
        ),
    }
