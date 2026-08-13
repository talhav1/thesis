"""Asymmetric threshold decision loss.

    L(q_hat, q_star) = c_under * (q_star - q_hat)_+  +  c_over * (q_hat - q_star)_+

Used when the estimated quantile is adopted as an operational threshold.  In a
sensitivity setting the two errors are not symmetric: declaring a stimulus safe
when it is not is usually far worse than being conservative.

Following the plan, the cost ratio is never fixed to one unverifiable number.
`loss_curve` sweeps it, so the reader sees the whole trade-off rather than a
single point chosen by the analyst.

Phases 0-2 only report these as descriptive summaries alongside coverage; the
decision-calibration layer proper belongs to Phase 5.
"""

from __future__ import annotations

import numpy as np


def threshold_loss(q_hat, q_star, c_under: float = 10.0, c_over: float = 1.0):
    """Elementwise asymmetric loss.

    `c_under` prices *underestimating* the true quantile -- calling a stimulus
    safe that is not -- and is the larger cost in the intended application.
    """
    q_hat = np.asarray(q_hat, dtype=float)
    q_star = np.asarray(q_star, dtype=float)
    under = np.clip(q_star - q_hat, 0, None)
    over = np.clip(q_hat - q_star, 0, None)
    return c_under * under + c_over * over


def expected_loss(q_hat, q_star, c_under=10.0, c_over=1.0):
    """Mean loss with its Monte Carlo standard error."""
    L = threshold_loss(q_hat, q_star, c_under, c_over)
    L = L[np.isfinite(L)]
    if len(L) < 2:
        return float("nan"), float("nan")
    return float(L.mean()), float(L.std(ddof=1) / np.sqrt(len(L)))


def loss_curve(q_hat, q_star, ratios=(1, 2, 5, 10, 20, 50, 100)):
    """Expected loss across a range of cost ratios c_under / c_over."""
    out = []
    for r in ratios:
        m, se = expected_loss(q_hat, q_star, c_under=float(r), c_over=1.0)
        out.append({"cost_ratio": r, "expected_loss": m, "se": se})
    return out


def false_certainty(covered, width, narrowness_threshold: float):
    """Rate of intervals that miss the truth *while being narrow*.

    The plan's headline reliability metric: an interval that is wrong is
    tolerable if it is visibly wide, and dangerous if it is confidently tight.
    Averaged MSE cannot see this distinction, which is why it is reported
    separately.
    """
    covered = np.asarray(covered, dtype=bool)
    width = np.asarray(width, dtype=float)
    narrow = width <= narrowness_threshold
    k = int(np.sum(narrow & ~covered))
    n = len(covered)
    p = k / n if n else float("nan")
    se = float(np.sqrt(max(p * (1 - p), 0.0) / n)) if n else float("nan")
    return {"false_certainty_rate": p, "se": se, "n": n, "n_events": k,
            "narrowness_threshold": float(narrowness_threshold),
            "frac_narrow": float(narrow.mean()) if n else float("nan")}
