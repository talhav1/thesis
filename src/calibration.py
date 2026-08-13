"""Calibration evaluation: simulation-based calibration and coverage maps.

The plan insists these four layers stay separate and are never collapsed into
one score:

1. **Prior-predictive / SBC calibration.**  Draw theta ~ pi_0, simulate the
   *entire* adaptive experiment, and check that the posterior rank of the true
   value is uniform.  Exact inference must pass.  This is the check that
   actually settles the "does adaptivity break the posterior?" question
   empirically, because the policy runs inside the loop.
2. **Fixed-parameter operating characteristics.**  Fix theta_0 and repeat.
   Coverage need not equal the nominal level here, especially in small samples
   under an informative prior, so a deviation is not by itself evidence of a
   bug -- a point the plan is explicit about.
3. **Local coverage across observable path strata.**
4. **Decision calibration** (Phase 5+; not exercised in Phases 0-2).

The rank statistic used throughout is the continuous one,
r = P(Q < Q_true | data), rather than a discrete rank among draws.  It is
uniform under exact inference for any continuous posterior and avoids the
tie-handling and binning artefacts of the discrete version.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .diagnostics import degeneracy_flags, information_summary, path_summary
from .policies import BrucetonPolicy, PolicyState, build_policy
from .posterior_grid import build_reference_posterior
from .priors import Prior, PriorSpec, rotem_stimulus_grid
from .response_models import ProbitCurve
from .rotem_particles import ParticlePosterior
from .simulator import ExperimentConfig, build_prior, latent_uniforms


def sbc_replicate(config: ExperimentConfig, replicate: int, seed_base: int,
                  use_reference: bool = True, compute_widths: bool = False):
    """One SBC draw: theta ~ prior, full adaptive experiment, rank statistics.

    The response curve is *always* the fitted family here -- SBC is only
    meaningful when the model is correct.  Any deviation from uniformity is
    therefore attributable to the inference, not to misspecification.
    """
    rng = np.random.default_rng([seed_base, replicate, 0x5BC])
    prior = build_prior(config.prior)
    mu_true, sigma_true = prior.sample(1, rng)
    mu_true, sigma_true = float(mu_true[0]), float(sigma_true[0])
    curve = ProbitCurve(mu_true, sigma_true)

    candidates = rotem_stimulus_grid(config.calibration, config.grid_points,
                                     config.grid_half_width)
    if config.policy == "bruceton":
        policy = BrucetonPolicy(step=config.bruceton_step, x0=float(rng.normal(prior.spec.mu0, 3.0)))
    else:
        policy = build_policy(config.policy, **config.policy_kwargs)

    posterior = ParticlePosterior(prior, config.n_particles, rng,
                                  rejuvenate=config.rejuvenate)
    state = PolicyState(candidates=candidates, posterior=posterior)
    u = latent_uniforms(seed_base + 1, replicate, config.n_steps)

    xs, ys = [], []
    for k in range(config.n_steps):
        x, _ = policy.select(state, rng)
        y = int(u[k] < float(np.atleast_1d(curve.prob(np.array([x])))[0]))
        posterior.update(x, y)
        xs.append(x)
        ys.append(y)
        state.x_hist, state.y_hist, state.step = xs, ys, k + 1
        state.invalidate()

    row = {
        "replicate": replicate,
        "policy": config.policy,
        "n": config.n_steps,
        "mu_true_draw": mu_true,
        "sigma_true_draw": sigma_true,
    }
    path = path_summary(xs, ys, candidates)
    row.update(path)
    snap = posterior.snapshot()
    row.update({f"particle_{k}": v for k, v in snap.items()})
    row.update(degeneracy_flags(path, snap))
    row.update(information_summary(xs, ys, posterior.median_rotem("mu"),
                                   max(posterior.median_rotem("sigma"), 1e-6)))

    truths = {}
    for t in config.targets:
        truths[t] = (mu_true if t == "mu" else sigma_true if t == "sigma"
                     else float(curve.quantile(float(t[1:]))))
        row[f"{t}_rank_particle"] = posterior.cdf_at(t, truths[t])
        row[f"{t}_covered_particle"] = posterior.covered(t, truths[t],
                                                         config.credible_level)
        if compute_widths:
            lo, hi = posterior.credible_interval(t, config.credible_level)
            row[f"{t}_width_particle"] = hi - lo

    if use_reference:
        ref = build_reference_posterior(prior, xs, ys, tol=config.reference_tol,
                                        n_max=config.reference_n_max)
        row["ref_converged"] = bool(ref.converged)
        row["ref_final_n"] = int(ref.convergence["final_n"])
        row["ref_boundary_mass"] = float(ref.convergence["boundary_mass"])
        row["ref_max_delta"] = float(ref.convergence["max_relative_delta"])
        for t in config.targets:
            row[f"{t}_rank_ref"] = ref.cdf_at(t, truths[t])
            row[f"{t}_covered_ref"] = ref.covered(t, truths[t], config.credible_level)
            if compute_widths:
                lo, hi = ref.credible_interval(t, config.credible_level)
                row[f"{t}_width_ref"] = hi - lo
    return row


def uniformity_report(ranks, n_bins: int = 20) -> dict:
    """Test a batch of SBC rank statistics against Uniform(0, 1)."""
    r = np.asarray(ranks, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 2:
        return {"n": n, "ks_stat": np.nan, "ks_p": np.nan,
                "chi2_stat": np.nan, "chi2_p": np.nan, "mean": np.nan}
    ks = stats.kstest(r, "uniform")
    counts, _ = np.histogram(r, bins=n_bins, range=(0, 1))
    expected = n / n_bins
    chi2 = float(np.sum((counts - expected) ** 2) / expected)
    chi2_p = float(stats.chi2.sf(chi2, n_bins - 1))
    return {
        "n": n,
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "chi2_stat": chi2,
        "chi2_p": chi2_p,
        "mean": float(r.mean()),
        "mean_se": float(r.std(ddof=1) / np.sqrt(n)),
        "bin_counts": counts.tolist(),
        "expected_per_bin": expected,
        # 95% simultaneous band for the ECDF deviation under uniformity
        "ks_band_95": float(1.358 / np.sqrt(n)),
    }


def coverage_report(covered, level: float = 0.95) -> dict:
    c = np.asarray(covered, dtype=bool)
    n = len(c)
    k = int(c.sum())
    p = k / n if n else np.nan
    se = float(np.sqrt(max(p * (1 - p), 0) / n)) if n else np.nan
    return {
        "n": n,
        "coverage": p,
        "se": se,
        "nominal": level,
        "z": (p - level) / se if se and se > 0 else np.nan,
    }


def stratified_coverage(rows, target: str, by: str, level: float = 0.95,
                        covered_key: str | None = None) -> list[dict]:
    """Coverage within strata of an observable path summary."""
    key = covered_key or f"{target}_covered"
    out = []
    values = sorted({r[by] for r in rows if by in r})
    for v in values:
        sel = [r for r in rows if r.get(by) == v and key in r]
        if not sel:
            continue
        rep = coverage_report([r[key] for r in sel], level)
        rep.update({"stratum": by, "value": v, "target": target})
        out.append(rep)
    return out


def quantile_bin(values, n_bins: int = 4, labels=None):
    """Bin a continuous diagnostic into quantile strata for stratification."""
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    edges = np.quantile(v[finite], np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, n_bins - 1)
    if labels is None:
        labels = [f"Q{i+1}" for i in range(n_bins)]
    return [labels[i] if finite[j] else "missing" for j, i in enumerate(idx)]
