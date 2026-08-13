"""Sequential sensitivity experiment simulator.

Design rules enforced here, from the plan's reproducibility section:

* **Common random numbers.** The latent uniforms u_1..u_n driving the binary
  responses are drawn from a stream keyed only by the replicate index, never
  by the policy.  Two policies compared at the same replicate therefore see
  the same latent draws, and the paired difference in their operating
  characteristics has far smaller Monte Carlo error than the difference of two
  independent estimates.
* **Nothing is discarded.** Degenerate paths (all-zero, all-one, no overlap,
  collapsed particle ESS) and raised exceptions are recorded as outcomes with
  flags; they are never filtered out.  `run_batch` returns the failures
  alongside the successes and the manifest counts both.
* **Slices, not reruns.** A single length-n_max path is recorded at several
  horizons, exactly as Rotem slices one n=50 run at n=20, 30, 50.  This keeps
  the horizon comparison paired.
"""

from __future__ import annotations

import traceback
from dataclasses import asdict, dataclass, field

import numpy as np

from .diagnostics import (
    degeneracy_flags,
    information_summary,
    path_summary,
    target_vs_support,
)
from .estimators import fit_probit_mle
from .policies import BrucetonPolicy, PolicyState, build_policy
from .posterior_grid import build_reference_posterior
from .priors import Prior, PriorSpec, rotem_stimulus_grid
from .response_models import BetaCDFCurve, ProbitCurve
from .rotem_particles import DEFAULT_N_PARTICLES, ParticlePosterior

DEFAULT_TARGETS = ("mu", "sigma", "q0.5", "q0.95", "q0.99")


@dataclass
class ExperimentConfig:
    """Fully declarative description of one simulated experiment family."""

    policy: str
    prior: dict
    curve: dict
    calibration: str = "well"
    n_steps: int = 50
    slices: tuple = (20, 30, 50)
    n_particles: int = DEFAULT_N_PARTICLES
    targets: tuple = DEFAULT_TARGETS
    credible_level: float = 0.95
    rejuvenate: bool = True
    policy_kwargs: dict = field(default_factory=dict)
    grid_points: int = 200
    grid_half_width: float = 14.0
    reference_at_slices: tuple = ()
    reference_tol: float = 1e-3
    reference_n_max: int = 1025
    reference_widths: bool = False
    reference_fixed_n: int | None = None
    bruceton_step: float = 2.0

    def as_dict(self):
        d = asdict(self)
        d["slices"] = list(self.slices)
        d["targets"] = list(self.targets)
        d["reference_at_slices"] = list(self.reference_at_slices)
        return d


def build_prior(spec_dict) -> Prior:
    return Prior(PriorSpec(**spec_dict))


def build_curve(curve_dict):
    """Instantiate a data-generating curve from a plain config dict."""
    from .curve_families import BASE_FAMILIES, TailPerturbedCurve, build_curve_family

    kind = curve_dict["family"]
    if kind == "probit":
        return ProbitCurve(curve_dict["mu"], curve_dict["sigma"])
    if kind == "beta_cdf":
        return BetaCDFCurve(
            curve_dict.get("a", 5.0), curve_dict.get("b", 8.0),
            curve_dict.get("lo", 18.0), curve_dict.get("hi", 48.0),
        )
    if kind == "tail_perturbed":
        return TailPerturbedCurve(
            shift95=curve_dict["shift95"],
            mu0=curve_dict.get("mu0", 30.0),
            sigma0=curve_dict.get("sigma0", 3.0),
            p0=curve_dict.get("p0", 0.85),
            name=curve_dict.get("name", ""),
        )
    if kind in BASE_FAMILIES:
        return build_curve_family(
            kind, curve_dict.get("lam", 1.0),
            mu0=curve_dict.get("mu0", 30.0),
            sigma0=curve_dict.get("sigma0", 3.0),
            gap=curve_dict.get("gap", 0.6),
            name=curve_dict.get("name", ""),
        )
    raise ValueError(f"unknown curve family {kind!r}")


def true_targets(curve, targets) -> dict:
    """The physical values of the estimands under the true curve.

    Under misspecification the fitted family's mu and sigma have no physical
    referent, so the plan's convention is used: 'mu' is the true curve's
    median and 'sigma' the standard deviation of the latent tolerance
    distribution the curve implies.  Quantiles are always the true curve's own
    quantiles, computed numerically.
    """
    from .curve_families import curve_true_sd

    out = {}
    for t in targets:
        if t == "mu":
            out[t] = float(np.atleast_1d(curve.quantile(0.5))[0])
        elif t == "sigma":
            explicit = getattr(curve, "sigma", None)
            if explicit is not None:
                out[t] = float(explicit)
            elif hasattr(curve, "implied_sd"):
                out[t] = float(curve.implied_sd)
            else:
                out[t] = curve_true_sd(curve)
        else:
            out[t] = float(np.atleast_1d(curve.quantile(float(t[1:])))[0])
    return out


def latent_uniforms(crn_seed: int, replicate: int, n: int) -> np.ndarray:
    """Response latents shared across policies at the same replicate index."""
    rng = np.random.default_rng([crn_seed, replicate, 0xC0FFEE])
    return rng.random(n)


def bruceton_start(crn_seed: int, replicate: int, mu0: float, sigma: float) -> float:
    rng = np.random.default_rng([crn_seed, replicate, 0xB0BB1E])
    return float(rng.normal(mu0, sigma))


def run_experiment(config: ExperimentConfig, replicate: int, seed_base: int,
                   crn_seed: int | None = None, record_trajectory: bool = True):
    """Run one sequential experiment and return a flat record dict."""
    crn_seed = seed_base if crn_seed is None else crn_seed
    rng = np.random.default_rng([seed_base, replicate])

    prior = build_prior(config.prior)
    curve = build_curve(config.curve)
    candidates = rotem_stimulus_grid(
        config.calibration, n_points=config.grid_points,
        half_width=config.grid_half_width,
    )
    truths = true_targets(curve, config.targets)

    if config.policy == "bruceton":
        policy = BrucetonPolicy(
            step=config.bruceton_step,
            x0=bruceton_start(crn_seed, replicate, prior.spec.mu0,
                              config.policy_kwargs.get("start_sd", 3.0)),
        )
    else:
        policy = build_policy(config.policy, **config.policy_kwargs)

    posterior = ParticlePosterior(prior, config.n_particles, rng,
                                  rejuvenate=config.rejuvenate)
    state = PolicyState(candidates=candidates, posterior=posterior)
    u = latent_uniforms(crn_seed, replicate, config.n_steps)

    xs, ys = [], []
    policy_diag = []
    records = {}

    for k in range(config.n_steps):
        x, info = policy.select(state, rng)
        p_true = float(np.atleast_1d(curve.prob(np.array([x])))[0])
        y = int(u[k] < p_true)
        posterior.update(x, y)
        xs.append(x)
        ys.append(y)
        state.x_hist = xs
        state.y_hist = ys
        state.step = k + 1
        state.invalidate()
        policy_diag.append({kk: vv for kk, vv in info.items()
                            if not isinstance(vv, np.ndarray)})

        n_done = k + 1
        if n_done in config.slices:
            records[n_done] = _slice_record(
                config, posterior, prior, xs, ys, candidates, truths, policy_diag,
                n_done, curve,
            )

    out = {
        "replicate": replicate,
        "policy": config.policy,
        "calibration": config.calibration,
        "prior_name": config.prior.get("name", ""),
        "curve": config.curve["family"],
        "slices": records,
    }
    if record_trajectory:
        out["x"] = np.asarray(xs)
        out["y"] = np.asarray(ys, dtype=np.int8)
    return out


def _slice_record(config, posterior, prior, xs, ys, candidates, truths,
                  policy_diag, n_done, curve):
    rec = {}
    path = path_summary(xs, ys, candidates)
    rec.update(path)

    mu_med = posterior.median_rotem("mu")
    sig_med = posterior.median_rotem("sigma")
    rec.update(information_summary(xs, ys, mu_med, max(sig_med, 1e-6)))
    snap = posterior.snapshot()
    rec.update({f"particle_{k}": v for k, v in snap.items()})
    rec.update(degeneracy_flags(path, snap))

    for t in config.targets:
        est = posterior.median_rotem(t)
        lo, hi = posterior.credible_interval(t, config.credible_level)
        truth = truths[t]
        rec[f"{t}_bayes"] = est
        rec[f"{t}_mean"] = posterior.mean(t)
        rec[f"{t}_sd"] = posterior.sd(t)
        rec[f"{t}_lo"] = lo
        rec[f"{t}_hi"] = hi
        rec[f"{t}_true"] = truth
        rec[f"{t}_err"] = est - truth
        rec[f"{t}_covered"] = posterior.covered(t, truth, config.credible_level)
        rec[f"{t}_width"] = hi - lo
        rec[f"{t}_rank_particle"] = posterior.cdf_at(t, truth)

    mle = fit_probit_mle(xs, ys)
    rec.update(mle.as_dict())
    for t in config.targets:
        if t.startswith("q"):
            p = float(t[1:])
            rec[f"{t}_mle_err"] = mle.quantile(p) - truths[t]
        elif t == "mu":
            rec["mu_mle_err"] = mle.mu - truths["mu"]
        elif t == "sigma":
            rec["sigma_mle_err"] = mle.sigma - truths["sigma"]

    rec.update(target_vs_support(truths.get("q0.95", np.nan), xs, candidates))

    dens = [d.get("quadrature_density_integral") for d in policy_diag
            if d.get("quadrature_density_integral") is not None]
    if dens:
        rec["quad_density_min"] = float(np.min(dens))
        rec["quad_density_mean"] = float(np.mean(dens))
    empt = [d.get("quadrature_empty_windows") for d in policy_diag
            if d.get("quadrature_empty_windows") is not None]
    if empt:
        rec["quad_empty_max"] = int(np.max(empt))

    if n_done in config.reference_at_slices:
        ref = build_reference_posterior(
            prior, xs, ys, tol=config.reference_tol, n_max=config.reference_n_max,
            fixed_n=config.reference_fixed_n,
        )
        rec["ref_converged"] = ref.converged if ref.converged is None else bool(ref.converged)
        rec["ref_boundary_mass"] = float(ref.convergence["boundary_mass"])
        rec["ref_final_n"] = int(ref.convergence["final_n"])
        rec["ref_max_delta"] = float(ref.convergence["max_relative_delta"])
        for t in config.targets:
            truth = truths[t]
            # Coverage and the rank statistic need one CDF evaluation each.
            # Interval endpoints need a bisection over many such evaluations,
            # which dominates the whole run, so they are opt-in.
            rec[f"{t}_ref_mean"] = ref.mean(t)
            rec[f"{t}_ref_sd"] = ref.sd(t)
            rec[f"{t}_rank_ref"] = ref.cdf_at(t, truth)
            rec[f"{t}_ref_covered"] = ref.covered(t, truth, config.credible_level)
            rec[f"{t}_particle_vs_ref_mean"] = rec[f"{t}_mean"] - rec[f"{t}_ref_mean"]
            rec[f"{t}_particle_vs_ref_sd_ratio"] = (
                rec[f"{t}_sd"] / rec[f"{t}_ref_sd"] if rec[f"{t}_ref_sd"] > 0 else np.nan
            )
            if config.reference_widths:
                # the reference *mean* is already recorded above; a separate
                # median costs another bisection for very little extra
                lo, hi = ref.credible_interval(t, config.credible_level)
                rec[f"{t}_ref_lo"] = lo
                rec[f"{t}_ref_hi"] = hi
                rec[f"{t}_ref_width"] = hi - lo
                rec[f"{t}_particle_vs_ref_width_ratio"] = (
                    rec[f"{t}_width"] / rec[f"{t}_ref_width"]
                    if rec[f"{t}_ref_width"] > 0 else np.nan)
    return rec


def run_batch(config: ExperimentConfig, n_replicates: int, seed_base: int,
              crn_seed: int | None = None, progress=None):
    """Run `n_replicates` experiments; return (rows, failures, trajectories).

    Exceptions are caught per replicate, recorded with a traceback, and
    counted.  A crashed replicate is a datum about the method, not a licence
    to shrink the sample.
    """
    rows, failures, trajectories = [], [], []
    for r in range(n_replicates):
        try:
            res = run_experiment(config, r, seed_base, crn_seed=crn_seed)
        except Exception as exc:  # noqa: BLE001 - deliberate: record, don't hide
            failures.append({
                "replicate": r,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=6),
            })
            continue
        trajectories.append({"replicate": r, "x": res.pop("x", None),
                             "y": res.pop("y", None)})
        for n_done, rec in res["slices"].items():
            row = {
                "replicate": res["replicate"],
                "policy": res["policy"],
                "calibration": res["calibration"],
                "prior_name": res["prior_name"],
                "curve": res["curve"],
                "n": n_done,
            }
            row.update(rec)
            rows.append(row)
        if progress is not None and (r + 1) % progress == 0:
            print(f"    {config.policy}: {r + 1}/{n_replicates}", flush=True)
    return rows, failures, trajectories
