"""Does the section 5.3 scalar-MI estimator stay accurate *after* the prior?

D13 rests on an audit (`phase1_quadrature_audit.py`) that is exact only at step
zero: the closed form for I(sigma; Y | x) needs mu ~ N(mu0, s_mu^2) independent
of sigma, and the first observation destroys that independence.  So the
published "correlation 0.996, efficiency 1.000" establishes the estimator at
the prior and says nothing about the other twenty-nine steps of the run.

That gap matters because D14 measures the sigma criterion at -27.7% bias with
36.5% of estimates negative at Rotem's own settings.  A criterion that small
relative to its own noise could be producing an essentially arbitrary argmax
once the posterior moves, and the edge-pinning that D13 attributes to the
mathematics would then be partly *our* estimator failing.

This script closes the gap by replacing the closed form with the reference grid
posterior, which stays exact after data arrive.  Under a product grid over
(mu, eta = log sigma),

    P(Y=1 | sigma_j, x) = sum_i p(mu_i | sigma_j) Phi((x - mu_i)/sigma_j)

is a weighted sum over the grid, so I(sigma; Y | x) is quadrature-exact with no
density estimation anywhere.  At step 0 it reproduces the closed form to 4e-5
relative, which is the check that licenses using it at steps 5, 10, 20 and 30.

Two policies are audited.  `entropy_sigma` is the one under suspicion, but its
history is degenerate by construction, so `entropy_vector` supplies a control
where the posterior localises normally: if the estimator degrades on both, the
fault is the estimator; if only on the pinned history, it is the geometry.

The decisive column is `efficiency` -- the exact criterion value at the
stimulus the estimator picked, over the exact maximum.  1.0 means the estimator
chose as well as an oracle; near 0 means the design is being driven by noise.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from _cli import parse as _parse_cli  # noqa: E402
from _runner import save_csv, write_manifest  # noqa: E402

from src.policies import EntropyScalarPolicy, PolicyState, build_policy  # noqa: E402
from src.posterior_grid import build_reference_posterior  # noqa: E402
from src.priors import rotem_prior, rotem_stimulus_grid  # noqa: E402
from src.response_models import binary_entropy  # noqa: E402
from src.rotem_particles import ParticlePosterior  # noqa: E402
from src.simulator import ProbitCurve, latent_uniforms  # noqa: E402

RUN = _parse_cli("phase1_sigma_criterion_drift", replicates=20, seed_base=20260814)

AUDIT_STEPS = (0, 5, 10, 20, 30)
N_STEPS = 30
POLICIES = ("entropy_sigma", "entropy_vector")
MU_TRUE, SIGMA_TRUE = 30.0, 3.0
N_PARTICLES = 40_000
GRID_POINTS = 80
GRID_HALF_WIDTH = 14.0
REF_N = 513
R, C_FRACTION = 100, 0.1          # Rotem's stated settings
EDGE_BAND = 0.05                  # within 5% of the candidate range = an edge design


def grid_mi_sigma(post, xs):
    """Exact I(sigma; Y | x) under a product-grid posterior.

    Quadrature only -- no kernel, no density ratio.  This is the reference the
    section 5.3 estimator is scored against.
    """
    n_mu, n_eta = post.spec.n_mu, post.spec.n_eta
    W = post.w.reshape(n_mu, n_eta)
    W = W / W.sum()
    mu_c = post.mu.reshape(n_mu, n_eta)[:, 0]
    sg_c = post.sigma.reshape(n_mu, n_eta)[0, :]
    w_sig = W.sum(axis=0)
    Wn = W / np.where(w_sig > 0, w_sig, 1.0)      # p(mu | sigma_j)

    out = np.empty(len(xs))
    for k, xv in enumerate(xs):
        P_sig = (Wn * norm.cdf((xv - mu_c[:, None]) / sg_c[None, :])).sum(axis=0)
        out[k] = binary_entropy(float(w_sig @ P_sig)) - float(w_sig @ binary_entropy(P_sig))
    return out


def audit_state(state, post, cands):
    """Score the section 5.3 estimator against the grid reference."""
    est = EntropyScalarPolicy("sigma", R=R, c_fraction=C_FRACTION,
                              clip_negative=False)._scores(state).mi
    ref = grid_mi_sigma(post, cands)

    i_est, i_ref = int(np.argmax(est)), int(np.argmax(ref))
    ref_max = float(ref.max())
    # efficiency: how good is the estimator's pick, judged by the exact criterion
    eff = float(ref[i_est] / ref_max) if ref_max > 0 else float("nan")
    sd_e, sd_r = est.std(), ref.std()
    corr = float(np.corrcoef(est, ref)[0, 1]) if sd_e > 0 and sd_r > 0 else float("nan")

    # "at the edge" must be a band, not an exact index: the optimum drifts a
    # node or two inward while still being an edge design in every sense that
    # matters.  Distance is reported as a fraction of the candidate range.
    lo, hi = float(cands[0]), float(cands[-1])
    span = hi - lo

    def edge_frac(xv):
        return float(min(xv - lo, hi - xv) / span)

    return {
        "corr": corr,
        "efficiency": eff,
        "frac_negative": float((est < 0).mean()),
        "bias": float(est.mean() - ref.mean()),
        "est_max": float(est.max()),
        "ref_max": ref_max,
        "x_est": float(cands[i_est]),
        "x_ref": float(cands[i_ref]),
        "edge_frac_est": edge_frac(float(cands[i_est])),
        "edge_frac_ref": edge_frac(float(cands[i_ref])),
        "edge_est": bool(edge_frac(float(cands[i_est])) <= EDGE_BAND),
        "edge_ref": bool(edge_frac(float(cands[i_ref])) <= EDGE_BAND),
        "ref_at_mu0": float(ref[int(np.argmin(np.abs(cands - MU_TRUE)))]),
    }


def run_one(policy_name, replicate, prior, cands, curve):
    rng = np.random.default_rng([RUN.seed_base, replicate])
    crn = RUN.crn_seed if RUN.crn_seed is not None else RUN.seed_base
    u = latent_uniforms(crn, replicate, N_STEPS)

    policy = build_policy(policy_name, R=R, c_fraction=C_FRACTION) \
        if policy_name.startswith("entropy_") and "vector" not in policy_name \
        else build_policy(policy_name)
    posterior = ParticlePosterior(prior, N_PARTICLES, rng, rejuvenate=True)
    state = PolicyState(candidates=cands, posterior=posterior)

    xs, ys, rows = [], [], []
    for k in range(N_STEPS + 1):
        if k in AUDIT_STEPS:
            post = build_reference_posterior(
                prior, np.asarray(xs, float), np.asarray(ys, int), fixed_n=REF_N)
            rec = audit_state(state, post, cands)
            rec.update({"policy": policy_name, "replicate": replicate, "step": k,
                        "posterior_sd_mu": float(posterior.sd("mu")),
                        "posterior_sd_sigma": float(posterior.sd("sigma"))})
            rows.append(rec)
        if k == N_STEPS:
            break

        x, _ = policy.select(state, rng)
        y = int(u[k] < float(np.atleast_1d(curve.prob(np.array([x])))[0]))
        posterior.update(x, y)
        xs.append(x)
        ys.append(y)
        state.x_hist, state.y_hist, state.step = xs, ys, k + 1
        state.invalidate()
    return rows


def main():
    t0 = time.time()
    prior = rotem_prior("well", "independent")
    cands = rotem_stimulus_grid("well", n_points=GRID_POINTS,
                                half_width=GRID_HALF_WIDTH)
    curve = ProbitCurve(MU_TRUE, SIGMA_TRUE)

    rows, failures = [], []
    for rep in range(RUN.replicates):
        for pol in POLICIES:
            try:
                rows.extend(run_one(pol, rep, prior, cands, curve))
            except Exception as exc:  # noqa: BLE001
                failures.append({"replicate": rep, "policy": pol, "error": repr(exc)})
        print(f"  replicate {rep + 1}/{RUN.replicates}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    raw = pd.DataFrame(rows)
    raw_path = save_csv(raw, "phase1_sigma_criterion_drift_raw")

    agg = (raw.groupby(["policy", "step"])
              .agg(n=("efficiency", "size"),
                   efficiency_mean=("efficiency", "mean"),
                   efficiency_se=("efficiency", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                   efficiency_min=("efficiency", "min"),
                   corr_mean=("corr", "mean"),
                   frac_negative=("frac_negative", "mean"),
                   bias_mean=("bias", "mean"),
                   ref_max_mean=("ref_max", "mean"),
                   edge_est_rate=("edge_est", "mean"),
                   edge_ref_rate=("edge_ref", "mean"),
                   edge_frac_ref_med=("edge_frac_ref", "median"),
                   x_ref_med=("x_ref", "median"),
                   sd_mu_mean=("posterior_sd_mu", "mean"),
                   sd_sigma_mean=("posterior_sd_sigma", "mean"))
              .reset_index())
    sum_path = save_csv(agg, "phase1_sigma_criterion_drift")

    print()
    print(agg.to_string(index=False))

    write_manifest(
        experiment="phase1_sigma_criterion_drift",
        config={"audit_steps": list(AUDIT_STEPS), "n_steps": N_STEPS,
                "policies": list(POLICIES), "mu_true": MU_TRUE,
                "sigma_true": SIGMA_TRUE, "n_particles": N_PARTICLES,
                "grid_points": GRID_POINTS, "grid_half_width": GRID_HALF_WIDTH,
                "reference_n": REF_N, "R": R, "c_fraction": C_FRACTION},
        seed_base=RUN.seed_base,
        requested=RUN.replicates,
        completed=int(raw.replicate.nunique()) if len(raw) else 0,
        failures=failures,
        tolerances={"reference_vs_closed_form_rel": 3.9e-5, "edge_band": EDGE_BAND},
        mc_se={"efficiency_se": float(agg.efficiency_se.max())},
        degeneracies={"n_rows": int(len(raw))},
        artifacts=[{"role": "raw", "path": str(raw_path)},
                   {"role": "summary", "path": str(sum_path)}],
        wall=time.time() - t0,
        notes="Extends the step-0 quadrature audit past the prior using the "
              "reference grid posterior; closes the D13 gap on whether "
              "edge-pinning is the criterion or the estimator.",
        **RUN.manifest_kwargs(),
    )


if __name__ == "__main__":
    main()
