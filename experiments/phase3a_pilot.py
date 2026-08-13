"""Phase 3A stage 1: where do no-overlap paths come from?

Under the adaptive design with a correct probit model, a history with no
overlapping pattern occurred in 2.3% of Phase 2's prior-predictive draws.
Generating 2,000 of them by brute force would need ~87,000 histories, which is
not affordable.  The protocol therefore allows stratified enrichment: find the
(mu, sigma) regions where the event is common, sample those, and report the
event rate per stratum together with the totals.

This pilot only *generates histories* -- no reference posterior -- so it is
cheap.  Its output is an allocation table, not a result.
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

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _runner import N_WORKERS, save_csv, write_manifest  # noqa: E402

from src.diagnostics import path_summary  # noqa: E402
from src.policies import PolicyState, build_policy  # noqa: E402
from src.priors import rotem_prior, rotem_stimulus_grid  # noqa: E402
from src.response_models import ProbitCurve  # noqa: E402
from src.rotem_particles import ParticlePosterior  # noqa: E402
from src.simulator import latent_uniforms  # noqa: E402

SEED_BASE = 20260901
N_STEPS = 30
N_PARTICLES = 10_000
# Guarded so that importing this module from another script (which has its
# own argv) cannot pick up an unrelated command-line argument.
def _arg_int(default):
    if len(sys.argv) > 1 and Path(sys.argv[0]).name == "phase3a_pilot.py":
        return int(sys.argv[1])
    return default


REPS_PER_CELL = _arg_int(30)
MU_GRID = (14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0)
SIGMA_GRID = (0.3, 0.6, 1.2, 3.0, 6.0, 12.0)


def generate_history(mu_true, sigma_true, replicate, seed_base, n_steps=N_STEPS,
                     policy_name="entropy_vector", calibration="well",
                     n_particles=N_PARTICLES):
    """Run the adaptive design once; return the history and particle state.

    Returns the full (x, y) path plus the particle posterior, so a caller can
    decide *after the fact* whether to spend a reference posterior on it.
    """
    rng = np.random.default_rng([seed_base, replicate, int(mu_true * 100),
                                 int(sigma_true * 100)])
    prior = rotem_prior(calibration, "independent")
    candidates = rotem_stimulus_grid(calibration, 200, 14.0)
    curve = ProbitCurve(mu_true, sigma_true)
    policy = build_policy(policy_name)
    posterior = ParticlePosterior(prior, n_particles, rng)
    state = PolicyState(candidates=candidates, posterior=posterior)
    u = latent_uniforms(seed_base, replicate, n_steps)

    xs, ys = [], []
    for k in range(n_steps):
        x, _ = policy.select(state, rng)
        y = int(u[k] < float(np.atleast_1d(curve.prob(np.array([x])))[0]))
        posterior.update(x, y)
        xs.append(x)
        ys.append(y)
        state.x_hist, state.y_hist, state.step = xs, ys, k + 1
        state.invalidate()
    return np.asarray(xs), np.asarray(ys, dtype=np.int8), posterior, prior, candidates


def _worker(args):
    mu, sigma, start, stop = args
    rows = []
    for r in range(start, stop):
        x, y, post, _, cand = generate_history(mu, sigma, r, SEED_BASE)
        ps = path_summary(x, y, cand)
        rows.append({"mu_true": mu, "sigma_true": sigma, "replicate": r,
                     "overlap": ps["overlap"], "all_zero": ps["all_zero"],
                     "all_one": ps["all_one"], "n_positive": ps["n_positive"],
                     "x_range": ps["x_range"], "ess": post.ess()})
    return rows


def main():
    tasks = [(m, s, 0, REPS_PER_CELL) for m in MU_GRID for s in SIGMA_GRID]
    print(f"Phase 3A pilot: {len(tasks)} (mu,sigma) cells x {REPS_PER_CELL} reps, "
          f"n={N_STEPS}", flush=True)
    rows, t0, done = [], time.time(), 0
    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        for res in pool.imap_unordered(_worker, tasks):
            rows.extend(res)
            done += 1
            if done % 6 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"  [pilot] {done}/{len(tasks)} {el:6.1f}s "
                      f"(eta {el/done*(len(tasks)-done):6.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    df["no_overlap"] = ~df.overlap.astype(bool)
    rate = (df.groupby(["mu_true", "sigma_true"], as_index=False)
              .agg(n=("no_overlap", "size"), n_no_overlap=("no_overlap", "sum"),
                   all_equal=("all_zero", "sum")))
    rate["rate"] = rate.n_no_overlap / rate.n
    rate["se"] = np.sqrt(rate.rate * (1 - rate.rate) / rate.n)
    path = save_csv(rate, "phase3a_pilot_rates")

    print("\nno-overlap rate by (mu, sigma):")
    print(rate.pivot(index="sigma_true", columns="mu_true", values="rate")
          .to_string(float_format=lambda v: f"{v:5.2f}"))
    print(f"\noverall rate {df.no_overlap.mean():.4f} over {len(df)} histories; "
          f"cells with rate >= 0.15: {int((rate.rate >= 0.15).sum())}/{len(rate)}")

    write_manifest(
        experiment="phase3a_pilot",
        config={"mu_grid": list(MU_GRID), "sigma_grid": list(SIGMA_GRID),
                "reps_per_cell": REPS_PER_CELL, "n_steps": N_STEPS,
                "policy": "entropy_vector", "n_particles": N_PARTICLES,
                "calibration": "well"},
        seed_base=SEED_BASE,
        requested=len(tasks) * REPS_PER_CELL, completed=int(len(df)), failures=[],
        tolerances={"note": "history generation only; no posterior evaluated"},
        mc_se={f"rate::mu{r.mu_true}_sigma{r.sigma_true}": r.se for r in rate.itertuples()},
        degeneracies={"n_no_overlap": int(df.no_overlap.sum()),
                      "n_all_zero": int(df.all_zero.sum()),
                      "n_all_one": int(df.all_one.sum())},
        artifacts=[path], wall=time.time() - t0,
        notes="Allocation pilot for the Phase 3A stratified enrichment.",
    )


if __name__ == "__main__":
    main()
