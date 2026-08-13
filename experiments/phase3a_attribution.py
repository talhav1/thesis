"""Phase 3A attribution: is the particle-vs-reference gap Monte Carlo error?

The Phase 3 gate requires any failure mechanism to be attributable to
adaptivity, misspecification or weak exploration *rather than to implementation
error*.  For a particle approximation there is a decisive test: if the gap is
genuine Monte Carlo degeneracy, it must shrink as the particle count grows, at
roughly the Monte Carlo rate.  If it is a bug, more particles will not help.

Two things are varied on the *same* histories:

* `n_particles` in {10k, 40k, 160k} -- if the gap is degeneracy, it shrinks;
* `rejuvenate` on/off, and the KL threshold -- which isolates how much of the
  degeneracy Rotem's rejuvenation step is failing to repair.

The histories themselves are held fixed (generated once at N = 10,000, the
production setting), so the design is identical across arms and only the
inference changes.
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
from phase3a_pilot import generate_history  # noqa: E402

from src.posterior_grid import build_reference_posterior  # noqa: E402
from src.priors import rotem_prior  # noqa: E402
from src.response_models import ProbitCurve  # noqa: E402
from src.rotem_particles import ParticlePosterior  # noqa: E402

SEED_BASE = 20260903
N_STEPS = 30
TARGETS = ("q0.5", "q0.95", "q0.99")
LEVEL = 0.95
N_HISTORIES = int(sys.argv[1]) if len(sys.argv) > 1 else 60
CELLS = ((30.0, 0.3), (22.0, 0.3), (38.0, 0.3), (30.0, 3.0))
ARMS = (
    ("N=10k rejuv", 10_000, True, 0.1),
    ("N=40k rejuv", 40_000, True, 0.1),
    ("N=160k rejuv", 160_000, True, 0.1),
    ("N=10k no-rejuv", 10_000, False, np.inf),
    ("N=10k rejuv KL=0.02", 10_000, True, 0.02),
)


def _worker(args):
    mu, sigma, r = args
    x, y, _, _, _ = generate_history(mu, sigma, r, SEED_BASE, n_steps=N_STEPS)
    prior = rotem_prior("well", "independent")
    curve = ProbitCurve(mu, sigma)
    truths = {t: float(curve.quantile(float(t[1:]))) for t in TARGETS}

    ref = build_reference_posterior(prior, x, y, fixed_n=513)
    overlap = bool((y == 0).any() and (y == 1).any()
                   and x[y == 0].max() > x[y == 1].min())

    rows = []
    for label, n_part, rejuv, kl in ARMS:
        # SeedSequence entropy must be non-negative; encode "no rejuvenation"
        # (kl = inf) as 0 rather than -1.
        kl_seed = int(kl * 1000) if np.isfinite(kl) else 0
        rng = np.random.default_rng([SEED_BASE, r, n_part, kl_seed])
        post = ParticlePosterior(prior, n_part, rng, kl_threshold=kl,
                                 rejuvenate=rejuv)
        for xi, yi in zip(x, y):
            post.update(float(xi), int(yi))
        row = {"mu_true": mu, "sigma_true": sigma, "replicate": r, "arm": label,
               "n_particles": n_part, "rejuvenate": rejuv, "kl_threshold": kl,
               "overlap": overlap, "ess": post.ess(),
               "max_weight": float(post.w.max()),
               "n_resamples": post.diag.n_resamples,
               "n_invalid": post.diag.n_invalid_particles}
        for t in TARGETS:
            row[f"{t}_p_covered"] = post.covered(t, truths[t], LEVEL)
            row[f"{t}_r_covered"] = ref.covered(t, truths[t], LEVEL)
            row[f"{t}_cov_diff"] = int(row[f"{t}_p_covered"]) - int(row[f"{t}_r_covered"])
            row[f"{t}_rank_diff"] = post.cdf_at(t, truths[t]) - ref.cdf_at(t, truths[t])
            row[f"{t}_shift_in_ref_sd"] = (post.mean(t) - ref.mean(t)) / ref.sd(t)
            row[f"{t}_sd_ratio"] = post.sd(t) / ref.sd(t)
        rows.append(row)
    return rows


def main():
    t0 = time.time()
    tasks = [(m, s, r) for m, s in CELLS for r in range(N_HISTORIES)]
    print(f"Phase 3A attribution: {len(CELLS)} cells x {N_HISTORIES} histories "
          f"x {len(ARMS)} inference arms", flush=True)
    rows, done = [], 0
    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        for res in pool.imap_unordered(_worker, tasks):
            rows.extend(res)
            done += 1
            if done % 20 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"  [attr] {done}/{len(tasks)} {el:6.1f}s "
                      f"(eta {el/done*(len(tasks)-done):6.1f}s)", flush=True)
    df = pd.DataFrame(rows)

    summ = []
    for (arm, sigma), g in df.groupby(["arm", "sigma_true"]):
        for t in TARGETS:
            cd = g[f"{t}_cov_diff"].to_numpy()
            summ.append({
                "arm": arm, "sigma_true": sigma, "target": t, "n": len(g),
                "n_particles": int(g.n_particles.iloc[0]),
                "cov_particle": float(g[f"{t}_p_covered"].mean()),
                "cov_reference": float(g[f"{t}_r_covered"].mean()),
                "cov_diff": float(cd.mean()),
                "cov_diff_se": float(cd.std(ddof=1) / np.sqrt(len(g))),
                "abs_rank_diff": float(g[f"{t}_rank_diff"].abs().mean()),
                "abs_shift_ref_sd": float(g[f"{t}_shift_in_ref_sd"].abs().mean()),
                "sd_ratio": float(g[f"{t}_sd_ratio"].mean()),
                "ess": float(g.ess.mean()),
                "ess_per_particle": float((g.ess / g.n_particles).mean()),
                "n_resamples": float(g.n_resamples.mean()),
            })
    sdf = pd.DataFrame(summ)
    path = save_csv(sdf, "phase3a_attribution")
    save_csv(df.drop(columns=[]), "phase3a_attribution_raw")

    print("\n=== q0.95, by arm and true sigma ===")
    sel = sdf[sdf.target == "q0.95"].sort_values(["sigma_true", "n_particles", "arm"])
    print(sel[["sigma_true", "arm", "n", "cov_particle", "cov_reference", "cov_diff",
               "cov_diff_se", "abs_rank_diff", "sd_ratio", "ess", "n_resamples"]]
          .to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    print("\n=== Monte Carlo scaling check (rejuvenating arms, sigma=0.3) ===")
    sc = sdf[(sdf.sigma_true == 0.3) & sdf.arm.str.contains("rejuv")
             & ~sdf.arm.str.contains("no-rejuv") & ~sdf.arm.str.contains("KL")]
    for t in TARGETS:
        s = sc[sc.target == t].sort_values("n_particles")
        if len(s) >= 2:
            r = s.abs_rank_diff.to_numpy()
            print(f"  {t}: |rank diff| {np.round(r,4)} at N={list(s.n_particles)} "
                  f"-> ratio {r[0]/max(r[-1],1e-12):.2f}x for a {s.n_particles.iloc[-1]//s.n_particles.iloc[0]}x "
                  f"particle increase (Monte Carlo would predict "
                  f"{np.sqrt(s.n_particles.iloc[-1]/s.n_particles.iloc[0]):.2f}x)")

    write_manifest(
        experiment="phase3a_attribution",
        config={"cells": [list(c) for c in CELLS], "n_histories": N_HISTORIES,
                "arms": [{"label": a[0], "n_particles": a[1], "rejuvenate": a[2],
                          "kl_threshold": None if not np.isfinite(a[3]) else a[3]}
                         for a in ARMS],
                "n_steps": N_STEPS, "reference_fixed_n": 513},
        seed_base=SEED_BASE, requested=len(tasks), completed=int(df.replicate.nunique()
                                                                * len(CELLS)),
        failures=[], tolerances={"reference_fixed_n": 513},
        mc_se={f"cov_diff::{r.arm}::sigma{r.sigma_true}::{r.target}": r.cov_diff_se
               for r in sdf.itertuples()},
        degeneracies={"min_ess": float(df.ess.min()),
                      "max_weight_max": float(df.max_weight.max())},
        artifacts=[path], wall=time.time() - t0,
        notes="Attribution of the Phase 3A gap: particle-count scaling and "
              "rejuvenation ablation on identical histories.",
    )


if __name__ == "__main__":
    main()
