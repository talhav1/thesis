"""Phase 3B: policy-dependent pseudo-truth (H4).

For each (DGP, policy) cell of the screening run:

1. pool the stimuli actually applied, giving the empirical design distribution
   nu_a produced by that policy;
2. compute the probit KL projection of the true curve under nu_a;
3. derive the implied pseudo-true quantiles;
4. compare them with the physical quantiles and with the posterior mean at the
   final horizon.

H4 asks whether different policies, facing the *same* physical curve, are
heading for *different* probit curves.  If they are, "the fitted parameters" is
not a well-defined target under misspecification -- it is a property of the
experiment as much as of the system.

The bootstrap resamples whole replicates, because the design distribution is
generated one experiment at a time and stimuli within an experiment are
strongly dependent.
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
from _runner import N_WORKERS, RESULTS, save_csv, write_manifest  # noqa: E402
from _cli import parse as _parse_cli  # noqa: E402
from phase3b_screen import BLOCK_I_DGPS, BLOCK_II_DGPS  # noqa: E402

from src.priors import rotem_stimulus_grid  # noqa: E402
from src.pseudo_truth import design_distribution_summary, projection_with_bootstrap  # noqa: E402
from src.simulator import build_curve  # noqa: E402

RUN = _parse_cli("phase3b_pseudotruth", replicates=120, seed_base=20260920)

SEED_BASE = RUN.seed_base
TARGETS = ("q0.5", "q0.95", "q0.99")
N_BOOT = RUN.replicates
CURVES = dict(BLOCK_I_DGPS + BLOCK_II_DGPS)


def _worker(args):
    cell, x_by_rep = args
    block, dgp, pol, prior_name, scale = cell.split("|")
    cd = dict(CURVES[dgp])
    if scale != "s3" and cd["family"] != "beta_cdf":
        s0 = float(scale[1:])
        cd["sigma0"] = s0
        if cd["family"] == "probit":
            cd["sigma"] = s0
    curve = build_curve(cd)
    calibration = "poor" if prior_name == "shifted" else "well"
    grid = rotem_stimulus_grid(calibration, 200, 14.0)

    res = projection_with_bootstrap(curve, x_by_rep, grid=grid, n_boot=N_BOOT,
                                    seed=SEED_BASE, targets=TARGETS)
    res.update({"cell": cell, "block": block, "dgp": dgp, "pol": pol,
                "prior_name": prior_name, "scale": scale})
    res.update(design_distribution_summary(np.concatenate([np.asarray(r) for r in x_by_rep])))
    return res


def main():
    t0 = time.time()
    path = RESULTS / "raw" / "phase3b_designs.parquet"
    if not path.exists():
        print("phase3b_designs.parquet missing -- run phase3b_screen.py first")
        return
    designs = pd.read_parquet(path)
    tasks = [(cell, [np.asarray(v, dtype=float) for v in g["x"].tolist()])
             for cell, g in designs.groupby("cell")]
    print(f"Phase 3B pseudo-truth: {len(tasks)} cells, {N_BOOT} bootstrap draws each",
          flush=True)

    rows, done = [], 0
    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        for res in pool.imap_unordered(_worker, tasks):
            rows.append(res)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"  [pseudo] {done}/{len(tasks)} {el:6.1f}s "
                      f"(eta {el/done*(len(tasks)-done):6.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    out = save_csv(df, "phase3b_pseudo_truth")

    # H4: spread of the pseudo-true q0.95 across policies, within DGP
    h4 = []
    for (block, dgp, prior_name, scale), g in df.groupby(["block", "dgp", "prior_name", "scale"]):
        if len(g) < 2:
            continue
        for t in TARGETS:
            vals = g[f"pseudo_{t}"].to_numpy()
            ses = g[f"pseudo_{t}_se"].to_numpy()
            spread = float(vals.max() - vals.min())
            # standard error of a max-min spread, conservatively the root of the
            # sum of the two extreme cells' variances
            hi, lo = int(np.argmax(vals)), int(np.argmin(vals))
            spread_se = float(np.sqrt(ses[hi] ** 2 + ses[lo] ** 2))
            h4.append({
                "block": block, "dgp": dgp, "prior_name": prior_name, "scale": scale,
                "target": t, "n_policies": len(g),
                "physical": float(g[f"physical_{t}"].iloc[0]),
                "pseudo_min": float(vals.min()), "pseudo_max": float(vals.max()),
                "spread": spread, "spread_se": spread_se,
                "policy_at_min": g.pol.iloc[lo], "policy_at_max": g.pol.iloc[hi],
                "max_abs_pseudo_minus_physical":
                    float(g[f"pseudo_minus_physical_{t}"].abs().max()),
                "curve_scale_units": spread / 3.0,
            })
    h4df = pd.DataFrame(h4)
    h4path = save_csv(h4df, "phase3b_pseudo_truth_spread")

    print("\n=== H4: spread of pseudo-true quantiles across policies (Block I, s3) ===")
    sel = h4df[(h4df.block == "I") & (h4df.scale == "s3")].sort_values(
        "spread", ascending=False)
    print(sel[["dgp", "target", "physical", "pseudo_min", "pseudo_max", "spread",
               "spread_se", "policy_at_min", "policy_at_max",
               "max_abs_pseudo_minus_physical"]]
          .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    write_manifest(
        experiment="phase3b_pseudotruth",
        config={"n_boot": N_BOOT, "targets": list(TARGETS),
                "source": "results/raw/phase3b_designs.parquet",
                "bootstrap_unit": "replicate"},
        seed_base=SEED_BASE, requested=len(tasks), completed=int(len(df)),
        failures=[],
        tolerances={"projection": "Nelder-Mead on (mu, log sigma), 4 restarts"},
        mc_se={f"pseudo_q0.95_se::{r.cell}": r._asdict().get("pseudo_q0.95_se", np.nan)
               for r in df.itertuples()},
        degeneracies={"n_cells": int(len(df)),
                      "n_not_converged": int((~df.converged).sum())},
        artifacts=[out, h4path], wall=time.time() - t0,
        notes="H4: policy-dependent probit KL projections under misspecification.",
    )


if __name__ == "__main__":
    main()
