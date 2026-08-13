"""Phase 3B confirmatory tier.

Cells chosen by the rule frozen in protocol section 6, applied mechanically to
the screening output:

* **Cell A** -- the Block I (DGP, adaptive policy) cell with the largest screened
  false-certainty rate for q_0.95 or q_0.99 at n = 50.
* **Cell B** -- the same DGP with the non-adaptive `fixed_design` policy.  This
  is the contrast that decides whether any failure is attributable to
  *adaptivity* or merely to the DGP.
* **Cell C** -- correct probit with Cell A's policy: the control that fixes the
  false-certainty rate under a correct model at the same horizon.

Common random numbers are shared across the three cells at each replicate
index, so the A-vs-B contrast is paired.

Thresholds are read from the frozen `configs/phase3_thresholds.json`; they are
not recomputed here.
"""

from __future__ import annotations

import json
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
from _runner import N_WORKERS, RESULTS, _single, save, save_csv, write_manifest  # noqa: E402
from phase3b_screen import (  # noqa: E402
    BLOCK_I_DGPS,
    CURVE_SCALE,
    curve_scale_lookup,
    make_config,
)

from src.priors import rotem_prior  # noqa: E402

SEED_BASE = 20260940
CRN_SEED = 414001
TARGETS = ("q0.5", "q0.95", "q0.99")
N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
BATCH = 100
TARGET_SE = 0.008
CELL_BUDGET_S = float(os.environ.get("CELL_BUDGET_S", "1500"))
THRESHOLDS = json.loads((RESULTS.parent / "configs" / "phase3_thresholds.json").read_text())
CURVES = dict(BLOCK_I_DGPS)


def select_cells():
    """Apply the protocol's frozen selection rule to the screening summary."""
    s = pd.read_csv(RESULTS / "summaries" / "phase3b_screen_summary.csv")
    sel = s[(s.block == "I") & (s.backend == "ref") & (s.n == 50)
            & s.target.isin(["q0.95", "q0.99"])
            & s.pol.isin(["entropy_vector", "entropy_q0.95"])]
    best = sel.sort_values(["false_certainty", "coverage"],
                           ascending=[False, True]).iloc[0]
    dgp, pol, target = best.dgp, best.pol, best.target
    return (
        [("A", dgp, pol), ("B", dgp, "fixed_design"), ("C", "probit", pol)],
        {"dgp": dgp, "policy": pol, "target": target,
         "screened_false_certainty": float(best.false_certainty),
         "screened_coverage": float(best.coverage)},
    )


def _worker(args):
    label, dgp, pol, start, stop = args
    prior = rotem_prior("well", "independent").spec
    cfg = make_config(CURVES[dgp], pol, "well", prior)
    rows, failures = [], []
    for r in range(start, stop):
        rr, ff, _ = _single(cfg, r, SEED_BASE, CRN_SEED)
        for row in rr:
            row.update({"label": label, "dgp": dgp, "pol": pol})
        rows.extend(rr)
        failures.extend(ff)
    return rows, failures


def metrics(g, dgp, n):
    from src.decision_loss import loss_curve

    out = {}
    scale = CURVE_SCALE.get((dgp, "s3"), 3.0)
    for t in TARGETS:
        cov = g[f"{t}_ref_covered"].astype(float)
        width = g[f"{t}_ref_width"]
        est = g[f"{t}_ref_mean"]
        truth = g[f"{t}_true"]
        err = est - truth
        nrep = len(g)
        w_star = THRESHOLDS["thresholds"].get(f"{t}|n{n}", np.nan)
        fc = ((~cov.astype(bool)) & (width <= w_star)).astype(float)
        dang = (err < -0.5 * scale).astype(float)
        out[t] = {
            "n_rep": nrep,
            "coverage": float(cov.mean()),
            "coverage_se": float(cov.std(ddof=1) / np.sqrt(nrep)),
            "false_certainty": float(fc.mean()),
            "false_certainty_se": float(fc.std(ddof=1) / np.sqrt(nrep)),
            "dangerous_error": float(dang.mean()),
            "dangerous_error_se": float(dang.std(ddof=1) / np.sqrt(nrep)),
            "bias": float(err.mean()), "rmse": float(np.sqrt((err**2).mean())),
            "mean_width": float(width.mean()), "w_star": w_star,
            "frac_narrow": float((width <= w_star).mean()),
            "loss_curve": loss_curve(est, truth),
        }
    return out


def main():
    global CURVE_SCALE
    import phase3b_screen

    phase3b_screen.CURVE_SCALE = curve_scale_lookup()
    CURVE_SCALE = phase3b_screen.CURVE_SCALE

    t0 = time.time()
    cells, chosen = select_cells()
    print(f"Phase 3B confirmatory. Selection rule picked: {chosen}", flush=True)
    print(f"cells: {cells}, up to {N_REPS} replicates each "
          f"(target SE {TARGET_SE}, budget {CELL_BUDGET_S:.0f}s/cell)", flush=True)

    all_rows, all_fail, stopping = [], [], []
    ctx = mp.get_context("fork")
    for label, dgp, pol in cells:
        cell_rows, done, t_cell = [], 0, time.time()
        reason = "max_replicates"
        for b0 in range(0, N_REPS, BATCH):
            b1 = min(b0 + BATCH, N_REPS)
            chunks = [(label, dgp, pol, s, min(s + 25, b1))
                      for s in range(b0, b1, 25)]
            with ctx.Pool(N_WORKERS) as pool:
                for rr, ff in pool.imap_unordered(_worker, chunks):
                    cell_rows.extend(rr)
                    all_fail.extend(ff)
            done = b1
            g = pd.DataFrame(cell_rows)
            g50 = g[g.n == 50]
            se = metrics(g50, dgp, 50)["q0.99"]["false_certainty_se"]
            el = time.time() - t_cell
            print(f"  [{label}] {done} reps, false-certainty SE {se:.4f}, {el:.0f}s",
                  flush=True)
            if se <= TARGET_SE:
                reason = "target_se_reached"
                break
            if el > CELL_BUDGET_S:
                reason = "cpu_budget"
                break
        stopping.append({"label": label, "dgp": dgp, "pol": pol,
                         "n_replicates": done, "stop_reason": reason,
                         "wall_seconds": time.time() - t_cell})
        all_rows.extend(cell_rows)

    df = pd.DataFrame(all_rows)
    raw = save(df, "phase3b_confirm_raw")

    rows = []
    for (label, dgp, pol), g in df[df.n == 50].groupby(["label", "dgp", "pol"]):
        m = metrics(g, dgp, 50)
        for t, vals in m.items():
            rows.append({"label": label, "dgp": dgp, "pol": pol, "target": t,
                         **{k: v for k, v in vals.items() if k != "loss_curve"}})
    summ = pd.DataFrame(rows)
    spath = save_csv(summ, "phase3b_confirm_summary")
    stop_df = pd.DataFrame(stopping)
    save_csv(stop_df, "phase3b_confirm_stopping")

    print("\n=== confirmatory results (reference posterior, n=50) ===")
    print(summ[summ.target != "q0.5"].to_string(index=False,
                                                float_format=lambda v: f"{v:9.4f}"))
    print("\n=== stopping ===")
    print(stop_df.to_string(index=False))

    # H2: paired A vs B on common random numbers
    print("\n=== H2: adaptive (A) vs non-adaptive (B), paired on CRN ===")
    h2 = []
    for t in ("q0.95", "q0.99"):
        a = df[(df.label == "A") & (df.n == 50)].set_index("replicate")
        b = df[(df.label == "B") & (df.n == 50)].set_index("replicate")
        common = a.index.intersection(b.index)
        w_star = THRESHOLDS["thresholds"][f"{t}|n50"]
        fa = ((~a.loc[common, f"{t}_ref_covered"].astype(bool))
              & (a.loc[common, f"{t}_ref_width"] <= w_star)).astype(float)
        fb = ((~b.loc[common, f"{t}_ref_covered"].astype(bool))
              & (b.loc[common, f"{t}_ref_width"] <= w_star)).astype(float)
        d = (fa - fb).to_numpy()
        h2.append({"target": t, "n_pairs": len(common),
                   "fc_adaptive": float(fa.mean()), "fc_fixed": float(fb.mean()),
                   "paired_diff": float(d.mean()),
                   "paired_se": float(d.std(ddof=1) / np.sqrt(len(d))),
                   "meets_h2_threshold": bool(d.mean() >= 0.05)})
    h2df = pd.DataFrame(h2)
    save_csv(h2df, "phase3b_confirm_h2")
    print(h2df.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    write_manifest(
        experiment="phase3b_confirm",
        config={"cells": cells, "selection": chosen, "n_reps_max": N_REPS,
                "batch": BATCH, "target_se": TARGET_SE,
                "cell_budget_s": CELL_BUDGET_S, "crn_seed": CRN_SEED,
                "thresholds_hash": THRESHOLDS["hash"], "n_steps": 50},
        seed_base=SEED_BASE, requested=N_REPS * len(cells),
        completed=int(len(df[df.n == 50])), failures=all_fail,
        tolerances={"target_false_certainty_se": TARGET_SE,
                    "achieved": {r.label: float(r.n_replicates) for r in stop_df.itertuples()}},
        mc_se={f"{r.label}::{r.target}::false_certainty": r.false_certainty_se
               for r in summ.itertuples()},
        degeneracies={"stopping": stopping,
                      "no_overlap": int((~df.overlap.astype(bool)).sum())},
        artifacts=[raw, spath], wall=time.time() - t0,
        notes="Confirmatory tier: H2 and H3 on the pre-registered cells.",
    )


if __name__ == "__main__":
    main()
