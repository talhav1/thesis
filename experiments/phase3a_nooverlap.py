"""Phase 3A: powered test of H1 -- does the particle approximation degrade on
adaptively generated no-overlap histories?

Design, following the protocol:

* **Stratified enrichment.** The pilot shows the event is essentially confined
  to two regimes: very steep curves (sigma ~ 0.3, truth inside the grid) and a
  truth at or outside the grid edge.  Sampling is concentrated there.  The
  event rate per stratum and the total number of histories generated are both
  reported, so the enrichment is fully visible.
* **Retain everything.** Every no-overlap history is kept.  Controls are drawn
  from the overlap histories of the *same* stratum, size-matched, so the
  comparison is not confounded by (mu, sigma).
* **Paired inference.** The reference and particle posteriors run on the
  *identical* retained histories.  H1 is about that paired difference; the
  marginal level of conditional coverage is not evidence for or against it.

Pass 1 generates histories and evaluates the particle posterior (which the
adaptive design needs anyway).  Pass 2 spends a reference posterior only on the
retained subset.  That split is what makes 2,000 events affordable.
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
from _runner import N_WORKERS, save, save_csv, write_manifest  # noqa: E402
from phase3a_pilot import generate_history  # noqa: E402

from src.decision_loss import threshold_loss  # noqa: E402
from src.diagnostics import information_summary, path_summary  # noqa: E402
from src.posterior_grid import build_reference_posterior  # noqa: E402
from src.priors import rotem_prior, rotem_stimulus_grid  # noqa: E402
from src.response_models import ProbitCurve  # noqa: E402

SEED_BASE = 20260902
N_STEPS = 30
TARGETS = ("q0.5", "q0.95", "q0.99")
LEVEL = 0.95
REF_FIXED_N = 513
SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

# stratum -> (mu, sigma, n_histories, mechanism label)
STRATA = (
    [(m, 0.3, int(340 * SCALE), "steep_in_support") for m in
     (18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0)]
    + [(m, s, int(160 * SCALE), "truth_at_or_outside_grid")
       for m in (14.0, 46.0) for s in (0.3, 0.6, 1.2)]
)


def _particle_metrics(post, truths):
    out = {"particle_ess": post.ess(), "particle_max_weight": float(post.w.max()),
           "particle_n_resamples": post.diag.n_resamples,
           "particle_n_invalid": post.diag.n_invalid_particles}
    for t in TARGETS:
        lo, hi = post.credible_interval(t, LEVEL)
        out[f"{t}_p_rank"] = post.cdf_at(t, truths[t])
        out[f"{t}_p_covered"] = post.covered(t, truths[t], LEVEL)
        out[f"{t}_p_mean"] = post.mean(t)
        out[f"{t}_p_median"] = post.median_rotem(t)
        out[f"{t}_p_sd"] = post.sd(t)
        out[f"{t}_p_width"] = hi - lo
    return out


def _gen_worker(args):
    mu, sigma, start, stop, mech = args
    rows = []
    for r in range(start, stop):
        x, y, post, _, cand = generate_history(mu, sigma, r, SEED_BASE,
                                               n_steps=N_STEPS)
        curve = ProbitCurve(mu, sigma)
        truths = {t: float(curve.quantile(float(t[1:]))) for t in TARGETS}
        ps = path_summary(x, y, cand)
        row = {"mu_true": mu, "sigma_true": sigma, "replicate": r,
               "mechanism": mech, "overlap": bool(ps["overlap"]),
               "all_zero": ps["all_zero"], "all_one": ps["all_one"],
               "n_positive": ps["n_positive"], "x_range": ps["x_range"],
               "n_distinct_x": ps["n_distinct_x"]}
        row.update(information_summary(x, y, post.median_rotem("mu"),
                                       max(post.median_rotem("sigma"), 1e-6)))
        row.update(_particle_metrics(post, truths))
        row.update({f"{t}_true": v for t, v in truths.items()})
        row["x"] = x.tolist()
        row["y"] = y.tolist()
        rows.append(row)
    return rows


def _ref_worker(chunk):
    """Reference posterior on retained histories.

    Coverage, the rank statistic, the mean and the sd each cost one weighted
    sum over the grid.  Interval endpoints and the posterior median each cost a
    bisection over many such sums, and together they dominate the pass, so they
    are computed only on a deterministic 20% subsample (`want_width`), which is
    ample for describing the width distribution.  Coverage -- the quantity H1
    is about -- is computed on every retained history.
    """
    prior = rotem_prior("well", "independent")
    out = []
    for rec in chunk:
        x = np.asarray(rec["x"], dtype=float)
        y = np.asarray(rec["y"], dtype=int)
        ref = build_reference_posterior(prior, x, y, fixed_n=REF_FIXED_N)
        want_width = (rec["key"] % 5) == 0
        row = {"key": rec["key"], "ref_boundary_mass": ref.convergence["boundary_mass"],
               "ref_has_width": want_width}
        for t in TARGETS:
            truth = rec[f"{t}_true"]
            row[f"{t}_r_rank"] = ref.cdf_at(t, truth)
            row[f"{t}_r_covered"] = ref.covered(t, truth, LEVEL)
            row[f"{t}_r_mean"] = ref.mean(t)
            row[f"{t}_r_sd"] = ref.sd(t)
            if want_width:
                lo, hi = ref.credible_interval(t, LEVEL)
                row[f"{t}_r_median"] = float(ref.quantile(t, 0.5))
                row[f"{t}_r_width"] = hi - lo
            else:
                row[f"{t}_r_median"] = np.nan
                row[f"{t}_r_width"] = np.nan
        out.append(row)
    return out


def main():
    t0 = time.time()
    tasks = [(m, s, 0, n, mech) for m, s, n, mech in STRATA]
    total_planned = sum(t[3] for t in tasks)
    print(f"Phase 3A: generating {total_planned} histories over {len(tasks)} strata, "
          f"n={N_STEPS}", flush=True)

    rows, done = [], 0
    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        for res in pool.imap_unordered(_gen_worker, tasks):
            rows.extend(res)
            done += 1
            el = time.time() - t0
            print(f"  [gen] {done}/{len(tasks)} strata {el:6.1f}s "
                  f"(eta {el/done*(len(tasks)-done):6.1f}s)", flush=True)
    gen = pd.DataFrame(rows)
    gen["no_overlap"] = ~gen.overlap
    gen["key"] = np.arange(len(gen))

    rates = (gen.groupby(["mechanism", "mu_true", "sigma_true"], as_index=False)
                .agg(n_generated=("no_overlap", "size"),
                     n_no_overlap=("no_overlap", "sum")))
    rates["rate"] = rates.n_no_overlap / rates.n_generated
    rates["rate_se"] = np.sqrt(rates.rate * (1 - rates.rate) / rates.n_generated)
    save_csv(rates, "phase3a_event_rates")
    print(f"\ngenerated {len(gen)}, no-overlap {int(gen.no_overlap.sum())} "
          f"({gen.no_overlap.mean():.3f})", flush=True)

    # size-matched controls drawn within stratum
    rng = np.random.default_rng(SEED_BASE)
    keep = []
    for (m, s), g in gen.groupby(["mu_true", "sigma_true"]):
        cases = g[g.no_overlap]
        ctrl_pool = g[~g.no_overlap]
        n_ctrl = min(len(cases), len(ctrl_pool))
        keep.append(cases.assign(arm="no_overlap"))
        if n_ctrl:
            pick = rng.choice(ctrl_pool.index.to_numpy(), n_ctrl, replace=False)
            keep.append(ctrl_pool.loc[pick].assign(arm="overlap_control"))
    retained = pd.concat(keep, ignore_index=True)
    print(f"retained {len(retained)} histories "
          f"({int((retained.arm == 'no_overlap').sum())} cases, "
          f"{int((retained.arm == 'overlap_control').sum())} controls)", flush=True)

    recs = retained.to_dict("records")
    chunks = [recs[i:i + 40] for i in range(0, len(recs), 40)]
    ref_rows, done = [], 0
    t1 = time.time()
    with ctx.Pool(N_WORKERS) as pool:
        for res in pool.imap_unordered(_ref_worker, chunks):
            ref_rows.extend(res)
            done += 1
            if done % 5 == 0 or done == len(chunks):
                el = time.time() - t1
                print(f"  [ref] {done}/{len(chunks)} chunks {el:6.1f}s "
                      f"(eta {el/done*(len(chunks)-done):6.1f}s)", flush=True)

    merged = retained.merge(pd.DataFrame(ref_rows), on="key", how="left")
    for t in TARGETS:
        merged[f"{t}_shift_in_ref_sd"] = (
            (merged[f"{t}_p_mean"] - merged[f"{t}_r_mean"]) / merged[f"{t}_r_sd"])
        merged[f"{t}_sd_ratio"] = merged[f"{t}_p_sd"] / merged[f"{t}_r_sd"]
        merged[f"{t}_width_ratio"] = merged[f"{t}_p_width"] / merged[f"{t}_r_width"]
        merged[f"{t}_rank_diff"] = merged[f"{t}_p_rank"] - merged[f"{t}_r_rank"]
        merged[f"{t}_cov_diff"] = (merged[f"{t}_p_covered"].astype(int)
                                   - merged[f"{t}_r_covered"].astype(int))
        p = float(t[1:])
        merged[f"{t}_p_loss"] = threshold_loss(merged[f"{t}_p_median"],
                                               merged[f"{t}_true"], 10.0, 1.0)
        merged[f"{t}_r_loss"] = threshold_loss(merged[f"{t}_r_median"],
                                               merged[f"{t}_true"], 10.0, 1.0)
    raw = save(merged.drop(columns=["x", "y"]), "phase3a_retained")
    save(gen[["mu_true", "sigma_true", "replicate", "mechanism", "no_overlap",
              "x", "y"]], "phase3a_histories")

    summ = []
    for (arm, mech), g in merged.groupby(["arm", "mechanism"]):
        for t in TARGETS:
            cd = g[f"{t}_cov_diff"].to_numpy()
            n = len(g)
            summ.append({
                "arm": arm, "mechanism": mech, "target": t, "n": n,
                "cov_particle": float(g[f"{t}_p_covered"].mean()),
                "cov_reference": float(g[f"{t}_r_covered"].mean()),
                "cov_diff": float(cd.mean()),
                "cov_diff_se": float(cd.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
                "n_discordant": int((cd != 0).sum()),
                "mean_shift_in_ref_sd": float(g[f"{t}_shift_in_ref_sd"].abs().mean()),
                "p95_shift_in_ref_sd": float(g[f"{t}_shift_in_ref_sd"].abs().quantile(0.95)),
                "sd_ratio_mean": float(g[f"{t}_sd_ratio"].mean()),
                "sd_ratio_p05": float(g[f"{t}_sd_ratio"].quantile(0.05)),
                "sd_ratio_p95": float(g[f"{t}_sd_ratio"].quantile(0.95)),
                "rank_diff_mean": float(g[f"{t}_rank_diff"].mean()),
                "rank_diff_p95_abs": float(g[f"{t}_rank_diff"].abs().quantile(0.95)),
                "width_ratio_mean": float(g[f"{t}_width_ratio"].mean()),
                "loss_particle": float(g[f"{t}_p_loss"].mean()),
                "loss_reference": float(g[f"{t}_r_loss"].mean()),
                "ess_mean": float(g["particle_ess"].mean()),
                "max_weight_mean": float(g["particle_max_weight"].mean()),
                "n_resamples_mean": float(g["particle_n_resamples"].mean()),
                "n_invalid_mean": float(g["particle_n_invalid"].mean()),
            })
    sdf = pd.DataFrame(summ)
    spath = save_csv(sdf, "phase3a_paired_summary")

    strat = []
    for (arm, m, s), g in merged.groupby(["arm", "mu_true", "sigma_true"]):
        for t in TARGETS:
            cd = g[f"{t}_cov_diff"].to_numpy()
            strat.append({"arm": arm, "mu_true": m, "sigma_true": s, "target": t,
                          "n": len(g), "cov_particle": float(g[f"{t}_p_covered"].mean()),
                          "cov_reference": float(g[f"{t}_r_covered"].mean()),
                          "cov_diff": float(cd.mean()),
                          "cov_diff_se": float(cd.std(ddof=1) / np.sqrt(len(g)))
                          if len(g) > 1 else np.nan})
    stpath = save_csv(pd.DataFrame(strat), "phase3a_stratified")

    print("\n" + sdf[["arm", "mechanism", "target", "n", "cov_particle",
                      "cov_reference", "cov_diff", "cov_diff_se",
                      "mean_shift_in_ref_sd", "sd_ratio_mean"]]
          .to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    n_case = int((merged.arm == "no_overlap").sum())
    write_manifest(
        experiment="phase3a_nooverlap",
        config={"strata": [{"mu": m, "sigma": s, "n": n, "mechanism": mech}
                           for m, s, n, mech in STRATA],
                "n_steps": N_STEPS, "policy": "entropy_vector",
                "n_particles": 10_000, "targets": list(TARGETS),
                "level": LEVEL, "reference_fixed_n": REF_FIXED_N,
                "scale": SCALE},
        seed_base=SEED_BASE,
        requested=total_planned, completed=int(len(gen)), failures=[],
        tolerances={"reference_mode": f"validated fixed n={REF_FIXED_N}"},
        mc_se={f"cov_diff::{r.arm}::{r.mechanism}::{r.target}": r.cov_diff_se
               for r in sdf.itertuples()},
        degeneracies={"n_generated": int(len(gen)),
                      "n_no_overlap": int(gen.no_overlap.sum()),
                      "overall_event_rate": float(gen.no_overlap.mean()),
                      "n_retained": int(len(merged)),
                      "n_cases": n_case,
                      "n_all_zero": int(gen.all_zero.sum()),
                      "n_all_one": int(gen.all_one.sum())},
        artifacts=[raw, spath, stpath], wall=time.time() - t0,
        notes="H1: paired particle-vs-reference comparison on identical "
              "no-overlap histories, stratified enrichment.",
    )


if __name__ == "__main__":
    main()
