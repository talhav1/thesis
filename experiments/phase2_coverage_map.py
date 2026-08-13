"""Phase 2 task 4: fixed-parameter coverage maps over (mu, sigma).

For each true (mu_0, sigma_0) on a 3 x 3 grid, repeat the whole adaptive
experiment and record how often the 95% credible interval for each target
quantile contains the truth.

Reading these correctly matters.  Fixed-theta coverage is *not* required to
equal the nominal level: the posterior is Bayes-calibrated over draws from the
prior (that is what the SBC arm tests), so at any single theta_0 -- especially
one in the prior's tail, with n = 30 and an informative prior -- systematic
departures are expected and are a property of the procedure, not a defect.
What the map is for is showing *where* in parameter space the departures live
and how they differ between the exact reference and Rotem's particle posterior.
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
from _runner import (  # noqa: E402
    N_WORKERS,
    _single,
    degeneracy_counts,
    save,
    save_csv,
    write_manifest,
)

from src.calibration import coverage_report, quantile_bin, stratified_coverage  # noqa: E402
from src.priors import rotem_prior  # noqa: E402
from src.simulator import ExperimentConfig  # noqa: E402

N_REPLICATES = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SEED_BASE = 20260815
CRN_SEED = 880001
N_STEPS = 30
TARGETS = ("mu", "sigma", "q0.5", "q0.95", "q0.99")
MU_GRID = (26.0, 30.0, 34.0)
SIGMA_GRID = (1.5, 3.0, 6.0)
POLICY = "entropy_vector"


def config(mu0, sigma0):
    prior = rotem_prior("well", "independent")
    return ExperimentConfig(
        policy=POLICY,
        prior=prior.spec.config(),
        curve={"family": "probit", "mu": mu0, "sigma": sigma0},
        calibration="well",
        n_steps=N_STEPS,
        slices=(N_STEPS,),
        n_particles=10_000,
        targets=TARGETS,
        grid_points=200,
        reference_at_slices=(N_STEPS,),
        reference_tol=1e-3,
        reference_n_max=1025,
    )


def _worker(args):
    key, mu0, sigma0, start, stop = args
    cfg = config(mu0, sigma0)
    rows, failures = [], []
    for r in range(start, stop):
        rr, ff, _ = _single(cfg, r, SEED_BASE, CRN_SEED)
        for row in rr:
            row.update({"setting": key, "mu_true0": mu0, "sigma_true0": sigma0})
        rows.extend(rr)
        failures.extend(ff)
    return rows, failures


def main():
    chunk = 25
    tasks = []
    for mu0 in MU_GRID:
        for sigma0 in SIGMA_GRID:
            key = f"mu{mu0}_sigma{sigma0}"
            for s in range(0, N_REPLICATES, chunk):
                tasks.append((key, mu0, sigma0, s, min(s + chunk, N_REPLICATES)))
    print(f"Phase 2 coverage map: {len(MU_GRID) * len(SIGMA_GRID)} cells x "
          f"{N_REPLICATES} replicates, n={N_STEPS}", flush=True)

    rows, failures, t0, done = [], [], time.time(), 0
    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        for rr, ff in pool.imap_unordered(_worker, tasks):
            rows.extend(rr)
            failures.extend(ff)
            done += 1
            el = time.time() - t0
            print(f"  [cov] {done}/{len(tasks)} chunks {el:6.1f}s "
                  f"(eta {el/done*(len(tasks)-done):6.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    raw = save(df, "phase2_coverage_map_raw")

    recs = []
    for (mu0, sigma0), g in df.groupby(["mu_true0", "sigma_true0"]):
        for t in TARGETS:
            for backend, col in (("particle", f"{t}_covered"),
                                 ("ref", f"{t}_ref_covered")):
                if col not in g:
                    continue
                cov = coverage_report(g[col].to_numpy())
                width_col = f"{t}_width" if backend == "particle" else f"{t}_ref_width"
                err_col = f"{t}_err"
                recs.append({
                    "mu_true": mu0, "sigma_true": sigma0, "target": t,
                    "backend": backend, "n": cov["n"],
                    "coverage": cov["coverage"], "coverage_se": cov["se"],
                    "z_vs_nominal": cov["z"],
                    "mean_width": float(g[width_col].mean()) if width_col in g else np.nan,
                    "median_width": float(g[width_col].median()) if width_col in g else np.nan,
                    "bias": float(g[err_col].mean()) if err_col in g else np.nan,
                    "rmse": float(np.sqrt((g[err_col] ** 2).mean()))
                    if err_col in g else np.nan,
                    "frac_no_overlap": float((~g["overlap"].astype(bool)).mean()),
                    "frac_all_equal": float(g["degenerate_all_equal"].mean()),
                    "mean_ess": float(g["particle_ess"].mean()),
                })
    cov_df = pd.DataFrame(recs)
    cov_path = save_csv(cov_df, "phase2_coverage_map")

    # false certainty: interval misses the truth AND is unusually narrow
    fc = []
    for t in TARGETS:
        col, wcol = f"{t}_covered", f"{t}_width"
        if col not in df or wcol not in df:
            continue
        thresh = float(df[wcol].quantile(0.25))
        for (mu0, sigma0), g in df.groupby(["mu_true0", "sigma_true0"]):
            narrow = g[wcol] <= thresh
            miss = ~g[col].astype(bool)
            k = int((narrow & miss).sum())
            fc.append({
                "mu_true": mu0, "sigma_true": sigma0, "target": t,
                "narrowness_threshold": thresh,
                "false_certainty_rate": k / len(g),
                "false_certainty_se": float(np.sqrt(max(k / len(g) * (1 - k / len(g)), 0) / len(g))),
                "n": len(g),
            })
    fc_path = save_csv(pd.DataFrame(fc), "phase2_false_certainty")

    df = df.copy()
    df["overlap_status"] = np.where(df["overlap"], "overlap", "no_overlap")
    df["info_stratum"] = quantile_bin(df["info_min_eig"].to_numpy(), 4)
    strata = []
    recs_all = df.to_dict("records")
    for t in TARGETS:
        for by in ("overlap_status", "info_stratum"):
            for rec in stratified_coverage(recs_all, t, by):
                rec["backend"] = "particle"
                strata.append(rec)
            for rec in stratified_coverage(recs_all, t, by,
                                           covered_key=f"{t}_ref_covered"):
                rec["backend"] = "ref"
                strata.append(rec)
    strat_path = save_csv(pd.DataFrame(strata), "phase2_coverage_map_stratified")

    print("\n" + cov_df[cov_df.target.isin(["q0.5", "q0.95", "q0.99"])]
          .to_string(index=False))

    write_manifest(
        experiment="phase2_coverage_map",
        config={"mu_grid": list(MU_GRID), "sigma_grid": list(SIGMA_GRID),
                "policy": POLICY, "n_steps": N_STEPS, "targets": list(TARGETS),
                "n_particles": 10_000, "crn_seed": CRN_SEED,
                "example_config": config(MU_GRID[0], SIGMA_GRID[0]).as_dict()},
        seed_base=SEED_BASE,
        requested=N_REPLICATES * len(MU_GRID) * len(SIGMA_GRID),
        completed=int(len(df)),
        failures=failures,
        tolerances={"reference_tol": 1e-3, "reference_n_max": 1025},
        mc_se={f"coverage::{r.backend}::mu{r.mu_true}_sigma{r.sigma_true}::{r.target}":
               r.coverage_se for r in cov_df.itertuples()},
        degeneracies=degeneracy_counts(df) | {
            "ref_not_converged": int((~df["ref_converged"].astype(bool)).sum())
            if "ref_converged" in df else -1},
        artifacts=[raw, cov_path, fc_path, strat_path],
        wall=time.time() - t0,
        notes="Fixed-theta coverage over a 3x3 (mu, sigma) grid, particle and "
              "reference backends on identical histories.",
    )


if __name__ == "__main__":
    main()
