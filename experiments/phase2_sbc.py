"""Phase 2 tasks 1-3: simulation-based calibration with the policy in the loop.

Draw theta ~ pi_0, simulate the *entire* adaptive experiment, and test the
posterior rank of the true value for uniformity.

Two inference backends are run on the identical simulated histories:

* the deterministic refined-grid reference -- must pass;
* Rotem's weighted-particle posterior -- any failure here is computational.

Splitting them is the point.  The gate in the plan is explicit: if the
reference fails SBC, stop and fix the implementation; if only the particle
method fails, computational approximation becomes its own thesis chapter.

Policies: `uniform_grid` (non-adaptive control) and `entropy_vector` (Rotem's
adaptive design).  If adaptive dependence invalidated the product likelihood,
the adaptive arm would fail while the control passed.
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
from _runner import N_WORKERS, degeneracy_counts, save, save_csv, write_manifest  # noqa: E402
from _cli import parse as _parse_cli  # noqa: E402

from src.calibration import (  # noqa: E402
    coverage_report,
    quantile_bin,
    stratified_coverage,
    uniformity_report,
)
from src.calibration import sbc_replicate  # noqa: E402
from src.priors import rotem_prior  # noqa: E402
from src.simulator import ExperimentConfig  # noqa: E402

RUN = _parse_cli("phase2_sbc", replicates=800, seed_base=20260814)

N_DRAWS = RUN.replicates
SEED_BASE = RUN.seed_base
N_STEPS = 30
TARGETS = ("mu", "sigma", "q0.5", "q0.95", "q0.99")
POLICIES = ("uniform_grid", "entropy_vector")


def config(policy):
    prior = rotem_prior("well", "independent")
    return ExperimentConfig(
        policy=policy,
        prior=prior.spec.config(),
        curve={"family": "probit", "mu": 30.0, "sigma": 3.0},  # replaced per draw
        calibration="well",
        n_steps=N_STEPS,
        slices=(N_STEPS,),
        n_particles=10_000,
        targets=TARGETS,
        grid_points=200,
        reference_tol=1e-3,
        reference_n_max=1025,
    )


def _worker(args):
    policy, start, stop = args
    cfg = config(policy)
    out = []
    for r in range(start, stop):
        try:
            row = sbc_replicate(cfg, r, SEED_BASE, use_reference=True)
            row["policy"] = policy
            out.append(row)
        except Exception as exc:  # noqa: BLE001
            out.append({"policy": policy, "replicate": r,
                        "error": f"{type(exc).__name__}: {exc}"})
    return out


def main():
    chunk = 20
    tasks = [(p, s, min(s + chunk, N_DRAWS))
             for p in POLICIES for s in range(0, N_DRAWS, chunk)]
    print(f"Phase 2 SBC: {N_DRAWS} draws x {len(POLICIES)} policies, n={N_STEPS}",
          flush=True)

    rows, t0, done = [], time.time(), 0
    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        for res in pool.imap_unordered(_worker, tasks):
            rows.extend(res)
            done += 1
            el = time.time() - t0
            print(f"  [sbc] {done}/{len(tasks)} chunks {el:6.1f}s "
                  f"(eta {el/done*(len(tasks)-done):6.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    errors = df[df.get("error").notna()] if "error" in df else pd.DataFrame()
    failures = errors.to_dict("records") if len(errors) else []
    df = df[df.get("error").isna()] if "error" in df else df
    raw = save(df, "phase2_sbc_raw")

    reports = []
    for policy in POLICIES:
        g = df[df.policy == policy]
        for backend in ("ref", "particle"):
            for t in TARGETS:
                col = f"{t}_rank_{backend}"
                if col not in g:
                    continue
                rep = uniformity_report(g[col].to_numpy())
                cov = coverage_report(g[f"{t}_covered_{backend}"].to_numpy())
                reports.append({
                    "policy": policy, "backend": backend, "target": t,
                    "n": rep["n"], "ks_stat": rep["ks_stat"], "ks_p": rep["ks_p"],
                    "ks_band_95": rep["ks_band_95"],
                    "chi2_p": rep["chi2_p"], "rank_mean": rep["mean"],
                    "rank_mean_se": rep["mean_se"],
                    "coverage": cov["coverage"], "coverage_se": cov["se"],
                    "passes_ks_05": rep["ks_p"] > 0.05,
                })
    rep_df = pd.DataFrame(reports)
    rep_path = save_csv(rep_df, "phase2_sbc_uniformity")

    # Task 5: stratify coverage by observable path status
    df = df.copy()
    df["overlap_status"] = np.where(df["overlap"], "overlap", "no_overlap")
    df["info_stratum"] = quantile_bin(df["info_min_eig"].to_numpy(), 4)
    df["ess_stratum"] = quantile_bin(df["particle_ess"].to_numpy(), 4)
    strata = []
    for policy in POLICIES:
        g = df[df.policy == policy].to_dict("records")
        for backend in ("ref", "particle"):
            for t in TARGETS:
                for by in ("overlap_status", "info_stratum", "ess_stratum"):
                    for rec in stratified_coverage(
                        g, t, by, covered_key=f"{t}_covered_{backend}"
                    ):
                        rec.update({"policy": policy, "backend": backend})
                        strata.append(rec)
    strat_path = save_csv(pd.DataFrame(strata), "phase2_coverage_stratified")

    print("\n" + rep_df.to_string(index=False))

    write_manifest(
        experiment="phase2_sbc",
        config={"policies": list(POLICIES), "n_draws": N_DRAWS,
                "n_steps": N_STEPS, "targets": list(TARGETS),
                "n_particles": 10_000,
                "example_config": config(POLICIES[0]).as_dict()},
        seed_base=SEED_BASE,
        requested=N_DRAWS * len(POLICIES),
        completed=int(len(df)),
        failures=failures,
        tolerances={"reference_tol": 1e-3, "reference_n_max": 1025,
                    "ks_alpha": 0.05},
        mc_se={f"rank_mean::{r.policy}::{r.backend}::{r.target}": r.rank_mean_se
               for r in rep_df.itertuples()}
        | {f"coverage::{r.policy}::{r.backend}::{r.target}": r.coverage_se
           for r in rep_df.itertuples()},
        degeneracies=degeneracy_counts(df) | {
            "ref_not_converged": int((~df["ref_converged"].astype(bool)).sum())
            if "ref_converged" in df else -1,
            "ref_max_delta_worst": float(df["ref_max_delta"].max())
            if "ref_max_delta" in df else float("nan"),
        },
        artifacts=[raw, rep_path, strat_path],
        wall=time.time() - t0,
        notes="SBC with the adaptive policy inside the simulator; reference and "
              "particle backends on identical histories.",
    )


if __name__ == "__main__":
    main()
