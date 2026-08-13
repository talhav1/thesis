"""Phase 1 task 3: reproduce Rotem's baseline MSE tables.

Targets: her Table 1 (Bayes estimators, probit truth, four prior settings) and
Table 3 (Bayes estimators, Beta-CDF truth, independent priors), plus the probit
MLE columns of Table 2 and the Bruceton column of Tables 6-7.

Two things this run does that the thesis does not:

* every MSE carries a Monte Carlo standard error, so "reproduced" can be given
  a meaning instead of being eyeballed;
* every degenerate path is counted and reported rather than dropped.

Horizon: n = 30 with a slice at n = 20.  All of Rotem's reported tables are at
n = 30; `phase1_horizon.py` runs the n = 50 arm separately on one setting.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _runner import (  # noqa: E402
    degeneracy_counts,
    mse_with_se,
    run_jobs,
    save,
    save_csv,
    write_manifest,
)
from _cli import parse as _parse_cli  # noqa: E402

from src.priors import rotem_prior  # noqa: E402
from src.response_models import BetaCDFCurve  # noqa: E402
from src.simulator import ExperimentConfig  # noqa: E402

RUN = _parse_cli("phase1_baseline", replicates=400, seed_base=20260813, crn_seed=777001)

N_REPLICATES = RUN.replicates
SEED_BASE = RUN.seed_base
CRN_SEED = RUN.crn_seed
N_STEPS = 30
SLICES = (20, 30)

PROBIT = {"family": "probit", "mu": 30.0, "sigma": 3.0}
BETA = {"family": "beta_cdf", "a": 5.0, "b": 8.0, "lo": 18.0, "hi": 48.0}

BAYESIAN = ["dror_steinberg", "entropy_vector", "entropy_mu", "entropy_sigma",
            "entropy_q0.95"]

SETTINGS = [
    # (label, curve, dependence, calibration, extra policies)
    ("probit_dep_well", PROBIT, "dependent", "well", []),
    ("probit_dep_poor", PROBIT, "dependent", "poor", []),
    ("probit_ind_well", PROBIT, "independent", "well", ["bruceton"]),
    ("probit_ind_poor", PROBIT, "independent", "poor", ["bruceton"]),
    ("beta_ind_well", BETA, "independent", "well", ["bruceton"]),
    ("beta_ind_poor", BETA, "independent", "poor", ["bruceton"]),
]

TARGETS_PROBIT = ("mu", "sigma", "q0.5", "q0.95", "q0.99")
TARGETS_BETA = ("mu", "sigma", "q0.01", "q0.05", "q0.5", "q0.95", "q0.99")


def build_jobs():
    jobs = {}
    for label, curve, dependence, calibration, extra in SETTINGS:
        prior = rotem_prior(calibration, dependence)
        targets = TARGETS_BETA if curve["family"] == "beta_cdf" else TARGETS_PROBIT
        for policy in BAYESIAN + extra:
            jobs[f"{label}|{policy}"] = ExperimentConfig(
                policy=policy,
                prior=prior.spec.config(),
                curve=curve,
                calibration=calibration,
                n_steps=N_STEPS,
                slices=SLICES,
                targets=targets,
                n_particles=10_000,
                grid_points=200,
                grid_half_width=14.0,
                bruceton_step=2.0,
                policy_kwargs={"start_sd": 3.0} if policy == "bruceton" else {},
            )
    return jobs


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """MSE (Bayes and MLE) with Monte Carlo SEs, per setting/policy/n/target."""
    out = []
    for (setting, policy, n), g in df.groupby(["setting", "policy", "n"], sort=False):
        for col in g.columns:
            if not col.endswith("_err"):
                continue
            target = col[: -len("_err")]
            kind = "mle" if target.endswith("_mle") else "bayes"
            target = target[: -len("_mle")] if kind == "mle" else target
            mse, se = mse_with_se(g[col])
            bias = float(np.nanmean(g[col]))
            out.append({
                "setting": setting.split("|")[0],
                "policy": policy,
                "n": n,
                "target": target,
                "estimator": kind,
                "mse": mse,
                "mse_se": se,
                "bias": bias,
                "rmse": np.sqrt(mse) if np.isfinite(mse) else np.nan,
                "n_used": int(np.isfinite(g[col]).sum()),
                "n_total": len(g),
                # coverage and width describe the *posterior*, so they belong
                # only to the Bayes rows; attaching them to the MLE rows would
                # silently label Bayes coverage as MLE coverage
                "coverage": float(g[f"{target}_covered"].mean())
                if kind == "bayes" and f"{target}_covered" in g else np.nan,
                "coverage_se": float(
                    np.sqrt(g[f"{target}_covered"].mean()
                            * (1 - g[f"{target}_covered"].mean()) / len(g))
                ) if kind == "bayes" and f"{target}_covered" in g else np.nan,
                "mean_width": float(g[f"{target}_width"].mean())
                if kind == "bayes" and f"{target}_width" in g else np.nan,
            })
    return pd.DataFrame(out)


def main():
    jobs = build_jobs()
    print(f"Phase 1 baseline: {len(jobs)} setting x policy cells, "
          f"{N_REPLICATES} replicates each, n={N_STEPS}", flush=True)

    beta = BetaCDFCurve(**{k: v for k, v in BETA.items() if k != "family"})
    print("  Beta-CDF truth: median=%.4f sd=%.4f q95=%.4f q99=%.4f q05=%.4f q01=%.4f"
          % (beta.implied_median, beta.implied_sd, beta.quantile(0.95),
             beta.quantile(0.99), beta.quantile(0.05), beta.quantile(0.01)),
          flush=True)

    rows, failures, traj, wall = run_jobs(
        jobs, N_REPLICATES, SEED_BASE, CRN_SEED, "phase1", progress_every=10
    )
    df = pd.DataFrame(rows)
    df["policy"] = df["policy"].astype(str)
    raw_path = save(df, "phase1_baseline_raw")

    summary = summarise(df)
    sum_path = save_csv(summary, "phase1_baseline_mse")

    # Rotem-format tables: MSE of the Bayes estimator at n = 30
    tables = []
    for setting in sorted(summary["setting"].unique()):
        sel = summary[(summary.setting == setting) & (summary.n == 30)
                      & (summary.estimator == "bayes")]
        piv = sel.pivot_table(index="target", columns="policy", values="mse")
        piv_se = sel.pivot_table(index="target", columns="policy", values="mse_se")
        piv.insert(0, "setting", setting)
        tables.append(piv.reset_index())
        piv_se.insert(0, "setting", setting)
        tables.append(piv_se.reset_index().assign(target=lambda d: d.target + "_SE"))
    table_path = save_csv(pd.concat(tables, ignore_index=True),
                          "phase1_table1_and_3_reproduction")

    traj_rows = []
    for key, items in traj.items():
        for it in items[:25]:            # a sample of paths for the figures
            if it["x"] is None:
                continue
            traj_rows.append({"setting": key, "replicate": it["replicate"],
                              "x": list(map(float, it["x"])),
                              "y": list(map(int, it["y"]))})
    traj_path = save(pd.DataFrame(traj_rows), "phase1_trajectories")

    mc_se = {}
    for _, r in summary[(summary.n == 30) & (summary.estimator == "bayes")].iterrows():
        mc_se[f"mse::{r.setting}::{r.policy}::{r.target}"] = r.mse_se
    for _, r in summary[(summary.n == 30) & (summary.estimator == "bayes")].iterrows():
        if np.isfinite(r.coverage_se):
            mc_se[f"coverage::{r.setting}::{r.policy}::{r.target}"] = r.coverage_se

    write_manifest(
        experiment="phase1_baseline",
        config={
            "settings": [s[0] for s in SETTINGS],
            "policies": BAYESIAN + ["bruceton"],
            "n_steps": N_STEPS,
            "slices": list(SLICES),
            "n_particles": 10_000,
            "curve_probit": PROBIT,
            "curve_beta": BETA,
            "beta_truth": beta.describe(),
            "crn_seed": CRN_SEED,
            "example_config": next(iter(jobs.values())).as_dict(),
        },
        seed_base=SEED_BASE,
        requested=N_REPLICATES * len(jobs),
        completed=int(df.replicate.nunique() * len(jobs)),
        failures=failures,
        tolerances={"reference_tol": "n/a (particle posterior only)",
                    "prob_floor": 1e-12},
        mc_se=mc_se,
        degeneracies=degeneracy_counts(df),
        artifacts=[raw_path, sum_path, table_path, traj_path],
        wall=wall,
        notes="Reproduction of Rotem Tables 1 and 3 (Bayes) and 2 (MLE); "
              "Bruceton column of Tables 6-7. MC SEs attached to every cell.",
    )
    print(f"done in {wall:.0f}s; rows={len(df)}, failures={len(failures)}")


if __name__ == "__main__":
    main()
