"""Phase 1 horizon check: does MSE fall monotonically in n, as Rotem states?

She reports tables at n = 30 and says n = 20 and n = 50 "largely match" with
MSE decreasing in n.  The main baseline run stops at n = 30 to stay inside the
compute budget; this arm runs one setting out to n = 50 so the claim is checked
rather than assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _runner import degeneracy_counts, mse_with_se, run_jobs, save, save_csv, write_manifest  # noqa: E402

from src.priors import rotem_prior  # noqa: E402
from src.simulator import ExperimentConfig  # noqa: E402

N_REPLICATES = int(sys.argv[1]) if len(sys.argv) > 1 else 120
SEED_BASE = 20260816
CRN_SEED = 990001
POLICIES = ["entropy_vector", "entropy_q0.95", "dror_steinberg"]
TARGETS = ("mu", "sigma", "q0.5", "q0.95", "q0.99")


def main():
    prior = rotem_prior("well", "independent")
    jobs = {
        f"probit_ind_well|{p}": ExperimentConfig(
            policy=p, prior=prior.spec.config(),
            curve={"family": "probit", "mu": 30.0, "sigma": 3.0},
            calibration="well", n_steps=50, slices=(20, 30, 50),
            targets=TARGETS, n_particles=10_000, grid_points=200,
        )
        for p in POLICIES
    }
    print(f"Phase 1 horizon: {len(jobs)} cells x {N_REPLICATES} replicates, "
          f"n in (20, 30, 50)", flush=True)
    rows, failures, _, wall = run_jobs(jobs, N_REPLICATES, SEED_BASE, CRN_SEED,
                                       "horizon", progress_every=5)
    df = pd.DataFrame(rows)
    raw = save(df, "phase1_horizon_raw")

    recs = []
    for (policy, n), g in df.groupby(["policy", "n"]):
        for t in TARGETS:
            mse, se = mse_with_se(g[f"{t}_err"])
            recs.append({"policy": policy, "n": n, "target": t, "mse": mse,
                         "mse_se": se,
                         "coverage": float(g[f"{t}_covered"].mean()),
                         "mean_width": float(g[f"{t}_width"].mean())})
    summ = pd.DataFrame(recs).sort_values(["policy", "target", "n"])
    path = save_csv(summ, "phase1_horizon_mse")

    checks = []
    for (policy, t), g in summ.groupby(["policy", "target"]):
        g = g.sort_values("n")
        m = g.mse.to_numpy()
        checks.append({"policy": policy, "target": t,
                       "mse_n20": m[0], "mse_n30": m[1], "mse_n50": m[2],
                       "monotone_decreasing": bool(np.all(np.diff(m) < 0))})
    chk = pd.DataFrame(checks)
    chk_path = save_csv(chk, "phase1_horizon_monotonicity")
    print(chk.to_string(index=False))

    write_manifest(
        experiment="phase1_horizon",
        config={"policies": POLICIES, "n_steps": 50, "slices": [20, 30, 50],
                "setting": "probit_ind_well", "crn_seed": CRN_SEED,
                "example_config": next(iter(jobs.values())).as_dict()},
        seed_base=SEED_BASE,
        requested=N_REPLICATES * len(jobs),
        completed=int(len(df)),
        failures=failures,
        tolerances={"prob_floor": 1e-12},
        mc_se={f"mse::{r.policy}::n{r.n}::{r.target}": r.mse_se
               for r in summ.itertuples()},
        degeneracies=degeneracy_counts(df),
        artifacts=[raw, path, chk_path],
        wall=wall,
        notes="Horizon arm: checks the monotone-in-n claim on one setting.",
    )


if __name__ == "__main__":
    main()
