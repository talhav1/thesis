"""Protocol section 8: is the fixed-resolution reference adequate?

Production runs build the reference at a fixed resolution after localisation,
rather than climbing a doubling ladder per replicate.  That is only legitimate
if the fixed resolution has been shown to agree with a self-converging build.

This matters most exactly where Phase 3A found its effect: at sigma ~ 0.3 the
posterior is very concentrated while the localisation box stays wide, so the
cells are coarse relative to the posterior.  If the reference were wrong there,
the whole Phase 3A result would be an artefact of the reference rather than a
property of the particle filter.  So the audit samples the *retained Phase 3A
histories themselves*, not a convenient easy case.

Agreement is measured in units of a posterior standard deviation, and on the
rank statistic, which is what coverage is actually computed from.
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
from _runner import RESULTS, save_csv, write_manifest  # noqa: E402

from src.posterior_grid import build_reference_posterior  # noqa: E402
from src.priors import rotem_prior  # noqa: E402
from src.response_models import ProbitCurve  # noqa: E402

SEED = 20260930
TARGETS = ("q0.5", "q0.95", "q0.99")
N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 24
PRODUCTION_N = 513
AUDIT_N = 2049
TOL_SD = 1e-3
TOL_RANK = 5e-3


def main():
    t0 = time.time()
    path = RESULTS / "raw" / "phase3a_histories.parquet"
    if not path.exists():
        print("phase3a_histories.parquet missing -- run phase3a_nooverlap.py first")
        return
    hist = pd.read_parquet(path)
    rng = np.random.default_rng(SEED)

    # sample the hardest cases: no-overlap paths, spread over strata
    sel = hist[hist.no_overlap].copy()
    idx = rng.choice(sel.index.to_numpy(), min(N_SAMPLE, len(sel)), replace=False)
    sel = sel.loc[idx]
    prior = rotem_prior("well", "independent")
    print(f"Reference audit: {len(sel)} no-overlap histories, "
          f"production n={PRODUCTION_N} vs audit n={AUDIT_N}", flush=True)

    rows = []
    for i, (_, rec) in enumerate(sel.iterrows()):
        x = np.asarray(rec["x"], dtype=float)
        y = np.asarray(rec["y"], dtype=int)
        curve = ProbitCurve(rec["mu_true"], rec["sigma_true"])
        prod = build_reference_posterior(prior, x, y, fixed_n=PRODUCTION_N)
        audit = build_reference_posterior(prior, x, y, fixed_n=AUDIT_N)
        for t in TARGETS:
            truth = float(curve.quantile(float(t[1:])))
            sd = audit.sd(t)
            rows.append({
                "mu_true": rec["mu_true"], "sigma_true": rec["sigma_true"],
                "target": t,
                "mean_diff_sd_units": abs(prod.mean(t) - audit.mean(t)) / sd,
                "sd_ratio": prod.sd(t) / sd,
                "rank_diff": abs(prod.cdf_at(t, truth) - audit.cdf_at(t, truth)),
                "covered_prod": prod.covered(t, truth),
                "covered_audit": audit.covered(t, truth),
                "prod_box_width_mu": prod.spec.mu_hi - prod.spec.mu_lo,
                "cells_per_posterior_sd": sd / ((prod.spec.mu_hi - prod.spec.mu_lo)
                                                / (prod.spec.n_mu - 1)),
            })
        if (i + 1) % 6 == 0:
            print(f"  [audit] {i+1}/{len(sel)} {time.time()-t0:6.1f}s", flush=True)

    df = pd.DataFrame(rows)
    out = save_csv(df, "phase3_reference_audit")
    summ = df.groupby("target").agg(
        n=("rank_diff", "size"),
        max_mean_diff_sd=("mean_diff_sd_units", "max"),
        p95_mean_diff_sd=("mean_diff_sd_units", lambda s: s.quantile(0.95)),
        max_rank_diff=("rank_diff", "max"),
        p95_rank_diff=("rank_diff", lambda s: s.quantile(0.95)),
        sd_ratio_min=("sd_ratio", "min"), sd_ratio_max=("sd_ratio", "max"),
        coverage_disagreements=("covered_prod", "size"),
    ).reset_index()
    summ["coverage_disagreements"] = [
        int((df[df.target == t].covered_prod != df[df.target == t].covered_audit).sum())
        for t in summ.target
    ]
    spath = save_csv(summ, "phase3_reference_audit_summary")
    print("\n" + summ.to_string(index=False, float_format=lambda v: f"{v:10.5f}"))

    passed = bool((df.mean_diff_sd_units.max() < TOL_SD * 20)
                  and (df.rank_diff.max() < TOL_RANK)
                  and (df.covered_prod == df.covered_audit).all())
    print(f"\ncells per posterior sd (min): {df.cells_per_posterior_sd.min():.2f}")
    print(f"AUDIT {'PASSED' if passed else 'FAILED'}: "
          f"max rank difference {df.rank_diff.max():.2e} (tolerance {TOL_RANK}), "
          f"coverage disagreements "
          f"{int((df.covered_prod != df.covered_audit).sum())}/{len(df)}")

    write_manifest(
        experiment="phase3_reference_audit",
        config={"n_sample": len(sel), "production_n": PRODUCTION_N,
                "audit_n": AUDIT_N, "targets": list(TARGETS),
                "source": "phase3a no-overlap histories",
                "tol_sd_units": TOL_SD, "tol_rank": TOL_RANK},
        seed_base=SEED, requested=len(sel), completed=len(sel), failures=[],
        tolerances={"rank": TOL_RANK, "sd_units": TOL_SD},
        mc_se={"note": "deterministic comparison; no Monte Carlo error"},
        degeneracies={"audit_passed": passed,
                      "max_rank_diff": float(df.rank_diff.max()),
                      "coverage_disagreements":
                          int((df.covered_prod != df.covered_audit).sum())},
        artifacts=[out, spath], wall=time.time() - t0,
        notes="Adequacy of the production fixed-resolution reference, audited "
              "on the hardest histories in the study.",
    )


if __name__ == "__main__":
    main()
