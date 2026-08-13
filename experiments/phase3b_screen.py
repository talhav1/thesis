"""Phase 3B screening: the misspecification failure atlas, exploratory tier.

Two blocks, analysed and reported separately and never pooled:

* **Block I** -- in-support misspecification.  Calibrated prior, grid [16, 44],
  every reported target quantile strictly inside it.
* **Block II** -- design-support failure.  Poorly-calibrated grid [8, 36] and/or
  a shifted truth, so that the target is near or outside the grid.

Also two small sub-blocks crossing curve scale and prior calibration against a
reduced DGP and policy set, to see whether the Block I picture survives them.

The **correct-probit / entropy_vector** cell of Block I is what fixes the
false-certainty width thresholds of protocol section 4.  It is run here, its
thresholds are frozen to `configs/phase3_thresholds.json`, and only then is any
misspecified cell scored against them.  The thresholds are never recomputed
inside a misspecified cell.

Primary inference is the reference posterior; the particle posterior runs on the
identical histories as a secondary overlay so that model error and numerical
error stay separable.
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

from src.manifest import config_hash  # noqa: E402
from src.priors import PriorSpec, rotem_prior  # noqa: E402
from src.simulator import ExperimentConfig  # noqa: E402

SEED_BASE = 20260910
CRN_SEED = 313001
TARGETS = ("q0.5", "q0.95", "q0.99")
REF_FIXED_N = 513
N_STEPS = 50
SLICES = (20, 30, 50)
REF_AT = (50,)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
REPS_SUB = max(20, int(REPS * 0.75))
THRESHOLD_PATH = RESULTS.parent / "configs" / "phase3_thresholds.json"

# ---- DGPs -----------------------------------------------------------------

BLOCK_I_DGPS = [
    ("probit", {"family": "probit", "mu": 30.0, "sigma": 3.0}),
    ("logistic_0.5", {"family": "logistic", "lam": 0.5}),
    ("logistic_1.0", {"family": "logistic", "lam": 1.0}),
    ("cloglog_0.5", {"family": "cloglog", "lam": 0.5}),
    ("cloglog_1.0", {"family": "cloglog", "lam": 1.0}),
    ("robit_0.5", {"family": "robit", "lam": 0.5}),
    ("robit_1.0", {"family": "robit", "lam": 1.0}),
    ("mixture_0.5", {"family": "mixture", "lam": 0.5}),
    ("mixture_1.0", {"family": "mixture", "lam": 1.0}),
    ("tail_0.5", {"family": "tail_perturbed", "shift95": 0.5}),
    ("tail_1.0", {"family": "tail_perturbed", "shift95": 1.0}),
    ("beta_cdf", {"family": "beta_cdf"}),
]

POLICIES = ["entropy_vector", "entropy_q0.95", "fixed_design", "uniform_grid"]

# Block II: target near or outside the stimulus grid
BLOCK_II_DGPS = [
    ("probit_shift38", {"family": "probit", "mu": 38.0, "sigma": 3.0}),
    ("probit_center", {"family": "probit", "mu": 30.0, "sigma": 3.0}),
    ("tail_1.0", {"family": "tail_perturbed", "shift95": 1.0}),
    ("beta_cdf", {"family": "beta_cdf"}),
]

SUB_DGPS = [("probit", BLOCK_I_DGPS[0][1]), ("cloglog_1.0", BLOCK_I_DGPS[4][1]),
            ("tail_1.0", BLOCK_I_DGPS[10][1])]
SUB_POLICIES = ["entropy_vector", "fixed_design"]


def curve_scale_lookup():
    """True latent SD of every DGP, for the dangerous-error margin delta.

    delta is defined as a fraction of the *true* curve's scale so that the
    event is comparable across DGPs of different spread.  Computed once here
    rather than per replicate.
    """
    from src.curve_families import curve_true_sd
    from src.simulator import build_curve

    out = {}
    for name, cd in BLOCK_I_DGPS + BLOCK_II_DGPS:
        for s0 in (None, 1.5, 6.0):
            cd2 = dict(cd)
            if s0 is not None and cd2["family"] != "beta_cdf":
                cd2["sigma0"] = s0
                if cd2["family"] == "probit":
                    cd2["sigma"] = s0
            key = (name, "s3" if s0 is None else f"s{s0}")
            c = build_curve(cd2)
            out[key] = float(getattr(c, "sigma", None) or getattr(c, "implied_sd", None)
                             or curve_true_sd(c))
    return out


CURVE_SCALE = None


def _policy_kwargs(policy, prior_spec):
    if policy == "fixed_design":
        return {"mu0": prior_spec.mu0, "sigma_hat": float(np.exp(prior_spec.sigma_log_location))}
    return {}


def make_config(curve, policy, calibration, prior_spec, sigma0=None):
    cd = dict(curve)
    if sigma0 is not None and cd["family"] not in ("beta_cdf",):
        cd["sigma0"] = sigma0
        if cd["family"] == "probit":
            cd["sigma"] = sigma0
    return ExperimentConfig(
        policy=policy,
        prior=prior_spec.config(),
        curve=cd,
        calibration=calibration,
        n_steps=N_STEPS,
        slices=SLICES,
        targets=TARGETS,
        n_particles=10_000,
        grid_points=200,
        grid_half_width=14.0,
        reference_at_slices=REF_AT,
        reference_fixed_n=REF_FIXED_N,
        # interval widths are needed both to fix w* and to score false
        # certainty, so the reference bisection cannot be skipped here
        reference_widths=True,
        policy_kwargs=_policy_kwargs(policy, prior_spec),
    )


def build_jobs():
    jobs = {}
    cal_prior = rotem_prior("well", "independent").spec
    for dgp, curve in BLOCK_I_DGPS:
        for pol in POLICIES:
            jobs[f"I|{dgp}|{pol}|cal|s3"] = (make_config(curve, pol, "well", cal_prior),
                                             REPS)
    # scale sub-block
    for dgp, curve in SUB_DGPS:
        for s0 in (1.5, 6.0):
            for pol in SUB_POLICIES:
                jobs[f"Ib|{dgp}|{pol}|cal|s{s0}"] = (
                    make_config(curve, pol, "well", cal_prior, sigma0=s0), REPS_SUB)
    # prior sub-block
    shifted = rotem_prior("poor", "independent").spec
    narrow = PriorSpec(mu0=30.0, sigma_mu=2.0, tau=0.25, tau_is_variance=True,
                       dependent=False, name="narrow")
    for dgp, curve in SUB_DGPS:
        for pname, pspec, cal in (("shifted", shifted, "poor"), ("narrow", narrow, "well")):
            for pol in SUB_POLICIES:
                jobs[f"Ic|{dgp}|{pol}|{pname}|s3"] = (
                    make_config(curve, pol, cal, pspec), REPS_SUB)
    # Block II: poorly calibrated grid [8, 36]
    poor_prior = rotem_prior("poor", "independent").spec
    for dgp, curve in BLOCK_II_DGPS:
        for pol in POLICIES:
            jobs[f"II|{dgp}|{pol}|shifted|s3"] = (
                make_config(curve, pol, "poor", poor_prior), REPS_SUB)
    return jobs


def _worker(args):
    key, cfg, start, stop = args
    rows, failures, designs = [], [], []
    for r in range(start, stop):
        rr, ff, tt = _single(cfg, r, SEED_BASE, CRN_SEED)
        for row in rr:
            row["cell"] = key
            block, dgp, pol, prior_name, scale = key.split("|")
            row.update({"block": block, "dgp": dgp, "pol": pol,
                        "prior_name": prior_name, "scale": scale})
        rows.extend(rr)
        failures.extend(ff)
        # keep the applied stimuli: the empirical design distribution is the
        # input to the policy-dependent pseudo-truth analysis
        for t in tt:
            if t["x"] is not None:
                designs.append({"cell": key, "replicate": t["replicate"],
                                "x": np.asarray(t["x"], dtype=float).tolist()})
    return rows, failures, designs


def freeze_thresholds(df: pd.DataFrame) -> dict:
    """Protocol section 4: w* from the correct-probit adaptive baseline."""
    base = df[(df.block == "I") & (df.dgp == "probit") & (df.pol == "entropy_vector")]
    th = {"rule": "25th percentile of reference 95% interval width, correct probit, "
                  "calibrated prior, entropy_vector policy",
          "source_cell": "I|probit|entropy_vector|cal|s3",
          "n_replicates": int(base.replicate.nunique()), "thresholds": {}}
    for n in sorted(base.n.unique()):
        g = base[base.n == n]
        for t in TARGETS:
            col = f"{t}_ref_width"
            if col in g and g[col].notna().any():
                th["thresholds"][f"{t}|n{n}"] = float(np.nanquantile(g[col], 0.25))
    th["hash"] = config_hash(th["thresholds"])
    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLD_PATH.write_text(json.dumps(th, indent=2))
    return th


def main():
    global CURVE_SCALE
    t0 = time.time()
    CURVE_SCALE = curve_scale_lookup()
    jobs = build_jobs()
    chunk = 25
    tasks = []
    for key, (cfg, reps) in jobs.items():
        for s in range(0, reps, chunk):
            tasks.append((key, cfg, s, min(s + chunk, reps)))
    print(f"Phase 3B screening: {len(jobs)} cells, {sum(r for _, r in jobs.values())} "
          f"replicates, n={N_STEPS}", flush=True)

    rows, failures, designs, done = [], [], [], 0
    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        for rr, ff, dd in pool.imap_unordered(_worker, tasks):
            rows.extend(rr)
            failures.extend(ff)
            designs.extend(dd)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"  [screen] {done}/{len(tasks)} chunks {el:6.1f}s "
                      f"(eta {el/done*(len(tasks)-done):6.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    raw = save(df, "phase3b_screen_raw")
    save(pd.DataFrame(designs), "phase3b_designs")
    th = freeze_thresholds(df)
    print(f"\nfrozen false-certainty thresholds (hash {th['hash']}):")
    for k, v in th["thresholds"].items():
        print(f"  w*[{k}] = {v:.4f}")

    summ = summarise(df, th)
    spath = save_csv(summ, "phase3b_screen_summary")

    print("\n=== Block I, n=50, reference posterior ===")
    sel = summ[(summ.block == "I") & (summ.n == 50) & (summ.backend == "ref")
               & (summ.target != "q0.5")]
    print(sel.pivot_table(index=["dgp", "target"], columns="pol",
                          values="false_certainty")
          .to_string(float_format=lambda v: f"{v:6.3f}"))
    print("\ncoverage:")
    print(sel.pivot_table(index=["dgp", "target"], columns="pol", values="coverage")
          .to_string(float_format=lambda v: f"{v:6.3f}"))

    write_manifest(
        experiment="phase3b_screen",
        config={"block_I_dgps": [d for d, _ in BLOCK_I_DGPS],
                "block_II_dgps": [d for d, _ in BLOCK_II_DGPS],
                "policies": POLICIES, "sub_policies": SUB_POLICIES,
                "reps": REPS, "reps_sub": REPS_SUB, "n_steps": N_STEPS,
                "slices": list(SLICES), "reference_at": list(REF_AT),
                "reference_fixed_n": REF_FIXED_N, "crn_seed": CRN_SEED,
                "thresholds": th},
        seed_base=SEED_BASE,
        requested=sum(r for _, r in jobs.values()),
        completed=int(len(df)), failures=failures,
        tolerances={"reference_fixed_n": REF_FIXED_N,
                    "screening_note": "exploratory tier; coverage SE ~0.03"},
        mc_se={f"coverage::{r.cell}::{r.backend}::n{r.n}::{r.target}": r.coverage_se
               for r in summ.itertuples()},
        degeneracies={
            "n_rows": int(len(df)),
            "no_overlap": int((~df.overlap.astype(bool)).sum()),
            "all_zero": int(df.all_zero.sum()), "all_one": int(df.all_one.sum()),
            "target_outside_grid": int(df.target_outside_grid.sum())
            if "target_outside_grid" in df else -1,
            "ref_not_converged_flagged": int(df.ref_converged.isna().sum())
            if "ref_converged" in df else -1,
        },
        artifacts=[raw, spath, THRESHOLD_PATH], wall=time.time() - t0,
        notes="Exploratory screening tier of the Phase 3B failure atlas.",
    )


def summarise(df: pd.DataFrame, th: dict) -> pd.DataFrame:
    """Per-cell metrics for both backends, scored against frozen thresholds."""
    from src.decision_loss import loss_curve, threshold_loss

    out = []
    for (cell, n), g in df.groupby(["cell", "n"]):
        block, dgp, pol, prior_name, scale = cell.split("|")
        for backend, cov_c, med_c, w_c in (
            ("particle", "{t}_covered", "{t}_bayes", "{t}_width"),
            ("ref", "{t}_ref_covered", "{t}_ref_mean", "{t}_ref_width"),
        ):
            for t in TARGETS:
                cc, mc, wc = cov_c.format(t=t), med_c.format(t=t), w_c.format(t=t)
                if cc not in g or g[cc].isna().all():
                    continue
                cov = g[cc].astype(float)
                est = g[mc] if mc in g else pd.Series(np.nan, index=g.index)
                width = g[wc] if wc in g else pd.Series(np.nan, index=g.index)
                truth = g[f"{t}_true"]
                err = est - truth
                nrep = len(g)
                w_star = th["thresholds"].get(f"{t}|n{n}", np.nan)
                narrow = width <= w_star
                fc = float(((~cov.astype(bool)) & narrow).mean()) if np.isfinite(w_star) else np.nan
                sigma_true = CURVE_SCALE.get((dgp, scale), 3.0)
                delta = 0.5 * sigma_true
                dangerous = float((err < -delta).mean())
                loss = threshold_loss(est, truth, 10.0, 1.0)
                out.append({
                    "cell": cell, "block": block, "dgp": dgp, "pol": pol,
                    "prior_name": prior_name, "scale": scale, "n": n,
                    "backend": backend, "target": t, "n_rep": nrep,
                    "coverage": float(cov.mean()),
                    "coverage_se": float(np.sqrt(max(cov.mean() * (1 - cov.mean()), 0) / nrep)),
                    "bias": float(err.mean()), "rmse": float(np.sqrt((err**2).mean())),
                    "mean_width": float(width.mean()), "median_width": float(width.median()),
                    "w_star": w_star, "frac_narrow": float(narrow.mean()),
                    "false_certainty": fc,
                    "false_certainty_se": float(np.sqrt(max(fc * (1 - fc), 0) / nrep))
                    if np.isfinite(fc) else np.nan,
                    "delta": delta, "curve_scale": sigma_true,
                    "dangerous_error": dangerous,
                    "dangerous_error_se": float(np.sqrt(max(dangerous * (1 - dangerous), 0) / nrep)),
                    "decision_loss_10to1": float(np.nanmean(loss)),
                    "unresolved_rate": float((~g.overlap.astype(bool)).mean()),
                    "frac_all_equal": float((g.all_zero | g.all_one).mean()),
                    "x_min_mean": float(g.x_min.mean()), "x_max_mean": float(g.x_max.mean()),
                    "target_true": float(truth.iloc[0]),
                    "target_outside_sampled": float(
                        ((truth < g.x_min) | (truth > g.x_max)).mean()),
                })
    return pd.DataFrame(out)


if __name__ == "__main__":
    main()
