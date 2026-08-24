"""Phase 4A -- is the Phase 3 failure visible in the data the design collects?

Phase 3 established that a curve which agrees with the probit everywhere the
adaptive design looks, and departs from it only at the estimand, drives
q_0.99 coverage to 0.433 +/- 0.029 while producing the *narrowest* intervals in
the study (`docs/claims.md` C5).  A referee's first reaction is that the
analyst would notice: fit a bigger model, run a posterior predictive check,
look at the residuals.  This experiment asks whether that is true, and the
answer decides whether a safeguard is a research contribution or a
write-up note.

Four layers per replicate, in increasing order of what the analyst is assumed
to know, all conditioned on the design that was actually run:

  L0  `design_information` -- the discriminating content of the sample, in
      nats, and the Neyman-Pearson ceiling on the power of *any* level-alpha
      test that follows from it.  Also the counterfactual: how many of the n
      responses would have come out differently had the truth been the probit.
  L1  posterior predictive p-values under the fitted probit, omnibus and
      tail-directed.
  L2  Bayes factors against four alternative link families under a common
      prior -- including the tail-perturbed family frozen at the true shift,
      which is the *oracle* alternative no real analyst would have guessed.
  L3  a likelihood ratio against the tail-perturbed family with its shift free:
      the exact functional form of the truth, with only its size estimated.

Each cell is paired by common random numbers with the correct-probit cell
under the same policy, which serves two purposes: it is the null that
calibrates L1-L3 (so the reported detection rates are not asymptotic
approximations at n = 50 under a sequential design), and it is the reference
curve for L0.

The design cells are the Phase 3B Block I cells, at the same horizon, prior and
grid, so that every detectability number lands next to a coverage number that
Phase 3 already established rather than next to a new one.
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
from _cli import parse as _parse_cli  # noqa: E402
from _runner import N_WORKERS, degeneracy_counts, save, save_csv, write_manifest  # noqa: E402
from phase3b_screen import BLOCK_I_DGPS, make_config  # noqa: E402

from src import model_check as mc  # noqa: E402
from src.priors import rotem_prior  # noqa: E402
from src.response_models import ProbitCurve  # noqa: E402
from src.simulator import ExperimentConfig, build_curve, run_experiment, true_targets  # noqa: E402

RUN = _parse_cli("phase4_undetectability", replicates=200,
                 seed_base=20260940, crn_seed=414001)

SEED_BASE = RUN.seed_base
CRN_SEED = RUN.crn_seed
N_REPS = RUN.replicates
CHUNK = 10

# Phase 3B Block I settings, unchanged, so detectability and coverage are read
# off the same runs rather than off two experiments that merely resemble one
# another.
N_STEPS = 50
REF_FIXED_N = 513
TARGET = "q0.99"
ALPHA = 0.05
PPC_DRAWS = 200

# The matched probit every alternative departs from: curve_families builds each
# DGP by perturbing exactly this one, so it is the reference for L0 and the
# null DGP for the calibration cells.
REF_MU, REF_SIGMA = 30.0, 3.0

# `probit` is the null cell and must be present: it calibrates every test.
DGPS = ["probit", "tail_1.0", "tail_0.5", "robit_1.0"]
POLICIES = ["entropy_vector", "entropy_q0.95", "fixed_design", "uniform_grid"]

CURVES = dict(BLOCK_I_DGPS)

# The shift the oracle alternative is frozen at: the primary cell's, so that
# one column means the same thing in every cell and the null cell calibrates it
# directly.  Letting each DGP be scored against an alternative frozen at *its
# own* truth would need a separate null per shift and would leave the probit
# cell -- which has no true shift -- without one.
ORACLE_SHIFT = 1.0

# Alternatives offered to the Bayes-factor layer.  The first three are families
# a working analyst might actually try, and are summarised as
# `log_bf_best_guessable`; the fourth is the oracle, which is the exact shape
# of the tail_1.0 truth and which nobody would have guessed.  Note that the
# robit family is genuinely in the guessable set, so for the robit_1.0 DGP the
# "guessable" column is itself near-oracular -- a contrast the write-up makes
# rather than hides.
GUESSABLE = [mc.LOGISTIC, mc.CLOGLOG, mc.ROBIT]
ORACLE = mc.tail_fixed(ORACLE_SHIFT)


def _config(dgp: str, policy: str) -> ExperimentConfig:
    prior_spec = rotem_prior("well", "independent").spec
    cfg = make_config(CURVES[dgp], policy, "well", prior_spec)
    # The reference posterior is built once here instead, and reused for the
    # predictive check and the outcome summaries, so the simulator's own
    # reference pass would be pure duplicated cost.
    return ExperimentConfig(**{**cfg.as_dict(),
                              "reference_at_slices": (), "slices": (N_STEPS,)})


def analyse(dgp: str, policy: str, replicate: int) -> dict:
    """One replicate: run the design, then interrogate what it collected."""
    from src.posterior_grid import build_reference_posterior
    from src.simulator import latent_uniforms

    cfg = _config(dgp, policy)
    res = run_experiment(cfg, replicate, SEED_BASE, crn_seed=CRN_SEED)
    x, y = np.asarray(res["x"], dtype=float), np.asarray(res["y"]).astype(int)
    u = latent_uniforms(CRN_SEED, replicate, N_STEPS)

    prior = rotem_prior("well", "independent")
    curve = build_curve(CURVES[dgp])
    reference = ProbitCurve(REF_MU, REF_SIGMA)
    truths = true_targets(curve, (TARGET,))

    row = {"dgp": dgp, "pol": policy, "replicate": replicate, "n": N_STEPS,
           "x_min": float(x.min()), "x_max": float(x.max()),
           "n_distinct_x": int(len(np.unique(x))), "n_positive": int(y.sum()),
           "all_zero": bool(y.sum() == 0), "all_one": bool(y.sum() == len(y))}

    # Where the tail-perturbed curve starts to depart from the probit.  The
    # fraction of the budget spent above it is the design-side statement of the
    # mechanism, and is defined for every DGP so the cells stay comparable.
    onset = REF_MU + REF_SIGMA * float(mc.norm.ppf(mc.TAIL_P0))
    row["perturbation_onset"] = onset
    row["frac_above_onset"] = float(np.mean(x > onset))

    # L0 -- what could be learned at all
    row.update(mc.design_information(curve, reference, x, u))

    # L3 -- oracle likelihood ratio (shape free).  Reported for every DGP: on
    # the non-tail curves it is a misdirected test, which is itself the point
    # of the comparison with the Bayes-factor layer.
    probit_fit = mc.fit_mle(mc.PROBIT, x, y)
    row.update(mc.lr_statistic(x, y, probit_fit=probit_fit))
    row["probit_mle_m"] = probit_fit.m
    row["probit_mle_s"] = probit_fit.s

    # L2 -- evidence on a shared box
    spec = mc.shared_box(prior, x, y)
    logz_probit, edge = mc.log_evidence(mc.PROBIT, prior, x, y, spec)
    row["log_evidence_probit"] = logz_probit
    edges = [edge]
    best = -np.inf
    for fam in GUESSABLE + [ORACLE]:
        lz, e = mc.log_evidence(fam, prior, x, y, spec)
        edges.append(e)
        bf = lz - logz_probit
        if fam is ORACLE:
            row["log_bf_oracle"] = bf
        else:
            row[f"log_bf_{fam.name}"] = bf
            best = max(best, bf)
    row["log_bf_best_guessable"] = float(best)
    row["evidence_max_boundary_mass"] = float(max(edges))

    # L1 -- posterior predictive checks, and the outcome they sit beside
    ref = build_reference_posterior(prior, x, y, fixed_n=REF_FIXED_N)
    rng = np.random.default_rng([SEED_BASE, replicate, 0x9DCE])
    row.update(mc.posterior_predictive_check(prior, x, y, rng,
                                             n_draws=PPC_DRAWS, posterior=ref))

    truth = truths[TARGET]
    row.update({
        f"{TARGET}_true": truth,
        f"{TARGET}_ref_mean": ref.mean(TARGET),
        f"{TARGET}_ref_sd": ref.sd(TARGET),
        f"{TARGET}_rank_ref": ref.cdf_at(TARGET, truth),
        f"{TARGET}_ref_covered": ref.covered(TARGET, truth, 0.95),
        f"{TARGET}_err": ref.mean(TARGET) - truth,
        "ref_boundary_mass": float(ref.convergence["boundary_mass"]),
    })
    return row


def _worker(args):
    dgp, policy, start, stop = args
    rows, failures = [], []
    for r in range(start, stop):
        try:
            rows.append(analyse(dgp, policy, r))
        except Exception as exc:  # noqa: BLE001 - recorded, never hidden
            import traceback
            failures.append({"dgp": dgp, "pol": policy, "replicate": r,
                             "error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc(limit=6)})
    return rows, failures


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

# Each test is (column, direction): "hi" rejects for large values, "lo" for
# small ones.  Critical values come from the correct-probit cell under the same
# policy -- an exact null sample at the same horizon under the same sequential
# design -- rather than from an asymptotic reference distribution that n = 50
# and a data-dependent design would both call into question.
TESTS = {
    "lr_oracle": ("lr", "hi"),
    "ppc_chi2": ("ppp_chi2", "lo"),
    "ppc_tail": ("ppp_tail", "lo"),
    "bf_oracle": ("log_bf_oracle", "hi"),
    "bf_guessable": ("log_bf_best_guessable", "hi"),
}


def _critical_values(null: pd.DataFrame, alpha: float) -> dict:
    """Level-alpha critical values from the null cell, with the tie convention.

    The oracle LR is exactly zero on a large share of null replicates (the
    shift estimate sits on its lower boundary), so its upper alpha-quantile can
    itself be zero.  Rejecting on `stat >= crit` would then reject everywhere;
    the convention here is strict exceedance for "hi" tests, which keeps the
    realised level at or below alpha and is recorded as `null_level`.
    """
    out = {}
    for name, (col, side) in TESTS.items():
        if col not in null.columns or null[col].isna().all():
            continue
        v = null[col].dropna().to_numpy(dtype=float)
        out[name] = float(np.quantile(v, 1 - alpha if side == "hi" else alpha))
    return out


def _reject(df: pd.DataFrame, name: str, crit: float) -> np.ndarray:
    col, side = TESTS[name]
    v = df[col].to_numpy(dtype=float)
    return (v > crit) if side == "hi" else (v <= crit)


def signal_split(g: pd.DataFrame) -> dict:
    """Coverage among runs that collected no signal, versus runs that did.

    `realized_flips` counts the responses that would have come out differently
    had the truth been the matched probit, so `realized_flips == 0` marks a run
    whose data are bit-identical to what the correct model would have produced.
    Splitting a cell on it separates the runs where inference had *something*
    to go on from the runs where it had nothing, and the coverage gap across
    that split is the mechanism stated as directly as a simulation can state
    it.

    This is a **mechanism decomposition, not a diagnostic**: the split variable
    is a counterfactual that requires knowing the true curve, so it is not
    computable in a real experiment. Its value to Phase 5 is as the target a
    computable proxy would have to approximate.
    """
    out = {}
    for label, sub in (("blind", g[g.realized_flips == 0]),
                       ("informed", g[g.realized_flips > 0])):
        n = len(sub)
        cov = sub[f"{TARGET}_ref_covered"].astype(float)
        out[f"n_{label}"] = int(n)
        out[f"coverage_{label}"] = float(cov.mean()) if n else np.nan
        out[f"coverage_{label}_se"] = (
            float(cov.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan)
        out[f"bias_{label}"] = float(sub[f"{TARGET}_err"].mean()) if n else np.nan
    return out


def summarise(df: pd.DataFrame, alpha: float = ALPHA) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, crit_rows = [], []
    for pol, g_pol in df.groupby("pol"):
        null = g_pol[g_pol.dgp == "probit"]
        crit = _critical_values(null, alpha)
        for name, c in crit.items():
            lvl = float(np.mean(_reject(null, name, c)))
            crit_rows.append({"pol": pol, "test": name, "critical_value": c,
                              "null_level": lvl, "n_null": int(len(null)),
                              "null_level_se": float(np.sqrt(max(lvl * (1 - lvl), 0)
                                                             / max(len(null), 1)))})
        for dgp, g in g_pol.groupby("dgp"):
            n = len(g)
            cov = g[f"{TARGET}_ref_covered"].astype(float)
            rec = {
                "dgp": dgp, "pol": pol, "n_rep": n,
                "coverage": float(cov.mean()),
                "coverage_se": float(cov.std(ddof=1) / np.sqrt(n)),
                "bias": float(g[f"{TARGET}_err"].mean()),
                "bias_se": float(g[f"{TARGET}_err"].std(ddof=1) / np.sqrt(n)),
                "x_max_mean": float(g.x_max.mean()),
                "frac_above_onset": float(g.frac_above_onset.mean()),
                "design_kl_nats": float(g.design_kl_nats.mean()),
                "design_kl_nats_se": float(g.design_kl_nats.std(ddof=1) / np.sqrt(n)),
                "max_power_bound": float(g.max_power_bound.mean()),
                "expected_flips": float(g.expected_flips.mean()),
                "realized_flips_mean": float(g.realized_flips.mean()),
                "frac_zero_flips": float((g.realized_flips == 0).mean()),
                "log_bf_oracle_median": float(g.log_bf_oracle.median()),
                "log_bf_guessable_median": float(g.log_bf_best_guessable.median()),
                "evidence_max_boundary_mass": float(g.evidence_max_boundary_mass.max()),
                **signal_split(g),
            }
            for name, c in crit.items():
                r = _reject(g, name, c).astype(float)
                rec[f"power_{name}"] = float(r.mean())
                rec[f"power_{name}_se"] = float(r.std(ddof=1) / np.sqrt(n))
            rows.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(crit_rows)


def main():
    t0 = time.time()
    cells = [(d, p) for d in DGPS for p in POLICIES]
    print(f"Phase 4A undetectability: {len(cells)} cells x {N_REPS} replicates "
          f"({N_STEPS} steps, target {TARGET}, alpha {ALPHA})", flush=True)

    tasks = [(d, p, s, min(s + CHUNK, N_REPS))
             for d, p in cells for s in range(0, N_REPS, CHUNK)]
    all_rows, all_fail, done = [], [], 0

    def _progress():
        el = time.time() - t0
        print(f"  {done}/{len(tasks)} chunks  {el:6.1f}s "
              f"(eta {el / max(done, 1) * (len(tasks) - done):6.1f}s)", flush=True)

    if N_WORKERS == 1:
        # Serial path.  Not only for debugging: `fork` does not exist on
        # Windows and `spawn` re-imports this module in every child, which
        # re-enters the CLI parser, so a single-worker run is the portable way
        # to reproduce a cell.
        results = (_worker(t) for t in tasks)
    else:
        methods = mp.get_all_start_methods()
        ctx = mp.get_context("fork" if "fork" in methods else "spawn")
        pool = ctx.Pool(N_WORKERS)
        results = pool.imap_unordered(_worker, tasks)

    for rows, failures in results:
        all_rows.extend(rows)
        all_fail.extend(failures)
        done += 1
        if done % 10 == 0 or done == len(tasks):
            _progress()
    if N_WORKERS != 1:
        pool.close()
        pool.join()

    df = pd.DataFrame(all_rows).sort_values(["dgp", "pol", "replicate"])
    raw = save(df, "phase4_undetectability_raw")
    summ, crit = summarise(df)
    spath = save_csv(summ, "phase4_undetectability_summary")
    cpath = save_csv(crit, "phase4_undetectability_calibration")

    show = ["dgp", "pol", "coverage", "bias", "design_kl_nats", "expected_flips",
            "frac_zero_flips", "max_power_bound", "power_lr_oracle",
            "power_bf_oracle", "power_ppc_tail", "power_ppc_chi2"]
    print("\n=== detectability beside the outcome (reference posterior, n=50) ===")
    print(summ[show].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print("\n=== null calibration (correct probit, same policy) ===")
    print(crit.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    write_manifest(
        experiment="phase4_undetectability",
        config={"dgps": DGPS, "policies": POLICIES, "n_steps": N_STEPS,
                "target": TARGET, "alpha": ALPHA, "ppc_draws": PPC_DRAWS,
                "reference_fixed_n": REF_FIXED_N, "crn_seed": CRN_SEED,
                "calibration": "null cell = probit DGP, same policy",
                "prior": "rotem well/independent", "grid_points": 200},
        seed_base=SEED_BASE, requested=N_REPS * len(cells), completed=int(len(df)),
        failures=all_fail,
        tolerances={"reference_fixed_n": REF_FIXED_N,
                    "evidence_grid_n": 385,
                    "evidence_max_boundary_mass": float(df.evidence_max_boundary_mass.max()),
                    "ref_max_boundary_mass": float(df.ref_boundary_mass.max())},
        mc_se={f"{r.dgp}::{r.pol}::coverage": r.coverage_se for r in summ.itertuples()}
        | {f"{r.dgp}::{r.pol}::power_lr_oracle": r.power_lr_oracle_se
           for r in summ.itertuples()},
        degeneracies=degeneracy_counts(df),
        artifacts=[raw, spath, cpath], wall=time.time() - t0,
        notes="Phase 4A: detectability of the Phase 3 tail failure from the "
              "data the design collects. Screening tier.",
    )


if __name__ == "__main__":
    main()
