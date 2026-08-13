"""Shared execution harness for experiment scripts.

Responsibilities: parallel execution over replicate chunks, incremental
persistence of raw per-replicate rows, and manifest emission.  No experiment
script writes to `results/` except through here, so the invariant "every
artefact has a manifest" is structural rather than a matter of discipline.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.manifest import RunManifest, mc_se_mean, mc_se_proportion  # noqa: E402
from src.provenance import active_spec, new_run_id  # noqa: E402
from src.simulator import run_batch  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
N_WORKERS = max(1, min(int(os.environ.get("N_WORKERS", "2")), os.cpu_count() or 1))
CHUNK = int(os.environ.get("CHUNK", "20"))


def _worker(args):
    key, config, start, stop, seed_base, crn_seed = args
    rows, failures, trajectories = [], [], []
    for r in range(start, stop):
        rows_r, fail_r, traj_r = _single(config, r, seed_base, crn_seed)
        rows.extend(rows_r)
        failures.extend(fail_r)
        trajectories.extend(traj_r)
    for row in rows:
        row["setting"] = key
    return key, rows, failures, trajectories


def _single(config, replicate, seed_base, crn_seed):
    import traceback

    from src.simulator import run_experiment

    try:
        res = run_experiment(config, replicate, seed_base, crn_seed=crn_seed)
    except Exception as exc:  # noqa: BLE001 - recorded, never hidden
        return [], [{"replicate": replicate, "error": f"{type(exc).__name__}: {exc}",
                     "traceback": traceback.format_exc(limit=6)}], []
    x, y = res.pop("x", None), res.pop("y", None)
    rows = []
    for n_done, rec in res["slices"].items():
        row = {"replicate": res["replicate"], "policy": res["policy"],
               "calibration": res["calibration"], "prior_name": res["prior_name"],
               "curve": res["curve"], "n": n_done}
        row.update(rec)
        rows.append(row)
    return rows, [], [{"replicate": replicate, "x": x, "y": y}]


def run_jobs(jobs, n_replicates, seed_base, crn_seed, label, progress_every=5):
    """`jobs` maps a setting key -> ExperimentConfig.  Runs all of them."""
    tasks = []
    for key, cfg in jobs.items():
        for start in range(0, n_replicates, CHUNK):
            tasks.append((key, cfg, start, min(start + CHUNK, n_replicates),
                          seed_base, crn_seed))

    all_rows, all_failures, all_traj = [], [], {}
    t0 = time.time()
    done = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        for key, rows, failures, traj in pool.imap_unordered(_worker, tasks):
            all_rows.extend(rows)
            all_failures.extend(failures)
            all_traj.setdefault(key, []).extend(traj)
            done += 1
            if done % progress_every == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"  [{label}] {done}/{len(tasks)} chunks  {el:6.1f}s "
                      f"(eta {el / done * (len(tasks) - done):6.1f}s)", flush=True)
    return all_rows, all_failures, all_traj, time.time() - t0


def current_run_id(experiment: str = "run") -> str:
    """The run identifier for this process, minted once and reused."""
    rid = os.environ.get("RUN_ID", "").strip()
    if not rid:
        rid = new_run_id(experiment)
        os.environ["RUN_ID"] = rid
    return rid


def _stamped(name: str) -> str:
    """Append the short run token so artefacts from different runs never
    silently overwrite each other; the full id lives in the manifest."""
    rid = os.environ.get("RUN_ID", "").strip()
    if not rid or os.environ.get("STAMP_ARTIFACTS", "1") != "1":
        return name
    return f"{name}__{rid.rsplit('_', 1)[-1]}"


def save(df: pd.DataFrame, name: str, subdir="raw", stamp: bool = False) -> Path:
    if stamp:
        name = _stamped(name)
    path = RESULTS / subdir / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def save_csv(df: pd.DataFrame, name: str, subdir="summaries", stamp: bool = False) -> Path:
    if stamp:
        name = _stamped(name)
    path = RESULTS / subdir / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def degeneracy_counts(df: pd.DataFrame) -> dict:
    cols = [c for c in df.columns if c.startswith("degenerate_")] + [
        "all_zero", "all_one", "overlap", "target_outside_grid",
    ]
    out = {}
    for c in cols:
        if c in df.columns:
            out[c] = int(df[c].sum())
    out["n_rows"] = int(len(df))
    if "mle_finite" in df.columns:
        out["mle_non_finite"] = int((~df["mle_finite"].astype(bool)).sum())
    return out


def mse_with_se(errors) -> tuple[float, float]:
    """MSE and its Monte Carlo standard error (SE of the mean of squares)."""
    e = np.asarray(errors, dtype=float)
    e = e[np.isfinite(e)]
    if len(e) < 2:
        return float("nan"), float("nan")
    sq = e**2
    return float(sq.mean()), float(sq.std(ddof=1) / np.sqrt(len(sq)))


def write_manifest(experiment, config, seed_base, requested, completed, failures,
                   tolerances, mc_se, degeneracies, artifacts, wall, notes="",
                   *, run_id=None, crn_seed=None, allow_dirty=None,
                   replicates_unit="replicates"):
    """Publish a manifest.  `run_id` defaults to this process's identifier so
    the manifest, the raw table and the summaries share one key."""
    spec = active_spec()
    run_id = run_id or spec["run_id"] or current_run_id(experiment)
    crn_seed = crn_seed if crn_seed is not None else spec["crn_seed"]
    allow_dirty = spec["allow_dirty"] if allow_dirty is None else allow_dirty
    if not notes and spec["note"]:
        notes = spec["note"]
    man = RunManifest(
        run_id=run_id,
        crn_seed=crn_seed,
        allow_dirty=allow_dirty,
        replicates_unit=replicates_unit,
        experiment=experiment,
        config=config,
        seed_base=seed_base,
        n_replicates_requested=requested,
        n_replicates_completed=completed,
        n_failures=len(failures),
        failures=failures,
        tolerances=tolerances,
        monte_carlo_se=mc_se,
        degeneracy_counts=degeneracies,
        artifacts=artifacts,
        wall_seconds=wall,
        notes=notes,
    )
    path = man.write(experiment)
    print(f"  manifest -> {path}  (run_id={run_id})")
    return path
