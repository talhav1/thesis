"""Single entry point: run an experiment from a config file.

    python experiments/run.py configs/experiments/phase1_baseline.json
    python experiments/run.py configs/experiments/phase1_baseline.json --allow-dirty

A run that can only be reproduced from a shell history is not reproducible.
Every published run should therefore have a config file in
``configs/experiments/`` recording the replicate count, the seed base and the
CRN seed actually used, and the manifest records the config path and its hash
so a result can be traced back to it.

The config is a small JSON object:

    {
      "experiment": "phase1_baseline",
      "replicates": 250,
      "seed_base": 20260813,
      "crn_seed": 777001,
      "workers": 8,
      "note": "Phase 1 gate run."
    }

Only ``experiment`` is required; anything omitted falls back to the script's
own default.  Command-line flags override the config, so a config can be reused
for a cheaper pilot with ``--replicates 20``.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.manifest import config_hash  # noqa: E402
from src.provenance import new_run_id, require_clean_tree  # noqa: E402

EXPERIMENTS = REPO_ROOT / "experiments"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="path to a configs/experiments/*.json file")
    ap.add_argument("--replicates", "-r", type=int, default=None)
    ap.add_argument("--seed-base", type=int, default=None)
    ap.add_argument("--crn-seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--note", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and print the invocation without running it")
    args = ap.parse_args(argv)

    cfg = json.loads(args.config.read_text())
    experiment = cfg["experiment"]
    script = EXPERIMENTS / f"{experiment}.py"
    if not script.exists():
        ap.error(f"no such experiment script: {script}")

    def pick(flag, key):
        return flag if flag is not None else cfg.get(key)

    forwarded: list[str] = []
    for flag, key, opt in (
        (args.replicates, "replicates", "--replicates"),
        (args.seed_base, "seed_base", "--seed-base"),
        (args.crn_seed, "crn_seed", "--crn-seed"),
        (args.workers, "workers", "--workers"),
    ):
        value = pick(flag, key)
        if value is not None:
            forwarded += [opt, str(value)]
    note = pick(args.note, "note")
    if note:
        forwarded += ["--note", note]
    allow_dirty = args.allow_dirty or bool(cfg.get("allow_dirty", False))
    if allow_dirty:
        forwarded.append("--allow-dirty")

    run_id = args.run_id or new_run_id(experiment)

    print(f"config    {args.config}  (hash {config_hash(cfg)})")
    print(f"script    {script.relative_to(REPO_ROOT)}")
    print(f"run_id    {run_id}")
    print(f"forwarded {' '.join(forwarded) or '(script defaults)'}")
    if args.dry_run:
        return 0

    require_clean_tree(allow_dirty, context=f"experiment '{experiment}'")

    os.environ["RUN_ID"] = run_id
    os.environ["CONFIG_PATH"] = str(args.config)
    os.environ["INVOKED_BY"] = "experiments/run.py"
    os.environ["CLI_ENTRY"] = experiment
    if allow_dirty:
        os.environ["ALLOW_DIRTY"] = "1"

    sys.argv = [str(script), *forwarded]
    sys.path.insert(0, str(EXPERIMENTS))
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
