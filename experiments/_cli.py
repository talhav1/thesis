"""Shared command line for experiment scripts.

Every experiment previously read its replicate count from ``sys.argv[1]`` and
hard-coded its seed base as a module constant, so a run could only be
reproduced from a shell history.  This module gives all of them one entry
surface:

    python experiments/phase1_baseline.py --replicates 250 --allow-dirty

and, through ``experiments/run.py``, one config-file surface:

    python experiments/run.py configs/experiments/phase1_baseline.json

Scripts stay backward compatible: a bare positional integer is still read as
the replicate count, so ``python experiments/phase1_baseline.py 250`` behaves
as before.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.provenance import new_run_id, publish_spec, require_clean_tree  # noqa: E402


@dataclass
class RunSpec:
    experiment: str
    replicates: int
    seed_base: int
    crn_seed: int | None
    run_id: str
    allow_dirty: bool
    note: str
    workers: int | None

    def manifest_kwargs(self) -> dict:
        """Provenance fields for `_runner.write_manifest`.

        Usually unnecessary: `parse` publishes the same values through
        `src.provenance.publish_spec`, and `write_manifest` picks them up as
        defaults.  Use this only to override them at a call site.
        """
        return {"run_id": self.run_id, "crn_seed": self.crn_seed,
                "allow_dirty": self.allow_dirty}


def _is_entry_point(experiment: str) -> bool:
    """True only when this script is the process entry point.

    Several experiments import each other (`phase3a_attribution` imports
    `phase3a_pilot`; `phase3b_confirm` imports `phase3b_screen`).  Parsing argv
    at import time would make the importing script's flags reach the imported
    one, so on import the defaults are returned silently and no clean-tree
    check runs -- the entry-point script has already done it.
    """
    if os.environ.get("CLI_ENTRY", "") == experiment:
        return True
    try:
        return Path(sys.argv[0]).stem == experiment
    except Exception:  # noqa: BLE001
        return False


def parse(experiment: str, *, replicates: int, seed_base: int,
          crn_seed: int | None = None, argv: list[str] | None = None) -> RunSpec:
    """Parse the shared arguments and enforce the clean-tree rule.

    The defaults passed in are the script's own historical constants, so an
    invocation with no flags reproduces the published run.
    """
    if not _is_entry_point(experiment) and argv is None:
        return RunSpec(experiment=experiment, replicates=replicates,
                       seed_base=seed_base, crn_seed=crn_seed,
                       run_id=os.environ.get("RUN_ID") or f"{experiment}_imported",
                       allow_dirty=os.environ.get("ALLOW_DIRTY", "") == "1",
                       note="", workers=None)
    parser = argparse.ArgumentParser(prog=experiment, description=f"Run {experiment}.")
    parser.add_argument("replicates_pos", nargs="?", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--replicates", "-r", type=int, default=None,
                        help=f"replicates per cell (default {replicates})")
    parser.add_argument("--seed-base", type=int, default=None,
                        help=f"authoritative seed base (default {seed_base})")
    parser.add_argument("--crn-seed", type=int, default=None,
                        help="common-random-numbers seed for paired designs")
    parser.add_argument("--run-id", default=None,
                        help="override the minted run identifier")
    parser.add_argument("--allow-dirty", action="store_true",
                        default=os.environ.get("ALLOW_DIRTY", "") == "1",
                        help="permit a run from an uncommitted tree; recorded "
                             "in the manifest as provenance.allow_dirty")
    parser.add_argument("--note", default="", help="free text stored in the manifest")
    parser.add_argument("--workers", type=int, default=None, help="process pool size")
    args = parser.parse_args(argv)

    n = args.replicates if args.replicates is not None else (
        args.replicates_pos if args.replicates_pos is not None else replicates)
    spec = RunSpec(
        experiment=experiment,
        replicates=n,
        seed_base=args.seed_base if args.seed_base is not None else seed_base,
        crn_seed=args.crn_seed if args.crn_seed is not None else crn_seed,
        run_id=args.run_id or new_run_id(experiment),
        allow_dirty=args.allow_dirty,
        note=args.note or os.environ.get("RUN_NOTE", ""),
        workers=args.workers,
    )
    if spec.workers:
        os.environ["N_WORKERS"] = str(spec.workers)
    publish_spec(spec.run_id, seed_base=spec.seed_base, crn_seed=spec.crn_seed,
                 allow_dirty=spec.allow_dirty, note=spec.note)

    state = require_clean_tree(spec.allow_dirty, context=f"experiment '{experiment}'")
    print(f"[{experiment}] run_id={spec.run_id}")
    print(f"[{experiment}] commit={state.label} branch={state.branch}"
          + (f" tag={state.tag}" if state.tag else ""))
    print(f"[{experiment}] replicates={spec.replicates} seed_base={spec.seed_base}"
          + (f" crn_seed={spec.crn_seed}" if spec.crn_seed is not None else ""))
    return spec
