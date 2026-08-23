"""Run provenance: git state, clean-tree enforcement, and run identifiers.

Motivation
----------
Every manifest written before this module records a git commit ending in
``-dirty``, so no result in the project is pinned to a committed state of the
source.  That is not a cosmetic gap: the bisection settings in
``GridPosterior._invert_cdf`` changed inside the same commit that contains the
Phase 3A outputs, and because the tree was dirty the manifest cannot establish
which settings Phase 3A actually used.  Interval widths across Phase 3A and 3B
are consequently not comparable (``docs/discrepancies.md`` D-series).

This module makes that failure mode structural rather than a matter of
discipline:

* :func:`require_clean_tree` refuses to start a run from a dirty working tree
  unless the operator passes ``--allow-dirty`` explicitly, and the waiver is
  recorded in the manifest.
* :func:`new_run_id` mints one identifier per run, used as the filename stem
  for the raw table, the summary tables and the manifest, so the three can be
  joined without inference.
* :func:`seed_rule` states the seeding scheme as executed, replacing the
  ``seed_range`` field, which was fictional in every legacy manifest.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Seeding scheme as implemented in :mod:`src.simulator`.  Recorded verbatim in
#: every manifest.  If the simulator's seeding changes, change this string in
#: the same commit; ``tests/test_provenance.py`` asserts the two agree.
SEED_RULE = (
    "numpy.random.default_rng([seed_base, replicate]) for the inference stream; "
    "default_rng([crn_seed, replicate, 0xC0FFEE]) for latent uniforms; "
    "default_rng([crn_seed, replicate, 0xB0BB1E]) for the Bruceton start. "
    "Replicates are indexed 0..n_replicates_requested-1. "
    "seed_base and crn_seed are authoritative; there is no contiguous seed range."
)


class DirtyTreeError(RuntimeError):
    """Raised when a run is attempted from an uncommitted working tree."""


@dataclass
class GitState:
    commit: str
    branch: str
    dirty: bool
    dirty_files: list = field(default_factory=list)
    tag: str | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["dirty_files"] = self.dirty_files[:100]
        d["dirty_file_count"] = len(self.dirty_files)
        return d

    @property
    def label(self) -> str:
        return self.commit + ("-dirty" if self.dirty else "")


def _git(*args: str, timeout: int = 10, strip: bool = True) -> str:
    out = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return out.stdout.strip() if strip else out.stdout


def git_state() -> GitState:
    """Current commit, branch, tag and the list of uncommitted paths."""
    try:
        commit = _git("rev-parse", "HEAD") or "unknown"
        branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
        # NOT stripped: porcelain status codes are two columns, and a
        # working-tree-only modification starts with a space (" M path").
        # Stripping the whole output eats that space on the first line and
        # truncates the path by one character.
        porcelain = _git("status", "--porcelain", strip=False)
        dirty_files = [ln[3:].strip() for ln in porcelain.splitlines() if ln.strip()]
        tag = _git("describe", "--tags", "--exact-match") or None
        return GitState(commit=commit, branch=branch, dirty=bool(dirty_files),
                        dirty_files=dirty_files, tag=tag)
    except Exception as exc:  # noqa: BLE001 - provenance must never be silent
        return GitState(commit=f"unknown ({type(exc).__name__})", branch="unknown",
                        dirty=True, dirty_files=["<git unavailable>"])


def require_clean_tree(allow_dirty: bool = False, *, context: str = "run") -> GitState:
    """Return the git state, refusing to proceed from a dirty tree.

    Passing ``allow_dirty=True`` (from ``--allow-dirty`` on the command line, or
    ``ALLOW_DIRTY=1`` in the environment) downgrades the refusal to a warning.
    The waiver is recorded in the manifest as ``provenance.allow_dirty``, so a
    result produced from an unpinned tree is identifiable after the fact.
    """
    state = git_state()
    if not state.dirty:
        publish_git_state(state)
        return state
    listing = "\n".join(f"    {p}" for p in state.dirty_files[:20])
    more = "" if len(state.dirty_files) <= 20 else \
        f"\n    ... and {len(state.dirty_files) - 20} more"
    message = (
        f"Refusing to start {context}: the working tree has "
        f"{len(state.dirty_files)} uncommitted path(s).\n{listing}{more}\n\n"
        "A result produced from a dirty tree cannot be tied to a state of the "
        "source, which has already cost this project one non-comparison "
        "(docs/discrepancies.md, Phase 3A vs 3B interval widths).\n"
        "Commit the tree, or re-run with --allow-dirty to record the waiver."
    )
    if not allow_dirty:
        raise DirtyTreeError(message)
    print("WARNING: " + message.replace("Refusing to start", "Starting"), flush=True)
    publish_git_state(state)
    return state


def new_run_id(experiment: str, *, when: float | None = None) -> str:
    """Mint a run identifier: ``{experiment}_{YYYYmmddTHHMMSSZ}_{token}``.

    ``RUN_ID`` in the environment overrides, so a multi-script pipeline can
    group its stages under one identifier.
    """
    override = os.environ.get("RUN_ID", "").strip()
    if override:
        return override
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(when))
    return f"{experiment}_{stamp}_{uuid.uuid4().hex[:6]}"


def publish_spec(run_id: str, *, seed_base: int | None = None,
                 crn_seed: int | None = None, allow_dirty: bool = False,
                 note: str = "") -> None:
    """Publish the active run's provenance to the environment.

    `write_manifest` reads these as defaults, so every experiment script gets
    correct provenance without editing its `write_manifest` call.  An explicit
    keyword at the call site still wins.
    """
    os.environ["RUN_ID"] = run_id
    os.environ["ALLOW_DIRTY"] = "1" if allow_dirty else "0"
    if seed_base is not None:
        os.environ["RUN_SEED_BASE"] = str(seed_base)
    if crn_seed is not None:
        os.environ["RUN_CRN_SEED"] = str(crn_seed)
    if note:
        os.environ["RUN_NOTE"] = note


def active_spec() -> dict:
    """The published spec, for `write_manifest` defaults."""
    def _int(key):
        v = os.environ.get(key, "").strip()
        return int(v) if v.lstrip("-").isdigit() else None
    return {
        "run_id": os.environ.get("RUN_ID") or None,
        "seed_base": _int("RUN_SEED_BASE"),
        "crn_seed": _int("RUN_CRN_SEED"),
        "allow_dirty": os.environ.get("ALLOW_DIRTY", "") == "1",
        "note": os.environ.get("RUN_NOTE", ""),
    }


#: Environment key carrying the launch-time git state, JSON-encoded.
GIT_STATE_ENV = "RUN_GIT_STATE"


def publish_git_state(state: GitState) -> None:
    """Freeze the launch-time git state for `provenance_block` to read back.

    Published as JSON in the environment rather than in a module global so that
    it survives the `spawn` start method, where each worker re-imports the
    module into a fresh interpreter.
    """
    os.environ[GIT_STATE_ENV] = json.dumps(state.as_dict())


def published_git_state() -> GitState | None:
    """The frozen launch-time state, or None if nothing was published."""
    raw = os.environ.get(GIT_STATE_ENV, "").strip()
    if not raw:
        return None
    try:
        d = json.loads(raw)
        return GitState(commit=d["commit"], branch=d["branch"], dirty=d["dirty"],
                        dirty_files=d.get("dirty_files", []), tag=d.get("tag"))
    except Exception:  # noqa: BLE001 - a corrupt hand-off must not fake provenance
        return None


def provenance_block(run_id: str, allow_dirty: bool = False) -> dict:
    """The provenance payload embedded in every manifest.

    The state is the one captured by `require_clean_tree` at launch, **not** a
    fresh query. Querying git here instead is self-defeating: `results/` is
    tracked, so by the time a run reaches this function it has written its own
    outputs into the working tree and dirtied it. Every manifest then recorded
    `-dirty` and `pinned: false`, and dropped the tag, no matter how clean the
    tree was when the run started -- which is precisely the failure this module
    was written to prevent, and the likely reason every legacy manifest carries
    a `-dirty` commit. A run's provenance is the state of the code that
    produced the result; the artefacts it goes on to write are outputs, not
    inputs, and must not retroactively unpin it.

    A live query remains the fallback for a caller that never passed through
    `require_clean_tree`, where nothing was frozen and the honest answer is
    whatever git says now.
    """
    state = published_git_state() or git_state()
    return {
        "run_id": run_id,
        "git": state.as_dict(),
        "git_commit": state.label,  # legacy field name, kept for old readers
        "allow_dirty": bool(allow_dirty),
        "pinned": (not state.dirty),
        "seed_rule": SEED_RULE,
        "invoked_by": os.environ.get("INVOKED_BY", "direct"),
        "config_path": os.environ.get("CONFIG_PATH") or None,
    }
