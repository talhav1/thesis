"""Lint `results/manifests/` against the provenance contract.

    python tools/manifest_lint.py            # report
    python tools/manifest_lint.py --strict   # exit 1 on any non-legacy violation

Checks, in the order a reader would care about them:

1. **run_id present** and used as the manifest's filename stem, so the manifest,
   the raw table and the summaries join on one key.
2. **pinned to a clean tree**, or a recorded `allow_dirty` waiver.
3. **no `seed_range`** -- the field was fictional in every legacy manifest
   (the code seeds with `default_rng([seed_base, replicate])`, not a contiguous
   range), so its presence is now an error rather than decoration.
4. **`replicates_unit` declared** whenever `n_replicates_completed` exceeds
   `n_replicates_requested`, which is the signature of a row count masquerading
   as a replicate count (`phase1_horizon`, `phase3b_screen`).
5. **artifacts are role-tagged** and the referenced files exist. Legacy manifests
   record absolute paths from a machine that no longer exists
   (`/home/claude/thesis-reliability/...`); existence is checked by falling back to
   the same path resolved relative to the repo root (matched on a recognised
   top-level directory such as `results/` or `configs/`) before it is reported
   missing. The path recorded in the manifest is never rewritten -- only how the
   linter *checks* it changes.
6. **Monte Carlo standard errors recorded** for at least one reported quantity.

The 13 manifests written before this contract existed are listed in `LEGACY`.
They are reported but do not fail the lint: rewriting them would fabricate
provenance they never had. They are superseded when their experiment is next
re-run from a clean tree.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO_ROOT / "results" / "manifests"

LEGACY = {
    "phase1_baseline", "phase1_horizon", "phase1_quadrature_audit",
    "phase1_utility_surface", "phase2_coverage_map", "phase2_sbc",
    "phase3_reference_audit", "phase3a_attribution", "phase3a_nooverlap",
    "phase3a_pilot", "phase3b_confirm", "phase3b_pseudotruth", "phase3b_screen",
}

# Top-level directories that mark the start of a repo-relative path inside an
# absolute path recorded from a machine that no longer exists.
_REPO_MARKERS = ("results", "configs", "src", "experiments", "docs")


def _is_foreign_absolute(rel: str) -> bool:
    """True for a path absolute on *some* OS, which may not be this one.

    Every legacy manifest records a POSIX path (`/home/claude/...`) minted on
    a Linux sandbox. `Path(rel).is_absolute()` is False for that string on
    Windows, so it can't be used to detect "this came from another machine."
    """
    return rel.startswith(("/", "\\")) or bool(re.match(r"^[A-Za-z]:[\\/]", rel))


def _resolve_artifact(rel: str) -> Path:
    """Resolve a manifest-recorded artifact path against this checkout.

    Repo-relative paths resolve directly. Foreign-absolute paths (every
    legacy manifest records `/home/claude/thesis-reliability/...`, a machine
    that no longer exists) are tried as-is first, then -- if that doesn't
    exist -- re-rooted at the first segment that matches a known top-level
    repo directory, split on `/` since that's what every manifest uses
    regardless of host OS. This only changes how existence is *checked*; the
    path stored in the manifest is untouched.
    """
    if not _is_foreign_absolute(rel):
        return REPO_ROOT / rel
    p = Path(rel)
    if p.exists():
        return p
    parts = rel.replace("\\", "/").strip("/").split("/")
    for marker in _REPO_MARKERS:
        if marker in parts:
            candidate = REPO_ROOT.joinpath(*parts[parts.index(marker):])
            if candidate.exists():
                return candidate
    return p


def lint_one(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        d = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        return [f"unreadable: {exc}"]

    schema = d.get("schema_version", 1)
    if schema < 2:
        problems.append("schema_version < 2 (written before the provenance contract)")

    if not d.get("run_id"):
        problems.append("no run_id")
    elif path.stem != d["run_id"] and path.stem != d.get("experiment"):
        problems.append(f"filename stem {path.stem!r} is neither run_id nor experiment")

    prov = d.get("provenance", {})
    commit = prov.get("git_commit") or d.get("git_commit", "")
    if commit.endswith("-dirty"):
        if prov.get("allow_dirty"):
            problems.append("dirty tree, waiver recorded (allow_dirty=true)")
        else:
            problems.append("dirty tree with NO recorded waiver -- result is unpinned")
    if commit.startswith("unknown"):
        problems.append("git commit unknown")

    if "seed_range" in d:
        problems.append("carries the fictional `seed_range` field")
    if d.get("seed_base") is None and "seed" not in json.dumps(d.get("config", {})).lower():
        problems.append("no seed_base recorded")
    if not (d.get("seed_rule") or prov.get("seed_rule")):
        problems.append("no seed_rule recorded")

    req, done = d.get("n_replicates_requested"), d.get("n_replicates_completed")
    if isinstance(req, int) and isinstance(done, int) and done > req:
        if d.get("replicates_unit", "replicates") == "replicates":
            problems.append(
                f"n_replicates_completed ({done}) > requested ({req}) but "
                "replicates_unit says 'replicates' -- likely a row count")

    arts = d.get("artifacts", [])
    if not arts:
        problems.append("no artifacts listed")
    for a in arts:
        if isinstance(a, str):
            problems.append(f"artifact not role-tagged: {a}")
            rel = a
        else:
            rel = a.get("path", "")
        if rel and not _resolve_artifact(rel).exists():
            problems.append(f"artifact missing on disk: {rel}")

    if not d.get("monte_carlo_se"):
        problems.append("no Monte Carlo standard errors recorded")

    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any non-legacy manifest has a problem")
    ap.add_argument("--include-legacy", action="store_true",
                    help="fail on legacy manifests too")
    args = ap.parse_args(argv)

    if not MANIFEST_DIR.exists():
        print("no results/manifests/ directory")
        return 0

    failures = 0
    for path in sorted(MANIFEST_DIR.glob("*.json")):
        problems = lint_one(path)
        legacy = path.stem in LEGACY and not args.include_legacy
        if not problems:
            print(f"OK      {path.name}")
            continue
        tag = "LEGACY " if legacy else "PROBLEM"
        print(f"{tag} {path.name}")
        for p in problems:
            print(f"        - {p}")
        if not legacy:
            failures += 1

    print(f"\n{failures} manifest(s) with problems outside the legacy set.")
    return 1 if (args.strict and failures) else 0


if __name__ == "__main__":
    sys.exit(main())
