"""Machine-readable run manifests.

The plan requires every experimental result to record its configuration and
hash, the source commit, the seed range, the number of completed replicates,
Monte Carlo standard errors, the inference backend and tolerances, the failure
and exception counts, and the exact table or figure it produced.

`RunManifest` is the single object that carries all of that, and
`write_manifest` is the only sanctioned way to publish a result.  Nothing in
`results/` should exist without a sibling manifest.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.provenance import SEED_RULE, git_state, new_run_id, provenance_block

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO_ROOT / "results" / "manifests"

#: Bumped whenever the manifest payload changes shape.  Readers should branch
#: on this rather than probing for fields.  Schema 1 = the legacy manifests in
#: results/manifests written before run identifiers existed; they carry a
#: fictional `seed_range` and no `run_id`, and are grandfathered by
#: tools/manifest_lint.py rather than rewritten.
SCHEMA_VERSION = 2


def git_commit() -> str:
    """Commit label, with a `-dirty` suffix when the tree is not pinned."""
    return git_state().label


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, default=_json_default)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _normalise_artifacts(artifacts) -> list:
    """Artifacts as {role, path} records.

    Accepts a bare path (role inferred from `results/<subdir>/`), a
    (role, path) pair, or an explicit mapping.  Roles make the raw/summary/
    figure/manifest set joinable by `run_id` without guessing from filenames.
    """
    def _rel(path: str) -> str:
        """Repo-relative. Legacy manifests recorded absolute paths from a
        machine that no longer exists, which made them unresolvable."""
        try:
            return str(Path(path).resolve().relative_to(REPO_ROOT))
        except Exception:  # noqa: BLE001
            return str(path)

    out = []
    for a in artifacts:
        if isinstance(a, dict):
            out.append({"role": a.get("role", "unknown"), "path": _rel(a.get("path", ""))})
            continue
        if isinstance(a, (tuple, list)) and len(a) == 2:
            out.append({"role": str(a[0]), "path": _rel(a[1])})
            continue
        path = str(a)
        parts = Path(path).parts
        role = "unknown"
        for marker in ("raw", "summaries", "figures", "manifests", "tables"):
            if marker in parts:
                role = {"summaries": "summary", "figures": "figure",
                        "manifests": "manifest", "tables": "table"}.get(marker, marker)
                break
        out.append({"role": role, "path": _rel(path)})
    return out


@dataclass
class RunManifest:
    experiment: str
    config: dict
    seed_base: int
    n_replicates_requested: int
    run_id: str | None = None
    crn_seed: int | None = None
    allow_dirty: bool = False
    n_replicates_completed: int = 0
    n_failures: int = 0
    failures: list = field(default_factory=list)
    backend: dict = field(default_factory=dict)
    tolerances: dict = field(default_factory=dict)
    monte_carlo_se: dict = field(default_factory=dict)
    degeneracy_counts: dict = field(default_factory=dict)
    artifacts: list = field(default_factory=list)
    wall_seconds: float = 0.0
    notes: str = ""
    #: `n_replicates_completed` means replicates by default.  Two legacy
    #: manifests (phase1_horizon, phase3b_screen) recorded output *rows*
    #: instead; anything that counts rows must say so here.
    replicates_unit: str = "replicates"

    def finalise(self) -> dict:
        run_id = self.run_id or new_run_id(self.experiment)
        prov = provenance_block(run_id, allow_dirty=self.allow_dirty)
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "experiment": self.experiment,
            "config_hash": config_hash(self.config),
            "config": self.config,
            "provenance": prov,
            "git_commit": prov["git_commit"],
            "seed_base": self.seed_base,
            "crn_seed": self.crn_seed,
            "seed_rule": SEED_RULE,
            "replicate_index_range": [0, max(self.n_replicates_requested - 1, 0)],
            "n_replicates_requested": self.n_replicates_requested,
            "n_replicates_completed": self.n_replicates_completed,
            "replicates_unit": self.replicates_unit,
            "n_failures": self.n_failures,
            "failures": self.failures[:50],
            "backend": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                **self.backend,
            },
            "tolerances": self.tolerances,
            "monte_carlo_se": self.monte_carlo_se,
            "degeneracy_counts": self.degeneracy_counts,
            "artifacts": _normalise_artifacts(self.artifacts),
            "wall_seconds": self.wall_seconds,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "notes": self.notes,
        }

    def write(self, name: str | None = None) -> Path:
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        payload = self.finalise()
        stem = name or payload["run_id"]
        path = MANIFEST_DIR / f"{stem}.json"
        path.write_text(json.dumps(payload, indent=2, default=_json_default))
        return path


# --------------------------------------------------------------------------
# Monte Carlo uncertainty
# --------------------------------------------------------------------------


def mc_se_mean(values) -> float:
    """Standard error of a Monte Carlo mean."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float("nan")
    return float(np.std(v, ddof=1) / np.sqrt(len(v)))


def mc_se_proportion(successes: int, n: int) -> float:
    """Standard error of a Monte Carlo proportion (e.g. coverage)."""
    if n <= 0:
        return float("nan")
    p = successes / n
    return float(np.sqrt(max(p * (1 - p), 0.0) / n))


def mc_se_mse(errors) -> float:
    """Standard error of an MSE estimate = SE of the mean of squared errors."""
    e = np.asarray(errors, dtype=float)
    e = e[np.isfinite(e)]
    if len(e) < 2:
        return float("nan")
    return mc_se_mean(e**2)


def replicates_for_se(target_se: float, p: float = 0.5) -> int:
    """Replicates needed for a proportion to reach `target_se` (worst case p=0.5)."""
    return int(np.ceil(p * (1 - p) / target_se**2))
