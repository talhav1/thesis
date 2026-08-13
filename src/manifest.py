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
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO_ROOT / "results" / "manifests"


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        commit = out.stdout.strip() or "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


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


@dataclass
class RunManifest:
    experiment: str
    config: dict
    seed_base: int
    n_replicates_requested: int
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

    def finalise(self) -> dict:
        return {
            "experiment": self.experiment,
            "config_hash": config_hash(self.config),
            "config": self.config,
            "git_commit": git_commit(),
            "seed_base": self.seed_base,
            "seed_range": [self.seed_base, self.seed_base + self.n_replicates_requested - 1],
            "n_replicates_requested": self.n_replicates_requested,
            "n_replicates_completed": self.n_replicates_completed,
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
            "artifacts": [str(a) for a in self.artifacts],
            "wall_seconds": self.wall_seconds,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "notes": self.notes,
        }

    def write(self, name: str | None = None) -> Path:
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        payload = self.finalise()
        stem = name or f"{self.experiment}_{payload['config_hash']}"
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
