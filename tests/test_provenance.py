"""The provenance contract, enforced.

Every check here corresponds to a failure the project actually had (D19-D22).
A test that passes only because the contract is documented somewhere is not
worth writing; these assert the behaviour.
"""

from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.manifest import SCHEMA_VERSION, RunManifest, config_hash  # noqa: E402
from src.provenance import (  # noqa: E402
    SEED_RULE,
    DirtyTreeError,
    GitState,
    new_run_id,
    provenance_block,
    require_clean_tree,
)


# ---------------------------------------------------------------------------
# D19 - runs must not start from an unpinned tree
# ---------------------------------------------------------------------------


def test_dirty_tree_refuses_by_default(monkeypatch):
    import src.provenance as prov

    monkeypatch.setattr(prov, "git_state", lambda: GitState(
        commit="abc123", branch="main", dirty=True, dirty_files=["src/posterior_grid.py"]))
    with pytest.raises(DirtyTreeError) as exc:
        prov.require_clean_tree(allow_dirty=False)
    # the operator must be told *which* paths are dirty, not merely that some are
    assert "src/posterior_grid.py" in str(exc.value)


def test_dirty_tree_waiver_proceeds_and_is_recorded(monkeypatch):
    import src.provenance as prov

    monkeypatch.setattr(prov, "git_state", lambda: GitState(
        commit="abc123", branch="main", dirty=True, dirty_files=["src/x.py"]))
    state = prov.require_clean_tree(allow_dirty=True)
    assert state.dirty
    block = prov.provenance_block("run-1", allow_dirty=True)
    assert block["allow_dirty"] is True
    assert block["pinned"] is False
    assert block["git_commit"].endswith("-dirty")


def test_clean_tree_is_pinned(monkeypatch):
    import src.provenance as prov

    monkeypatch.setattr(prov, "git_state", lambda: GitState(
        commit="deadbeef", branch="main", dirty=False, dirty_files=[]))
    state = prov.require_clean_tree()
    assert not state.dirty
    assert prov.provenance_block("run-2")["pinned"] is True


# ---------------------------------------------------------------------------
# D20 - run identifiers, and the seed rule must describe the real code
# ---------------------------------------------------------------------------


def test_run_id_is_unique_and_well_formed(monkeypatch):
    monkeypatch.delenv("RUN_ID", raising=False)
    ids = {new_run_id("phase9_demo") for _ in range(50)}
    assert len(ids) == 50
    for rid in ids:
        assert re.fullmatch(r"phase9_demo_\d{8}T\d{6}Z_[0-9a-f]{6}", rid), rid


def test_run_id_env_override_groups_a_pipeline(monkeypatch):
    monkeypatch.setenv("RUN_ID", "pipeline-42")
    assert new_run_id("anything") == "pipeline-42"


def test_manifest_carries_run_id_and_no_seed_range(tmp_path, monkeypatch):
    import src.manifest as m

    monkeypatch.setattr(m, "MANIFEST_DIR", tmp_path)
    monkeypatch.delenv("RUN_ID", raising=False)
    man = RunManifest(experiment="unit", config={"a": 1}, seed_base=555,
                      n_replicates_requested=8, n_replicates_completed=8,
                      crn_seed=99, monte_carlo_se={"coverage": 0.01},
                      artifacts=["results/summaries/unit.csv"])
    payload = json.loads(man.write().read_text())

    assert payload["schema_version"] == SCHEMA_VERSION >= 2
    assert payload["run_id"]
    assert payload["seed_base"] == 555 and payload["crn_seed"] == 99
    assert payload["replicate_index_range"] == [0, 7]
    assert payload["config_hash"] == config_hash({"a": 1})
    # D20: the field was fictional and must not come back
    assert "seed_range" not in payload
    assert payload["seed_rule"] == SEED_RULE


def test_manifest_filename_stem_is_the_run_id(tmp_path, monkeypatch):
    import src.manifest as m

    monkeypatch.setattr(m, "MANIFEST_DIR", tmp_path)
    monkeypatch.delenv("RUN_ID", raising=False)
    man = RunManifest(experiment="unit", config={}, seed_base=1,
                      n_replicates_requested=1)
    path = man.write()
    assert path.stem == json.loads(path.read_text())["run_id"]


def test_seed_rule_matches_the_simulator():
    """The manifest's stated seeding must be the seeding that runs.

    D20: `seed_range` survived for thirteen manifests precisely because no test
    compared the recorded provenance against the code.
    """
    from src import simulator

    source = inspect.getsource(simulator)
    for stream in ("[seed_base, replicate]", "0xC0FFEE", "0xB0BB1E"):
        assert stream in source, f"{stream} no longer in simulator"
        assert stream.strip("[]") in SEED_RULE or stream in SEED_RULE, \
            f"SEED_RULE does not mention {stream}"
    assert "seed_range" not in SEED_RULE


# ---------------------------------------------------------------------------
# D21/D22 - artifacts and replicate units
# ---------------------------------------------------------------------------


def test_artifacts_are_role_tagged_and_repo_relative(tmp_path, monkeypatch):
    import src.manifest as m

    monkeypatch.setattr(m, "MANIFEST_DIR", tmp_path)
    man = RunManifest(
        experiment="unit", config={}, seed_base=1, n_replicates_requested=1,
        artifacts=[
            str(REPO_ROOT / "results" / "summaries" / "phase1_baseline_mse.csv"),
            "results/raw/x.parquet",
            ("figure", "results/figures/f.png"),
        ])
    arts = json.loads(man.write().read_text())["artifacts"]
    assert all(set(a) == {"role", "path"} for a in arts)
    assert [a["role"] for a in arts] == ["summary", "raw", "figure"]
    # D21: absolute paths from a vanished machine resolved against nothing
    assert not any(a["path"].startswith("/") for a in arts)


def test_replicates_unit_defaults_to_replicates_and_can_say_rows(tmp_path, monkeypatch):
    import src.manifest as m

    monkeypatch.setattr(m, "MANIFEST_DIR", tmp_path)
    default = RunManifest(experiment="u", config={}, seed_base=1,
                          n_replicates_requested=1).finalise()
    assert default["replicates_unit"] == "replicates"
    rows = RunManifest(experiment="u", config={}, seed_base=1,
                       n_replicates_requested=300, n_replicates_completed=900,
                       replicates_unit="rows").finalise()
    assert rows["replicates_unit"] == "rows"


# ---------------------------------------------------------------------------
# The linter must actually catch the legacy failures
# ---------------------------------------------------------------------------


def _lint(path):
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from manifest_lint import lint_one

    return lint_one(path)


def test_linter_flags_a_legacy_manifest(tmp_path):
    legacy = {
        "experiment": "old", "config_hash": "x", "config": {},
        "git_commit": "abc-dirty", "seed_base": 1,
        "seed_range": [1, 10],  # the fictional field
        "n_replicates_requested": 300, "n_replicates_completed": 900,
        "artifacts": ["/gone/results/summaries/old.csv"],
        "monte_carlo_se": {},
    }
    p = tmp_path / "old.json"
    p.write_text(json.dumps(legacy))
    problems = " | ".join(_lint(p))
    for expected in ("no run_id", "dirty tree", "seed_range", "row count",
                     "not role-tagged", "Monte Carlo"):
        assert expected in problems, f"linter missed: {expected}\n{problems}"


def test_linter_passes_a_conforming_manifest(tmp_path, monkeypatch):
    import src.manifest as m

    monkeypatch.setattr(m, "MANIFEST_DIR", tmp_path)
    monkeypatch.delenv("RUN_ID", raising=False)
    man = RunManifest(
        experiment="unit", config={}, seed_base=1, n_replicates_requested=4,
        n_replicates_completed=4, monte_carlo_se={"coverage": 0.01},
        allow_dirty=True,  # this tree is dirty during development
        artifacts=["results/summaries/phase1_baseline_mse.csv"])
    problems = [p for p in _lint(man.write())
                if "waiver recorded" not in p]  # the waiver is reported, not a failure
    assert problems == [], problems


# ---------------------------------------------------------------------------
# The config-driven entry point
# ---------------------------------------------------------------------------


def test_every_experiment_script_has_a_config():
    scripts = {p.stem for p in (REPO_ROOT / "experiments").glob("phase*.py")}
    configs = {p.stem for p in (REPO_ROOT / "configs" / "experiments").glob("*.json")}
    missing = scripts - configs
    assert not missing, f"experiments with no config: {sorted(missing)}"


def test_configs_name_a_real_experiment_and_flag_unresolved_replicates():
    for cfg_path in (REPO_ROOT / "configs" / "experiments").glob("*.json"):
        cfg = json.loads(cfg_path.read_text())
        script = REPO_ROOT / "experiments" / f"{cfg['experiment']}.py"
        assert script.exists(), f"{cfg_path.name} points at a missing script"
        prov = cfg.get("_provenance", {})
        if "replicates" not in cfg:
            # D22: a per-cell count that could not be recovered must say so
            assert "UNRESOLVED" in prov, f"{cfg_path.name} silently omits replicates"


def test_run_py_dry_run_resolves_a_config():
    out = subprocess.run(
        [sys.executable, "experiments/run.py",
         "configs/experiments/phase1_baseline.json", "--dry-run", "--replicates", "5"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "--replicates 5" in out.stdout
    assert "run_id" in out.stdout


# ---------------------------------------------------------------------------
# Regressions found by the first end-to-end run of this contract
# ---------------------------------------------------------------------------


def test_porcelain_paths_are_not_truncated(monkeypatch):
    """`git status --porcelain` uses two status columns, so a working-tree-only
    modification begins with a space. Stripping the whole output ate that space
    on the first line and reported `ocs/discrepancies.md`."""
    import src.provenance as prov

    def fake_git(*args, timeout=10, strip=True):
        if args[0] == "status":
            out = " M docs/discrepancies.md\nM  src/manifest.py\n?? tools/new.py\n"
            return out.strip() if strip else out
        return {"rev-parse": "abc123"}.get(args[0], "")

    monkeypatch.setattr(prov, "_git", fake_git)
    assert prov.git_state().dirty_files == [
        "docs/discrepancies.md", "src/manifest.py", "tools/new.py"]


def test_published_spec_reaches_the_manifest_without_call_site_changes(monkeypatch):
    """The provenance must not depend on editing every `write_manifest` call:
    the first attempt did, and silently missed all thirteen scripts."""
    import src.provenance as prov

    for key in ("RUN_ID", "RUN_SEED_BASE", "RUN_CRN_SEED", "ALLOW_DIRTY", "RUN_NOTE"):
        monkeypatch.delenv(key, raising=False)
    prov.publish_spec("run-xyz", seed_base=11, crn_seed=777001, allow_dirty=True,
                      note="pilot")
    spec = prov.active_spec()
    assert spec == {"run_id": "run-xyz", "seed_base": 11, "crn_seed": 777001,
                    "allow_dirty": True, "note": "pilot"}
