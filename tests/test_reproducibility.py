"""Required test 10: deterministic reproduction of saved experiment configurations.

A result that cannot be regenerated from its manifest is not a result.  These
checks cover the three ways that could fail: the config not round-tripping, the
seeds not fully determining the run, and the common-random-number stream
leaking across policies.
"""

import json

import numpy as np

from src.manifest import RunManifest, config_hash, mc_se_mean, mc_se_proportion
from src.priors import rotem_prior
from src.simulator import ExperimentConfig, latent_uniforms, run_batch, run_experiment


def _config(policy="entropy_vector", n_steps=10):
    prior = rotem_prior("well", "independent")
    return ExperimentConfig(
        policy=policy,
        prior=prior.spec.config(),
        curve={"family": "probit", "mu": 30.0, "sigma": 3.0},
        calibration="well",
        n_steps=n_steps,
        slices=(n_steps,),
        n_particles=2500,
        grid_points=50,
    )


def test_config_round_trips_through_json():
    cfg = _config()
    payload = json.loads(json.dumps(cfg.as_dict()))
    restored = ExperimentConfig(**{
        **payload,
        "slices": tuple(payload["slices"]),
        "targets": tuple(payload["targets"]),
        "reference_at_slices": tuple(payload["reference_at_slices"]),
    })
    assert config_hash(restored.as_dict()) == config_hash(cfg.as_dict())


def test_same_seed_reproduces_bitwise(tmp_path):
    cfg = _config()
    a = run_experiment(cfg, replicate=3, seed_base=555)
    b = run_experiment(cfg, replicate=3, seed_base=555)
    assert np.array_equal(a["x"], b["x"])
    assert np.array_equal(a["y"], b["y"])
    ra, rb = a["slices"][10], b["slices"][10]
    for k, v in ra.items():
        if isinstance(v, float) and np.isfinite(v):
            assert v == rb[k], k
        elif isinstance(v, (bool, int, str)):
            assert v == rb[k], k


def test_different_seed_changes_the_run():
    cfg = _config()
    a = run_experiment(cfg, replicate=3, seed_base=555)
    b = run_experiment(cfg, replicate=3, seed_base=556)
    assert not np.array_equal(a["y"], b["y"])


def test_common_random_numbers_are_shared_across_policies():
    """The response latents must depend on the replicate, not on the policy.

    This is what makes policy comparisons paired; if it broke, every
    between-policy contrast would silently inflate its Monte Carlo error.
    """
    u1 = latent_uniforms(555, 7, 30)
    u2 = latent_uniforms(555, 7, 30)
    assert np.array_equal(u1, u2)
    assert not np.array_equal(u1, latent_uniforms(555, 8, 30))

    # two policies at the same replicate see identical latents
    seen = {}
    for pol in ("entropy_vector", "bruceton"):
        cfg = _config(pol)
        res = run_experiment(cfg, replicate=4, seed_base=555)
        seen[pol] = res
    # same latent stream implies identical y wherever the applied x coincide
    u = latent_uniforms(555, 4, 10)
    for pol, res in seen.items():
        from src.response_models import ProbitCurve

        p = ProbitCurve(30.0, 3.0).prob(res["x"])
        assert np.array_equal(res["y"].astype(bool), u < p)


def test_run_batch_records_failures_and_keeps_degenerate_paths():
    cfg = _config(n_steps=6)
    rows, failures, trajectories = run_batch(cfg, n_replicates=8, seed_base=777)
    assert len(trajectories) + len(failures) == 8
    assert len(rows) == len(trajectories) * len(cfg.slices)
    for row in rows:
        assert "degenerate_all_equal" in row
        assert "degenerate_no_overlap" in row
        assert np.isfinite(row["mu_bayes"])


def test_manifest_captures_provenance(tmp_path, monkeypatch):
    import src.manifest as m

    monkeypatch.setattr(m, "MANIFEST_DIR", tmp_path)
    cfg = _config()
    man = RunManifest(
        experiment="unit-test",
        config=cfg.as_dict(),
        seed_base=555,
        n_replicates_requested=8,
        n_replicates_completed=7,
        n_failures=1,
        tolerances={"reference_tol": cfg.reference_tol},
        monte_carlo_se={"coverage_q0.95": 0.01},
        degeneracy_counts={"no_overlap": 2},
    )
    path = man.write("unit-test")
    payload = json.loads(path.read_text())
    for key in ("config_hash", "git_commit", "run_id", "provenance", "seed_rule",
                "n_replicates_completed", "n_failures", "monte_carlo_se",
                "tolerances", "backend", "degeneracy_counts", "created_utc"):
        assert key in payload
    # `seed_range` was fictional in every legacy manifest (D20): the code seeds
    # with default_rng([seed_base, replicate]), so what is recorded now is the
    # authoritative seed_base plus the replicate index range.
    assert "seed_range" not in payload
    assert payload["seed_base"] == 555
    assert payload["replicate_index_range"] == [0, 7]
    assert payload["config_hash"] == config_hash(cfg.as_dict())


def test_monte_carlo_standard_errors():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2, 4000)
    assert abs(mc_se_mean(x) - 2 / np.sqrt(4000)) < 0.005
    assert abs(mc_se_proportion(500, 1000) - np.sqrt(0.25 / 1000)) < 1e-12
    assert np.isnan(mc_se_mean([1.0]))
