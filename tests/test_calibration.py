"""Required tests 8 and 9: SBC uniformity, non-adaptive and adaptive.

These are the empirical counterpart of the Phase 0 proposition.  If the
product likelihood were the wrong likelihood under an adaptive design, the
adaptive SBC would fail while the non-adaptive one passed.  Both must pass.

Kept small enough to run in the default suite; the Phase 2 experiment repeats
them at production replicate counts.
"""

import numpy as np
import pytest

from src.calibration import sbc_replicate, uniformity_report
from src.priors import rotem_prior
from src.simulator import ExperimentConfig

N_DRAWS = 200
TARGETS = ("mu", "q0.95")


def _config(policy, n_steps=8, n_particles=4000):
    prior = rotem_prior("well", "independent")
    return ExperimentConfig(
        policy=policy,
        prior=prior.spec.config(),
        curve={"family": "probit", "mu": 30.0, "sigma": 3.0},  # overridden by the draw
        calibration="well",
        n_steps=n_steps,
        slices=(n_steps,),
        n_particles=n_particles,
        targets=TARGETS,
        grid_points=60,
        reference_n_max=513,
    )


def _ranks(policy, seed, use_reference=True):
    cfg = _config(policy)
    rows = [sbc_replicate(cfg, r, seed, use_reference=use_reference)
            for r in range(N_DRAWS)]
    return rows


@pytest.mark.parametrize(
    "policy,label",
    [("uniform_grid", "non-adaptive"), ("entropy_vector", "adaptive")],
)
def test_reference_posterior_passes_sbc(policy, label):
    """The exact (grid) posterior must give uniform ranks under either design."""
    rows = _ranks(policy, seed=4242)

    # The reference must be trustworthy before its ranks mean anything.  A
    # handful of prior draws are extreme enough (sigma from 0.3 to 15 here)
    # that the refinement just misses the tolerance at this grid cap, so the
    # requirement is a high convergence rate plus a hard cap on how far any
    # single draw missed by -- not blanket success.
    rate = float(np.mean([r["ref_converged"] for r in rows]))
    worst = float(np.max([r["ref_max_delta"] for r in rows]))
    assert rate > 0.95, f"{label}: only {rate:.1%} of references converged"
    assert worst < 5e-3, f"{label}: worst reference delta {worst:.2e}"
    for t in TARGETS:
        rep = uniformity_report([r[f"{t}_rank_ref"] for r in rows])
        assert rep["ks_p"] > 0.01, (
            f"{label} / {t}: reference ranks non-uniform, KS p={rep['ks_p']:.4f}, "
            f"D={rep['ks_stat']:.4f}"
        )
        assert abs(rep["mean"] - 0.5) < 4 * rep["mean_se"]


def test_particle_posterior_sbc_is_close_to_uniform():
    """Rotem's particle posterior under the adaptive design.

    A failure here is *computational*, not a refutation of the likelihood
    factorisation -- which is exactly why the reference is tested separately
    and first.
    """
    rows = _ranks("entropy_vector", seed=909, use_reference=False)
    for t in TARGETS:
        rep = uniformity_report([r[f"{t}_rank_particle"] for r in rows])
        assert rep["ks_stat"] < 3 * rep["ks_band_95"], (
            f"{t}: particle ranks far from uniform, D={rep['ks_stat']:.4f}"
        )


def test_degenerate_paths_are_retained_in_sbc_output():
    """All-zero / all-one / no-overlap draws must appear, not vanish."""
    rows = _ranks("entropy_vector", seed=31337, use_reference=False)
    assert len(rows) == N_DRAWS
    flags = np.array([r["degenerate_all_equal"] or r["degenerate_no_overlap"]
                      for r in rows])
    assert flags.sum() > 0, "expected some degenerate paths at n=8"
    # and their rank statistics must still be present and finite
    for r in rows:
        assert np.isfinite(r["mu_rank_particle"])


def test_uniformity_report_detects_a_known_deviation():
    """The uniformity test must have power, or the SBC checks mean nothing."""
    rng = np.random.default_rng(0)
    good = uniformity_report(rng.uniform(size=2000))
    assert good["ks_p"] > 0.01
    biased = uniformity_report(rng.beta(1.15, 1.0, size=2000))
    assert biased["ks_p"] < 0.01
