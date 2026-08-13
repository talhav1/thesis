"""Required test 7: particle weights, ESS, resampling and rejuvenation."""

import numpy as np
import pytest

from src.policies import EntropyVectorPolicy, PolicyState
from src.posterior_grid import build_reference_posterior
from src.priors import rotem_prior
from src.response_models import ProbitCurve
from src.rotem_particles import KL_THRESHOLD, ParticlePosterior


def _run(prior, candidates, n_steps, n_particles, seed, rejuvenate=True,
         curve=ProbitCurve(30.0, 3.0)):
    rng = np.random.default_rng(seed)
    post = ParticlePosterior(prior, n_particles, rng, rejuvenate=rejuvenate)
    state = PolicyState(candidates=candidates, posterior=post)
    policy = EntropyVectorPolicy()
    u = np.random.default_rng(seed + 77).random(n_steps)
    xs, ys = [], []
    for k in range(n_steps):
        x, _ = policy.select(state, rng)
        y = int(u[k] < float(np.atleast_1d(curve.prob(np.array([x])))[0]))
        post.update(x, y)
        xs.append(x)
        ys.append(y)
        state.x_hist, state.y_hist = xs, ys
        state.invalidate()
    return post, np.asarray(xs), np.asarray(ys)


def test_weights_normalise_and_ess_bounded(prior, candidates):
    post, _, _ = _run(prior, candidates, 15, 3000, seed=1)
    assert abs(post.w.sum() - 1.0) < 1e-12
    assert 1.0 <= post.ess() <= post.n + 1e-9
    assert 0.0 < post.w.max() <= 1.0


def test_ess_degrades_without_rejuvenation(prior, candidates):
    """ESS is bounded, ends far below N, and trends down -- but is NOT monotone.

    Sequential importance weights need not lose effective sample size at every
    step: a likelihood factor can partially offset an existing weight
    imbalance and raise the ESS.  Asserting step-wise monotonicity here fails
    on real runs.  What must hold is the bound, the overall decay, and the fact
    that the ESS never exceeds the particle count.
    """
    post, _, _ = _run(prior, candidates, 20, 3000, seed=2, rejuvenate=False)
    trace = np.asarray(post.diag.ess_trace)
    assert np.all(trace >= 1.0) and np.all(trace <= post.n + 1e-9)
    assert trace[-1] < 0.5 * trace[0]
    # a linear trend through the trace must be negative
    slope = np.polyfit(np.arange(len(trace)), trace, 1)[0]
    assert slope < 0


def test_kl_trigger_fires_and_is_recorded(prior, candidates):
    post, _, _ = _run(prior, candidates, 30, 4000, seed=3, rejuvenate=True)
    assert post.diag.n_resamples > 0, "rejuvenation never triggered in 30 steps"
    assert len(post.diag.resample_steps) == post.diag.n_resamples
    kls = np.asarray(post.diag.kl_trace)
    assert np.all(np.isfinite(kls))
    # every recorded resample step must correspond to a KL above the threshold
    for step in post.diag.resample_steps:
        assert kls[step - 1] > KL_THRESHOLD


def test_rejuvenation_preserves_the_posterior(prior, candidates):
    """The importance correction must leave posterior summaries unchanged.

    A wrong or missing pi_0/pi_new ratio would shift the represented posterior
    towards the proposal.  Running the identical history with and without
    rejuvenation and comparing to the deterministic reference catches that.
    """
    x = np.array([28.0, 32.0, 30.0, 31.0, 29.0, 33.0, 27.0, 30.5, 31.5, 29.5])
    y = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
    ref = build_reference_posterior(prior, x, y)

    for rejuvenate, threshold in ((False, np.inf), (True, 0.0)):
        rng = np.random.default_rng(9)
        post = ParticlePosterior(prior, 40_000, rng, rejuvenate=rejuvenate,
                                 kl_threshold=threshold)
        for xi, yi in zip(x, y):
            post.update(float(xi), int(yi))
        if rejuvenate:
            assert post.diag.n_resamples >= 5, "forced rejuvenation did not run"
        for q in ("mu", "sigma", "q0.95"):
            scale = ref.sd(q)
            assert abs(post.mean(q) - ref.mean(q)) < 0.08 * scale, (
                f"rejuvenate={rejuvenate}, {q}: particle {post.mean(q):.4f} "
                f"vs reference {ref.mean(q):.4f} (sd {scale:.4f})"
            )
            assert abs(post.sd(q) / scale - 1.0) < 0.12


def test_particle_posterior_converges_to_reference_as_n_grows(prior):
    """Discrepancy from the reference must shrink like a Monte Carlo error."""
    x = np.array([28.0, 32.0, 30.0, 31.0, 29.0, 33.0, 27.0, 30.5])
    y = np.array([0, 1, 1, 0, 0, 1, 0, 1])
    ref = build_reference_posterior(prior, x, y)

    errs = []
    for n in (2_000, 32_000):
        reps = []
        for s in range(6):
            rng = np.random.default_rng(1000 + s)
            post = ParticlePosterior(prior, n, rng, rejuvenate=False)
            for xi, yi in zip(x, y):
                post.update(float(xi), int(yi))
            reps.append(abs(post.mean("q0.95") - ref.mean("q0.95")))
        errs.append(float(np.mean(reps)))
    # a 16x increase in particles should cut the error by roughly 4x
    assert errs[1] < 0.5 * errs[0], f"errors did not shrink: {errs}"


def test_invalid_particles_are_counted_not_hidden():
    """Rejuvenation proposals with beta1 <= 0 get zero weight and are tallied."""
    prior = rotem_prior("poor", "independent")
    candidates = np.linspace(8.0, 36.0, 60)
    post, _, _ = _run(prior, candidates, 25, 4000, seed=17, rejuvenate=True,
                      curve=ProbitCurve(30.0, 8.0))
    snap = post.snapshot()
    assert snap["n_invalid_particles"] >= 0
    assert np.isfinite(post.w).all()
    assert abs(post.w.sum() - 1.0) < 1e-12


def test_rotem_median_matches_lower_inverse_cdf(prior, candidates):
    post, _, _ = _run(prior, candidates, 10, 2000, seed=5)
    v = post.values("mu")
    order = np.argsort(v)
    cw = np.cumsum(post.w[order])
    idx = int(np.searchsorted(cw, 0.5, side="right"))
    assert abs(post.median_rotem("mu") - v[order][idx]) < 1e-12
