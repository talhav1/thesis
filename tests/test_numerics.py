"""Numerical-fidelity checks for the optimisations in the hot path.

Every performance shortcut in this codebase is required to be *exactly*
equivalent to the formulation it replaces, or to have its error bounded here.
"""

import numpy as np
import pytest

from src.policies import EntropyScalarPolicy, EntropyVectorPolicy, PolicyState
from src.priors import rotem_prior
from src.response_models import binary_entropy, derived_values, probit_prob_matrix
from src.rotem_particles import ParticlePosterior
from src.utilities import (
    mutual_information_scalar,
    mutual_information_scalar_literal,
    mutual_information_vector,
    mutual_information_vector_T,
    response_probability_matrix,
    response_probability_matrix_T,
)


@pytest.fixture
def state():
    rng = np.random.default_rng(7)
    prior = rotem_prior("well", "independent")
    post = ParticlePosterior(prior, 6000, rng)
    cand = np.linspace(16, 44, 45)
    st = PolicyState(candidates=cand, posterior=post)
    for x, y in [(28, 1), (31, 0), (29, 1), (33, 1), (27, 0), (30, 0)]:
        post.update(x, y)
        st.x_hist.append(x)
        st.y_hist.append(y)
    return st


def test_scalar_mi_matches_literal_formulae(state):
    """The fast quadrature must equal a direct transcription of section 5.3.

    The fast path uses the identity P_{r,x} = A_r / sum_{j in S_r} w_j, which
    holds because the normalisers of pi^(1), pi^(0) cancel against Phibar and
    1 - Phibar.  The literal version computes pi^(1), pi^(0) and Phibar
    separately and never uses that cancellation.
    """
    P = state.probs()
    for quantity in ("mu", "sigma", "q0.95"):
        theta = derived_values(state.posterior.mu, state.posterior.sigma, quantity)
        fast = mutual_information_scalar(theta, state.posterior.w, P).mi
        literal = mutual_information_scalar_literal(theta, state.posterior.w, P)
        assert np.max(np.abs(fast - literal)) < 1e-12, quantity
        assert int(np.argmax(fast)) == int(np.argmax(literal)), quantity


def test_transposed_layout_matches_row_major(state):
    """(L, N) and (N, L) formulations of the vector MI must agree."""
    P = state.probs()
    P_T = state.probs_T()
    a = mutual_information_vector(state.posterior.w, P)
    b = mutual_information_vector_T(state.posterior.w, P_T)
    assert np.max(np.abs(a - b)) < 1e-13


def test_generation_cache_does_not_change_results(state):
    """Cached policy scores equal uncached recomputation, bit for bit."""
    P = state.probs()
    mi_uncached = mutual_information_vector(state.posterior.w, P)
    mi_cached = EntropyVectorPolicy().scores(state)
    assert np.max(np.abs(mi_uncached - mi_cached)) < 1e-13

    for quantity in ("mu", "q0.95"):
        theta = derived_values(state.posterior.mu, state.posterior.sigma, quantity)
        # Compare like with like on both conventions.  The Phase 3 policy default
        # clips the estimator at zero (protocol section 8, D14); the unclipped
        # path reproduces Rotem's original convention and must still match.
        for clip in (False, True):
            ref = mutual_information_scalar(theta, state.posterior.w, P,
                                            clip_negative=clip).mi
            cached = EntropyScalarPolicy(quantity, clip_negative=clip)._scores(state).mi
            assert np.max(np.abs(ref - cached)) == 0.0, (quantity, clip)


def test_mi_clipping_records_the_unclipped_error(state):
    """Clipping must never hide the numerical error it is masking."""
    theta = derived_values(state.posterior.mu, state.posterior.sigma, "sigma")
    P = state.probs()
    raw = mutual_information_scalar(theta, state.posterior.w, P, clip_negative=False)
    clipped = mutual_information_scalar(theta, state.posterior.w, P, clip_negative=True)
    assert clipped.mi.min() >= 0.0
    assert clipped.min_unclipped == raw.min_unclipped
    assert clipped.n_clipped == int(np.sum(raw.mi < 0))
    # clipping changes only the negative entries
    keep = raw.mi >= 0
    assert np.array_equal(raw.mi[keep], clipped.mi[keep])


def test_cache_is_invalidated_on_rejuvenation(state):
    """A stale probability matrix after rejuvenation would be silently wrong."""
    before = state.probs_T()
    gen_before = state.posterior.generation
    state.posterior._rejuvenate()
    after = state.probs_T()
    assert state.posterior.generation == gen_before + 1
    assert after.shape == before.shape
    assert not np.array_equal(after, before)
    expected = response_probability_matrix_T(
        state.candidates, state.posterior.mu, state.posterior.sigma
    )
    assert np.array_equal(after, expected)


def test_float32_matrix_matches_float64_to_documented_tolerance(state):
    """Records what a float32 probability matrix would have cost in accuracy.

    float64 is used in production; this pins the reason.  Note the entropy
    matrix must be built from the float64 values: in float32 the clip bound
    1 - 1e-12 rounds to exactly 1, and log1p(-1) = -inf poisons the result.
    """
    P64 = probit_prob_matrix(state.candidates, state.posterior.mu,
                             state.posterior.sigma, dtype=np.float64)
    P32 = probit_prob_matrix(state.candidates, state.posterior.mu,
                             state.posterior.sigma, dtype=np.float32)
    assert np.max(np.abs(P64 - P32)) < 1e-6

    mi64 = mutual_information_vector(state.posterior.w, P64)
    mi32 = mutual_information_vector(state.posterior.w, P32)
    assert np.max(np.abs(mi64 - mi32)) < 1e-6
    # and the float32 entropy matrix is genuinely unusable without promotion
    assert np.isfinite(binary_entropy(P32)).all()


def test_binary_entropy_endpoints_and_symmetry():
    assert binary_entropy(np.array([0.0]))[0] < 1e-10
    assert binary_entropy(np.array([1.0]))[0] < 1e-10
    assert abs(binary_entropy(np.array([0.5]))[0] - np.log(2)) < 1e-12
    p = np.linspace(0.001, 0.999, 51)
    assert np.max(np.abs(binary_entropy(p) - binary_entropy(1 - p))) < 1e-12


def test_quadrature_density_integral_is_near_one(state):
    """The quadrature's own diagnostic must behave on a healthy posterior."""
    theta = derived_values(state.posterior.mu, state.posterior.sigma, "q0.95")
    res = mutual_information_scalar(theta, state.posterior.w, state.probs())
    assert 0.8 < res.density_integral < 1.02
    assert res.n_empty_windows == 0
    assert res.n_distinct_nodes > 50
