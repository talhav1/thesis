"""Required tests 3 and 4: the two-quantile reparameterisation.

Proposition (Phase 0).  For p1 != p2 the map (mu, sigma) -> (q_p1, q_p2) is a
bijection, with inverse

    sigma = (q_p2 - q_p1) / (z_p2 - z_p1),    mu = q_p1 - sigma z_p1.

Mutual information is invariant under bijective reparameterisation, so

    I((q_p1, q_p2); Y | x, H) = I((mu, sigma); Y | x, H)

for *every* distinct pair, and no larger collection of quantiles can carry
more.  "All quantiles" is redundant once two are in hand.
"""

import numpy as np
import pytest
from scipy.stats import norm

from src.response_models import (
    derived_values,
    params_from_two_quantiles,
    quantile_from_params,
)
from src.utilities import (
    mutual_information_scalar,
    mutual_information_vector,
    response_probability_matrix,
)

PAIRS = [(0.1, 0.9), (0.5, 0.95), (0.01, 0.99), (0.3, 0.7), (0.5, 0.99)]


@pytest.mark.parametrize("p1,p2", PAIRS)
def test_quantile_roundtrip(p1, p2):
    rng = np.random.default_rng(3)
    mu = rng.uniform(10, 50, 2000)
    sigma = rng.uniform(0.2, 12.0, 2000)
    q1 = quantile_from_params(mu, sigma, p1)
    q2 = quantile_from_params(mu, sigma, p2)
    mu_b, sigma_b = params_from_two_quantiles(q1, q2, p1, p2)
    assert np.max(np.abs(mu_b - mu)) < 1e-9
    assert np.max(np.abs(sigma_b - sigma)) < 1e-10


def test_equal_quantiles_rejected():
    with pytest.raises(ValueError):
        params_from_two_quantiles(1.0, 2.0, 0.5, 0.5)


def _particles(seed=5, n=4000):
    rng = np.random.default_rng(seed)
    mu = rng.normal(30, 4, n)
    sigma = np.exp(rng.normal(np.log(3), 0.4, n))
    w = rng.random(n) ** 2
    return mu, sigma, w / w.sum()


@pytest.mark.parametrize("p1,p2", PAIRS)
def test_vector_mi_invariant_under_two_quantile_reparameterization(p1, p2):
    """I((q_p1,q_p2); Y) == I((mu,sigma); Y), for any distinct p1, p2."""
    mu, sigma, w = _particles()
    x = np.linspace(20, 42, 40)

    mi_direct = mutual_information_vector(w, response_probability_matrix(x, mu, sigma))

    # go out to quantile coordinates and back through the general machinery
    q1 = quantile_from_params(mu, sigma, p1)
    q2 = quantile_from_params(mu, sigma, p2)
    mu_r, sigma_r = params_from_two_quantiles(q1, q2, p1, p2)
    mi_reparam = mutual_information_vector(
        w, response_probability_matrix(x, mu_r, sigma_r)
    )

    assert np.max(np.abs(mi_direct - mi_reparam)) < 1e-10


def test_more_quantiles_add_no_information():
    """A 12-quantile coordinate system gives exactly the two-quantile answer."""
    mu, sigma, w = _particles()
    x = np.linspace(20, 42, 40)
    mi_pair = mutual_information_vector(w, response_probability_matrix(x, mu, sigma))

    ps = np.linspace(0.02, 0.98, 12)
    Q = np.column_stack([quantile_from_params(mu, sigma, p) for p in ps])
    # any two columns reconstruct the whole parameter vector
    mu_r, sigma_r = params_from_two_quantiles(Q[:, 0], Q[:, -1], ps[0], ps[-1])
    mi_many = mutual_information_vector(
        w, response_probability_matrix(x, mu_r, sigma_r)
    )
    assert np.max(np.abs(mi_pair - mi_many)) < 1e-10


def test_single_quantile_carries_strictly_less_information():
    """Sanity that the scalar estimator is not silently returning the vector MI.

    One quantile is a single function of (mu, sigma) and cannot determine it,
    so its mutual information with the next response must be strictly smaller
    at stimuli where the pair is informative.
    """
    mu, sigma, w = _particles()
    x = np.linspace(20, 42, 40)
    P = response_probability_matrix(x, mu, sigma)
    mi_vec = mutual_information_vector(w, P)
    mi_scalar = mutual_information_scalar(
        derived_values(mu, sigma, "q0.95"), w, P
    ).mi
    assert np.all(mi_scalar <= mi_vec + 1e-9)
    assert np.max(mi_vec - mi_scalar) > 1e-3
