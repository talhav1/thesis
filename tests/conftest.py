import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.priors import rotem_prior, rotem_stimulus_grid  # noqa: E402


@pytest.fixture
def prior():
    return rotem_prior("well", "independent")


@pytest.fixture
def candidates():
    return rotem_stimulus_grid("well", n_points=60)


@pytest.fixture
def small_history():
    """A short history with an overlapping pattern, so the MLE is finite."""
    x = np.array([28.0, 32.0, 30.0, 31.0, 29.0, 33.0, 27.0, 30.5])
    y = np.array([0, 1, 1, 0, 0, 1, 0, 1])
    return x, y


@pytest.fixture
def theta_grid():
    mu = np.linspace(26.0, 34.0, 9)
    sigma = np.linspace(1.5, 5.0, 8)
    MU, SG = np.meshgrid(mu, sigma, indexing="ij")
    return MU.ravel(), SG.ravel()
