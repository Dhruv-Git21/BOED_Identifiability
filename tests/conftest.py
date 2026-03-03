"""Conftest for pytest fixtures."""

import pytest
import numpy as np
from causal_boed.utils.rng import set_seed


@pytest.fixture(scope="session", autouse=True)
def setup_random_seed():
    """Set random seed for all tests."""
    set_seed(42)
    yield


@pytest.fixture
def rng():
    """Provide RNG for tests."""
    from causal_boed.utils.rng import get_rng
    return get_rng()
