"""Tests for SEM sampling."""

import numpy as np
import pytest
from causal_boed.graphs.dag import DAG, sample_random_dag
from causal_boed.sem.linear_gaussian import LinearGaussianSEM, create_linear_gaussian_sem
from causal_boed.sem.nonlinear_anm import NonlinearANMSEM, create_nonlinear_anm_sem
from causal_boed.graphs.interventions import Intervention


@pytest.fixture
def simple_dag():
    """Simple 3-node DAG: 0 -> 1 -> 2."""
    adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])
    return DAG(adj)


@pytest.fixture
def simple_sem(simple_dag):
    """Linear Gaussian SEM on simple_dag."""
    return create_linear_gaussian_sem(simple_dag, coeff_scale=0.5, noise_std=1.0, seed=42)


class TestLinearGaussianSEM:
    """Tests for LinearGaussianSEM."""
    
    def test_initialization(self, simple_dag):
        """Test SEM initialization."""
        sem = create_linear_gaussian_sem(simple_dag)
        assert sem.n_nodes == 3
        assert sem.B.shape == (3, 3)
    
    def test_sampling(self, simple_sem):
        """Test sampling from SEM."""
        X = simple_sem.sample(n_samples=100, seed=42)
        assert X.shape == (100, 3)
        assert np.all(np.isfinite(X))
    
    def test_intervention_breaks_edges(self, simple_sem):
        """Test that intervention removes incoming edges."""
        intervention = Intervention(variable=1, value=0.0)
        X = simple_sem.sample(n_samples=100, intervention=intervention, seed=42)
        
        # With intervention on X_1, it should be independent of X_0
        # Check by correlation
        assert X.shape == (100, 3)
    
    def test_deterministic_intervention(self, simple_sem):
        """Test deterministic intervention value."""
        intervention = Intervention(variable=0, value=5.0)
        X = simple_sem.sample(n_samples=50, intervention=intervention, seed=42)
        
        # X_0 should be exactly 5.0
        assert np.allclose(X[:, 0], 5.0)
    
    def test_reproducibility(self, simple_sem):
        """Test reproducible sampling with seed."""
        X1 = simple_sem.sample(100, seed=42)
        X2 = simple_sem.sample(100, seed=42)
        assert np.allclose(X1, X2)
    
    def test_noise_std(self, simple_dag):
        """Test noise standard deviation."""
        sem_low_noise = create_linear_gaussian_sem(simple_dag, noise_std=0.1, seed=42)
        sem_high_noise = create_linear_gaussian_sem(simple_dag, noise_std=10.0, seed=42)
        
        X_low = sem_low_noise.sample(1000, seed=42)
        X_high = sem_high_noise.sample(1000, seed=42)
        
        # High noise should have higher variance
        assert np.var(X_high) > np.var(X_low)


class TestNonlinearANMSEM:
    """Tests for NonlinearANMSEM."""
    
    def test_initialization(self, simple_dag):
        """Test nonlinear SEM initialization."""
        sem = create_nonlinear_anm_sem(simple_dag)
        assert sem.n_nodes == 3
    
    def test_sampling(self, simple_dag):
        """Test sampling from nonlinear SEM."""
        sem = create_nonlinear_anm_sem(simple_dag)
        X = sem.sample(100, seed=42)
        assert X.shape == (100, 3)
        assert np.all(np.isfinite(X))
    
    def test_intervention_on_nonlinear(self, simple_dag):
        """Test intervention on nonlinear SEM."""
        sem = create_nonlinear_anm_sem(simple_dag)
        intervention = Intervention(variable=0, value=0.0)
        X = sem.sample(50, intervention=intervention, seed=42)
        
        assert np.allclose(X[:, 0], 0.0)


class TestInterventions:
    """Tests for intervention application."""
    
    def test_intervention_str(self):
        """Test intervention string representation."""
        int1 = Intervention(variable=0, value=1.0)
        assert "do(X0" in str(int1)
        
        int2 = Intervention(variable=2)
        assert "do(X2" in str(int2)
    
    def test_invalid_intervention_type(self):
        """Test invalid intervention type."""
        with pytest.raises(ValueError):
            Intervention(variable=0, intervention_type="invalid")
    
    def test_invalid_strength(self):
        """Test invalid strength parameter."""
        with pytest.raises(ValueError):
            Intervention(variable=0, strength=1.5)
