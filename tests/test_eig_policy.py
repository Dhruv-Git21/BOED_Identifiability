"""Tests for EIG and intervention policy."""

import numpy as np
import pytest
from causal_boed.graphs.dag import DAG, sample_random_dag
from causal_boed.graphs.interventions import Intervention
from causal_boed.inference.posterior import ParticlePosterior
from causal_boed.design.eig import estimate_eig_via_edge_uncertainty, _bernoulli_entropy
from causal_boed.design.policy_greedy import GreedyEIGPolicy, RandomPolicy, OraclePolicy
from causal_boed.sem.linear_gaussian import create_linear_gaussian_sem


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
def posterior_particles(simple_dag):
    """Create a posterior with particles."""
    dags = [
        simple_dag,
        sample_random_dag(3, expected_degree=1.0, seed=100),
        sample_random_dag(3, expected_degree=1.0, seed=101),
    ]
    log_weights = np.array([0.5, 0.3, 0.2])
    return ParticlePosterior(dags, log_weights)


class TestBernoulliEntropy:
    """Tests for Bernoulli entropy helper."""
    
    def test_entropy_bounds(self):
        """Test entropy is in [0, log(2)]."""
        for p in np.linspace(0, 1, 11):
            h = _bernoulli_entropy(p)
            assert 0 <= h <= np.log(2) + 1e-6
    
    def test_entropy_maximum(self):
        """Test entropy is max at p=0.5."""
        h_max = _bernoulli_entropy(0.5)
        h_edges = _bernoulli_entropy(0.1)
        assert h_max > h_edges
    
    def test_entropy_zero_at_extremes(self):
        """Test entropy is 0 at p=0 and p=1."""
        assert _bernoulli_entropy(0.0) == 0.0
        assert _bernoulli_entropy(1.0) == 0.0


class TestEIGEstimation:
    """Tests for EIG estimation."""
    
    def test_eig_edge_uncertainty_nonnegative(self, posterior_particles):
        """Test EIG via edge uncertainty is non-negative."""
        for var in range(3):
            intervention = Intervention(variable=var)
            eig = estimate_eig_via_edge_uncertainty(intervention, posterior_particles)
            assert eig >= 0.0
    
    def test_eig_edge_uncertainty_bounded(self, posterior_particles):
        """Test EIG is bounded by maximum possible entropy reduction."""
        max_entropy = np.log(len(posterior_particles.particles))
        for var in range(3):
            intervention = Intervention(variable=var)
            eig = estimate_eig_via_edge_uncertainty(intervention, posterior_particles)
            assert eig <= max_entropy * 3  # Multiple edges possible


class TestGreedyEIGPolicy:
    """Tests for greedy EIG policy."""
    
    def test_policy_initialization(self):
        """Test policy initialization."""
        policy = GreedyEIGPolicy(eig_method="edge_uncertainty")
        assert policy.eig_method == "edge_uncertainty"
    
    def test_policy_selection(self, posterior_particles):
        """Test intervention selection."""
        policy = GreedyEIGPolicy(eig_method="edge_uncertainty")
        intervention = policy.select_intervention(posterior_particles)
        
        assert isinstance(intervention, Intervention)
        assert 0 <= intervention.variable < 3
    
    def test_policy_restricted_to_ambiguous(self, posterior_particles):
        """Test policy restricted to ambiguous nodes."""
        policy = GreedyEIGPolicy(
            eig_method="edge_uncertainty",
            restrict_to_ambiguous=True
        )
        intervention = policy.select_intervention(
            posterior_particles,
            ambiguous_nodes=[0, 2]
        )
        
        assert intervention.variable in [0, 2]
    
    def test_policy_returns_intervention_object(self, posterior_particles):
        """Test policy returns valid Intervention."""
        policy = GreedyEIGPolicy()
        intervention = policy.select_intervention(posterior_particles)
        
        assert hasattr(intervention, 'variable')
        assert hasattr(intervention, 'value')


class TestRandomPolicy:
    """Tests for random policy."""
    
    def test_random_selection(self, posterior_particles):
        """Test random intervention selection."""
        policy = RandomPolicy()
        intervention = policy.select_intervention(posterior_particles)
        
        assert isinstance(intervention, Intervention)
        assert 0 <= intervention.variable < 3
    
    def test_random_is_random(self, posterior_particles):
        """Test that random policy samples different variables."""
        policy = RandomPolicy()
        interventions = [
            policy.select_intervention(posterior_particles)
            for _ in range(10)
        ]
        
        variables = [i.variable for i in interventions]
        # Should have some variation
        assert len(set(variables)) > 1


class TestOraclePolicy:
    """Tests for oracle policy."""
    
    def test_oracle_initialization(self, simple_dag):
        """Test oracle policy initialization."""
        policy = OraclePolicy(simple_dag)
        assert policy.ground_truth_dag == simple_dag
    
    def test_oracle_selection(self, simple_dag, posterior_particles):
        """Test oracle intervention selection."""
        policy = OraclePolicy(simple_dag)
        intervention = policy.select_intervention(posterior_particles)
        
        assert isinstance(intervention, Intervention)
        assert 0 <= intervention.variable < 3
