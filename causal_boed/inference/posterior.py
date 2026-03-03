"""Particle posterior over DAG structures."""

import numpy as np
from typing import List, Optional, Tuple, Dict
from causal_boed.graphs.dag import DAG


class ParticlePosterior:
    """
    Particle approximation of posterior distribution p(G | D).
    
    Maintains a weighted set of DAG particles.
    """
    
    def __init__(
        self,
        particles: List[DAG],
        log_weights: np.ndarray,
        normalize: bool = True
    ):
        """
        Initialize particle posterior.
        
        Args:
            particles: List of DAG particles
            log_weights: Log-unnormalized weights
            normalize: If True, normalize weights to sum to 1
        """
        self.particles = particles
        self.n_particles = len(particles)
        
        # Normalize log-weights
        if normalize:
            # Log-sum-exp trick for numerical stability
            max_log_w = np.max(log_weights)
            log_weights = log_weights - max_log_w
            weights = np.exp(log_weights)
            weights /= np.sum(weights)
            self.log_weights = np.log(weights + 1e-10)
        else:
            self.log_weights = log_weights.copy()
        
        self.weights = np.exp(self.log_weights)
    
    def entropy(self) -> float:
        """Compute Shannon entropy of posterior."""
        # H(p) = -sum p(G) log p(G)
        entropy = -np.sum(self.weights * self.log_weights)
        return entropy
    
    def marginal_edge_probs(self) -> np.ndarray:
        """
        Compute marginal edge probabilities.
        
        Returns:
            (n_vars, n_vars) matrix where [i,j] = P(i -> j | D)
        """
        n = self.particles[0].n_nodes
        edge_probs = np.zeros((n, n))
        
        for i, (dag, weight) in enumerate(zip(self.particles, self.weights)):
            edge_probs += weight * dag.adjacency
        
        return edge_probs
    
    def sample_from_posterior(
        self,
        n_samples: int,
        seed: Optional[int] = None
    ) -> List[DAG]:
        """
        Sample DAGs from posterior (with replacement).
        
        Args:
            n_samples: Number of samples
            seed: Random seed
            
        Returns:
            List of sampled DAGs
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(self.n_particles, size=n_samples, p=self.weights)
        return [self.particles[i] for i in indices]
    
    def map_dag(self) -> DAG:
        """Return maximum a posteriori DAG."""
        idx = np.argmax(self.weights)
        return self.particles[idx]
    
    def mean_structure(self) -> np.ndarray:
        """Compute expected adjacency matrix."""
        return self.marginal_edge_probs()
    
    def top_k_dags(self, k: int) -> List[Tuple[DAG, float]]:
        """Return top k DAGs by posterior weight."""
        indices = np.argsort(self.weights)[::-1][:k]
        return [(self.particles[i], self.weights[i]) for i in indices]
    
    def resample(self, n_particles: int, seed: Optional[int] = None) -> "ParticlePosterior":
        """
        Resample particles to reduce particle degeneracy.
        
        Args:
            n_particles: Number of particles after resampling
            seed: Random seed
            
        Returns:
            New ParticlePosterior with resampled particles
        """
        sampled = self.sample_from_posterior(n_particles, seed=seed)
        # Equal weights after resampling
        new_log_weights = np.zeros(n_particles)
        return ParticlePosterior(sampled, new_log_weights, normalize=False)
    
    def __repr__(self) -> str:
        return f"ParticlePosterior(particles={self.n_particles}, entropy={self.entropy():.3f})"


def update_particle_posterior(
    posterior: ParticlePosterior,
    score_fn,
    X_new: np.ndarray
) -> ParticlePosterior:
    """
    Update particle posterior with new data.
    
    Uses Bayes rule: p(G | D, D_new) ∝ p(D_new | G) * p(G | D)
    
    Args:
        posterior: Current posterior
        score_fn: Function that computes log p(D_new | G) for a DAG
        X_new: New data
        
    Returns:
        Updated posterior
    """
    # Compute likelihoods for new data under each particle
    log_likelihoods = np.array([
        score_fn(X_new, dag) for dag in posterior.particles
    ])
    
    # Update log-weights: log p(G | D, D_new) = log p(D_new | G) + log p(G | D)
    updated_log_weights = log_likelihoods + posterior.log_weights
    
    return ParticlePosterior(posterior.particles, updated_log_weights, normalize=True)
