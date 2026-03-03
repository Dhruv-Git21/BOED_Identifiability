"""Linear Gaussian SEM sampling and data generation."""

import numpy as np
from typing import Tuple, Optional
from causal_boed.graphs.dag import DAG
from causal_boed.graphs.interventions import Intervention, apply_intervention
from causal_boed.utils.rng import get_rng


class LinearGaussianSEM:
    """
    Linear Gaussian Structural Equation Model.
    
    Model: X_i = sum_{j in parents(i)} B_{j,i} * X_j + N_i
    where N_i ~ N(0, sigma_i^2)
    """
    
    def __init__(
        self,
        dag: DAG,
        B: Optional[np.ndarray] = None,
        noise_std: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize Linear Gaussian SEM.
        
        Args:
            dag: Causal DAG structure
            B: Coefficient matrix (n_nodes, n_nodes). If None, randomly initialized.
            noise_std: Standard deviation of noise for each variable
            seed: Random seed for initialization
        """
        self.dag = dag
        self.n_nodes = dag.n_nodes
        self.noise_std = noise_std
        
        if B is None:
            rng = get_rng() if seed is None else np.random.default_rng(seed)
            # Initialize B respecting DAG structure
            B = rng.uniform(-1, 1, (self.n_nodes, self.n_nodes))
            B = B * dag.adjacency  # Zero out non-edges
        
        self.B = B
        
        # Noise covariance (diagonal for now)
        self.noise_cov = np.eye(self.n_nodes) * (noise_std ** 2)
    
    def sample(
        self,
        n_samples: int,
        intervention: Optional[Intervention] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Sample from SEM, respecting topological order.
        
        Args:
            n_samples: Number of samples
            intervention: Optional intervention to apply
            seed: Random seed
            
        Returns:
            (n_samples, n_nodes) array of samples
        """
        rng = get_rng() if seed is None else np.random.default_rng(seed)
        
        # Get coefficient matrix (modified by intervention if present)
        B = self.B.copy()
        if intervention is not None:
            B = intervention.apply_to_coefficients(B)
        
        # Topological ordering
        order = self.dag.topological_sort()
        
        # Sample noise
        noise = rng.multivariate_normal(
            mean=np.zeros(self.n_nodes),
            cov=self.noise_cov,
            size=n_samples
        )
        
        # Initialize samples
        X = np.zeros((n_samples, self.n_nodes))
        
        # Fill in values in topological order
        for node in order:
            parents = self.dag.get_parents(node)
            if parents:
                # X_node = sum_j B[j, node] * X_j + noise
                X[:, node] = np.sum(
                    X[:, parents] * B[parents, node].reshape(1, -1),
                    axis=1
                ) + noise[:, node]
            else:
                # Root node
                X[:, node] = noise[:, node]
        
        # Apply intervention value if specified
        if intervention is not None and intervention.value is not None:
            X = apply_intervention(X, intervention, rng)
        
        return X
    
    def interventional_distribution(
        self,
        intervention: Intervention,
        n_samples: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Sample from interventional distribution."""
        return self.sample(n_samples, intervention=intervention, seed=seed)
    
    def __repr__(self) -> str:
        return f"LinearGaussianSEM(nodes={self.n_nodes}, edges={self.dag.edge_count()})"


def create_linear_gaussian_sem(
    dag: DAG,
    coeff_scale: float = 1.0,
    noise_std: float = 1.0,
    seed: Optional[int] = None
) -> LinearGaussianSEM:
    """
    Create a Linear Gaussian SEM with random coefficients.
    
    Args:
        dag: Causal structure
        coeff_scale: Scale of coefficient magnitudes
        noise_std: Noise standard deviation
        seed: Random seed
        
    Returns:
        LinearGaussianSEM instance
    """
    rng = get_rng() if seed is None else np.random.default_rng(seed)
    
    n = dag.n_nodes
    # Initialize coefficients uniformly on [-coeff_scale, coeff_scale]
    B = rng.uniform(-coeff_scale, coeff_scale, (n, n))
    # Zero out non-edges
    B = B * dag.adjacency
    
    return LinearGaussianSEM(dag, B=B, noise_std=noise_std, seed=seed)
