"""Nonlinear Additive Noise Model (ANM) SEM."""

import numpy as np
from typing import Callable, Optional
from causal_boed.graphs.dag import DAG
from causal_boed.graphs.interventions import Intervention
from causal_boed.utils.rng import get_rng


class NonlinearANMSEM:
    """
    Nonlinear Additive Noise Model.
    
    Model: X_i = f_i(parents(X_i)) + N_i
    where f_i is a nonlinear function and N_i is noise.
    """
    
    def __init__(
        self,
        dag: DAG,
        functions: Optional[dict] = None,
        noise_std: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize Nonlinear ANM SEM.
        
        Args:
            dag: Causal DAG
            functions: Dict mapping node -> f_i (function of parent values).
                      If None, use simple tanh-based functions.
            noise_std: Noise standard deviation
            seed: Random seed
        """
        self.dag = dag
        self.n_nodes = dag.n_nodes
        self.noise_std = noise_std
        
        if functions is None:
            # Default: simple tanh-based nonlinearity
            self.functions = {}
            for node in range(self.n_nodes):
                parents = self.dag.get_parents(node)
                if not parents:
                    # Root node: identity
                    self.functions[node] = lambda x: np.zeros(len(x)) if isinstance(x, np.ndarray) else 0
                else:
                    # Nonlinear: sum of tanh over given parent values
                    # Note: the function operates directly on the provided
                    # parent array, which is already subset to this node's parents.
                    def make_func():
                        def f(X):
                            # X has shape (n_samples, n_parents) or (n_parents,)
                            if X.ndim == 1:
                                return np.tanh(np.sum(X))
                            else:
                                return np.tanh(np.sum(X, axis=1))
                        return f
                    self.functions[node] = make_func()
        else:
            self.functions = functions
    
    def sample(
        self,
        n_samples: int,
        intervention: Optional[Intervention] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Sample from nonlinear ANM.
        
        Args:
            n_samples: Number of samples
            intervention: Optional intervention
            seed: Random seed
            
        Returns:
            (n_samples, n_nodes) array
        """
        rng = get_rng() if seed is None else np.random.default_rng(seed)
        order = self.dag.topological_sort()
        
        X = np.zeros((n_samples, self.n_nodes))
        
        # Sample noise
        noise = rng.normal(0, self.noise_std, (n_samples, self.n_nodes))
        
        # Fill in topological order
        for node in order:
            if intervention and intervention.variable == node:
                # Break causal connections for intervened node
                if intervention.value is not None:
                    X[:, node] = intervention.value + noise[:, node]
                else:
                    X[:, node] = noise[:, node]
            else:
                parents = self.dag.get_parents(node)
                if parents:
                    # f_i(parents) + noise
                    f_val = self.functions[node](X[:, parents])
                    X[:, node] = f_val + noise[:, node]
                else:
                    X[:, node] = noise[:, node]
        
        return X
    
    def __repr__(self) -> str:
        return f"NonlinearANMSEM(nodes={self.n_nodes}, edges={self.dag.edge_count()})"


def create_nonlinear_anm_sem(
    dag: DAG,
    noise_std: float = 1.0,
    seed: Optional[int] = None
) -> NonlinearANMSEM:
    """Create a Nonlinear ANM SEM with default functions."""
    return NonlinearANMSEM(dag, functions=None, noise_std=noise_std, seed=seed)
