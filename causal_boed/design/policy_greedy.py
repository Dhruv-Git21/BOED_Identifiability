"""Greedy intervention selection policy."""

import numpy as np
from typing import Optional, Callable, List, Tuple
from causal_boed.graphs.interventions import Intervention
from causal_boed.inference.posterior import ParticlePosterior
from causal_boed.design.eig import estimate_eig_via_edge_uncertainty, estimate_eig_monte_carlo


class GreedyEIGPolicy:
    """
    Greedy intervention selection based on Expected Information Gain (EIG).
    
    At each round, selects the intervention that maximizes EIG.
    """
    
    def __init__(
        self,
        eig_method: str = "edge_uncertainty",
        restrict_to_ambiguous: bool = True,
        ambiguity_threshold: float = 0.1,
        n_eig_samples: int = 10,
        seed: Optional[int] = None
    ):
        """
        Initialize greedy EIG policy.
        
        Args:
            eig_method: "edge_uncertainty" (fast heuristic) or "monte_carlo" (accurate)
            restrict_to_ambiguous: If True, only consider interventions on ambiguous nodes
            ambiguity_threshold: Threshold for identifying ambiguous edges
            n_eig_samples: Number of samples for Monte Carlo EIG
            seed: Random seed
        """
        self.eig_method = eig_method
        self.restrict_to_ambiguous = restrict_to_ambiguous
        self.ambiguity_threshold = ambiguity_threshold
        self.n_eig_samples = n_eig_samples
        self.seed = seed
    
    def select_intervention(
        self,
        posterior: ParticlePosterior,
        ambiguous_nodes: Optional[List[int]] = None,
        sem_fn: Optional[Callable] = None,
        score_fn: Optional[Callable] = None
    ) -> Intervention:
        """
        Select best intervention according to EIG.
        
        Args:
            posterior: Current posterior over DAGs
            ambiguous_nodes: Nodes involved in ambiguous edges (from prescreening)
            sem_fn: Function to create SEM (required if eig_method == "monte_carlo")
            score_fn: Function to score DAGs (required if eig_method == "monte_carlo")
            
        Returns:
            Selected intervention
        """
        n_vars = posterior.particles[0].n_nodes
        
        # Determine candidate variables
        if self.restrict_to_ambiguous and ambiguous_nodes:
            candidates = ambiguous_nodes
        else:
            candidates = list(range(n_vars))
        
        best_var = None
        best_eig = -np.inf
        
        # Evaluate EIG for each candidate
        for var in candidates:
            intervention = Intervention(variable=var, value=0.0)
            
            if self.eig_method == "edge_uncertainty":
                eig = estimate_eig_via_edge_uncertainty(
                    intervention,
                    posterior,
                    threshold=self.ambiguity_threshold
                )
            elif self.eig_method == "monte_carlo":
                if sem_fn is None or score_fn is None:
                    raise ValueError("sem_fn and score_fn required for monte_carlo EIG")
                eig = estimate_eig_monte_carlo(
                    intervention,
                    posterior,
                    sem_fn=sem_fn,
                    score_fn=score_fn,
                    n_posterior_samples=self.n_eig_samples,
                    n_predictive_samples=5,
                    seed=self.seed
                )
            else:
                raise ValueError(f"Unknown EIG method: {self.eig_method}")
            
            if eig > best_eig:
                best_eig = eig
                best_var = var
        
        if best_var is None:
            # Fallback: random intervention
            best_var = np.random.randint(0, n_vars)
        
        return Intervention(variable=best_var, value=0.0)


class RandomPolicy:
    """Random intervention selection (baseline)."""
    
    def select_intervention(
        self,
        posterior: ParticlePosterior,
        ambiguous_nodes: Optional[List[int]] = None,
        **kwargs
    ) -> Intervention:
        """Select a random intervention."""
        n_vars = posterior.particles[0].n_nodes
        var = np.random.randint(0, n_vars)
        return Intervention(variable=var, value=0.0)


class OraclePolicy:
    """
    Oracle policy: knows ground truth DAG and intervenes on uncertain edges.
    
    Useful as an upper bound on what's achievable.
    """
    
    def __init__(self, ground_truth_dag):
        """
        Initialize oracle policy.
        
        Args:
            ground_truth_dag: The true DAG (known to oracle)
        """
        self.ground_truth_dag = ground_truth_dag
    
    def select_intervention(
        self,
        posterior: ParticlePosterior,
        ambiguous_nodes: Optional[List[int]] = None,
        **kwargs
    ) -> Intervention:
        """
        Select intervention to resolve most uncertain edges w.r.t. ground truth.
        """
        # Find edges where we're most uncertain
        edge_probs = posterior.marginal_edge_probs()
        
        best_var = None
        best_score = -np.inf
        
        n_vars = self.ground_truth_dag.n_nodes
        
        for var in range(n_vars):
            # Score: how much do we learn by intervening on this variable?
            # Heuristic: intervene on variable involved in most uncertain edges
            uncertainty = 0.0
            
            # Edges incident to var
            for other in range(n_vars):
                if other != var:
                    p = edge_probs[var, other]
                    uncertainty += np.abs(p - 0.5)  # Distance from max uncertainty
                    
                    p_rev = edge_probs[other, var]
                    uncertainty += np.abs(p_rev - 0.5)
            
            if uncertainty > best_score:
                best_score = uncertainty
                best_var = var
        
        if best_var is None:
            best_var = 0
        
        return Intervention(variable=best_var, value=0.0)
