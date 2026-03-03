"""MEC-based observational prescreening using constraint-based skeleton detection."""

import numpy as np
from typing import List, Tuple, Set, Optional
from itertools import combinations
from causal_boed.graphs.dag import DAG


class ConstraintBasedPrescreen:
    """
    Simplified constraint-based method for skeleton discovery (similar to PC algorithm).
    
    Identifies conditional independencies from data and removes impossible edges.
    Returns a skeleton (undirected edges) of possibly ambiguous edges.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize constraint-based prescreening.
        
        Args:
            alpha: Significance level for conditional independence tests
        """
        self.alpha = alpha
        self.skeleton = None
        self.separating_sets = {}
    
    def run_pc_skeleton(self, X: np.ndarray) -> Tuple[np.ndarray, Set[Tuple[int, int]]]:
        """
        Run PC algorithm to discover skeleton.
        
        Args:
            X: Data matrix (n_samples, n_vars)
            
        Returns:
            (skeleton, ambiguous_edges) where:
            - skeleton[i,j] = 1 if edge exists (undirected)
            - ambiguous_edges: set of (i, j) that are ambiguous
        """
        n, d = X.shape
        
        # Initialize: complete undirected graph
        skeleton = np.ones((d, d)) - np.eye(d)
        self.separating_sets = {}
        
        # PC algorithm: progressively condition on larger sets
        depth = 0
        while depth < d - 1:
            # For each edge, check if it can be removed
            edges_to_remove = []
            
            for i in range(d):
                for j in range(i + 1, d):
                    if skeleton[i, j] == 0:
                        continue  # Edge already removed
                    
                    # Neighbors of i (not j)
                    neighbors_i = set(np.where(skeleton[i, :] > 0)[0]) - {j}
                    
                    if len(neighbors_i) < depth:
                        continue  # Not enough neighbors to condition on
                    
                    # Try all subsets of neighbors of size depth
                    for cond_set in combinations(neighbors_i, depth):
                        # Test conditional independence: X_i ⊥ X_j | X_cond_set
                        if self._is_independent(X, i, j, list(cond_set)):
                            edges_to_remove.append((i, j))
                            self.separating_sets[(i, j)] = set(cond_set)
                            break
            
            # Remove edges
            for i, j in edges_to_remove:
                skeleton[i, j] = 0
                skeleton[j, i] = 0
            
            depth += 1
        
        self.skeleton = skeleton
        
        # Find ambiguous edges (edges without directional info)
        # For now, all edges in skeleton are ambiguous (conservative estimate)
        ambiguous = set()
        for i, j in zip(*np.where(np.triu(skeleton, k=1) > 0)):
            ambiguous.add((i, j))
        
        return skeleton, ambiguous
    
    def _is_independent(
        self,
        X: np.ndarray,
        i: int,
        j: int,
        cond_set: List[int]
    ) -> bool:
        """
        Test conditional independence X_i ⊥ X_j | X_cond_set using partial correlation.
        
        Args:
            X: Data matrix
            i, j: Indices of variables
            cond_set: Conditioning set indices
            
        Returns:
            True if independent (null hypothesis not rejected)
        """
        n = X.shape[0]
        
        if not cond_set:
            # Unconditional: test X_i ⊥ X_j
            corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
        else:
            # Conditional: compute partial correlation
            # Residuals of regressing X_i and X_j on X_cond_set
            try:
                X_cond = X[:, cond_set]
                
                # Residuals for X_i
                try:
                    beta_i = np.linalg.lstsq(X_cond, X[:, i], rcond=None)[0]
                    res_i = X[:, i] - X_cond @ beta_i
                except:
                    return False
                
                # Residuals for X_j
                try:
                    beta_j = np.linalg.lstsq(X_cond, X[:, j], rcond=None)[0]
                    res_j = X[:, j] - X_cond @ beta_j
                except:
                    return False
                
                # Correlation of residuals
                corr = np.corrcoef(res_i, res_j)[0, 1]
            except:
                return False
        
        # Fisher z-test for correlation
        if np.isnan(corr) or np.abs(corr) >= 1.0:
            return False
        
        z = 0.5 * np.log((1 + corr) / (1 - corr + 1e-10)) * np.sqrt(n - len(cond_set) - 3)
        p_value = 2 * (1 - norm_cdf(np.abs(z)))
        
        return p_value > self.alpha


def norm_cdf(x: float) -> float:
    """Standard normal CDF (approximation)."""
    from scipy.special import erfc
    return 0.5 * erfc(-x / np.sqrt(2))


def identify_ambiguous_edges_from_observational(
    X: np.ndarray,
    threshold: float = 0.1
) -> Tuple[np.ndarray, List[int]]:
    """
    Identify potentially ambiguous edges from observational data.
    
    Uses PC algorithm skeleton to find edges, then returns nodes involved in ambiguous edges.
    
    Args:
        X: Observational data
        threshold: Not used in this version (for future threshold-based filtering)
        
    Returns:
        (skeleton, ambiguous_nodes) where ambiguous_nodes are candidates for intervention
    """
    prescreen = ConstraintBasedPrescreen(alpha=0.05)
    skeleton, ambiguous_edges = prescreen.run_pc_skeleton(X)
    
    # Find nodes involved in ambiguous edges
    ambiguous_nodes = set()
    for i, j in ambiguous_edges:
        ambiguous_nodes.add(i)
        ambiguous_nodes.add(j)
    
    return skeleton, list(ambiguous_nodes)
