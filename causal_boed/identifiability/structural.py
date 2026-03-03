"""Structural identifiability certificates."""

import numpy as np
from typing import Tuple, List
from causal_boed.inference.posterior import ParticlePosterior
from causal_boed.graphs.dag import DAG


class StructuralIdentifiabilityCertificate:
    """
    Certificate of structural identifiability.
    
    A structure is identifiable if we can uniquely determine the causal edges
    (up to Markov equivalence).
    """
    
    @staticmethod
    def fraction_oriented_edges(posterior: ParticlePosterior) -> float:
        """
        Compute fraction of edges that are uniquely oriented in posterior.
        
        An edge is uniquely oriented if (in all posterior particles) it only goes one direction.
        
        Returns:
            Fraction in [0, 1]. 1.0 means all edges are oriented.
        """
        n_particles = posterior.n_particles
        n_vars = posterior.particles[0].n_nodes
        
        oriented_count = 0
        total_edges = 0
        
        # For each potential edge
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                # Check if edge exists in any direction
                edge_forward_count = sum(
                    1 for dag in posterior.particles if dag.adjacency[i, j] > 0
                )
                edge_backward_count = sum(
                    1 for dag in posterior.particles if dag.adjacency[j, i] > 0
                )
                
                if edge_forward_count > 0 or edge_backward_count > 0:
                    total_edges += 1
                    
                    # Edge is oriented if it's always the same direction
                    if (edge_forward_count == 0) or (edge_backward_count == 0):
                        oriented_count += 1
        
        if total_edges == 0:
            return 1.0
        
        return oriented_count / total_edges
    
    @staticmethod
    def num_ambiguous_edges(
        posterior: ParticlePosterior,
        threshold: float = 0.4
    ) -> int:
        """
        Count edges that could be in either direction (ambiguous).
        
        An edge is ambiguous if P(i->j | D) is close to P(j->i | D).
        
        Args:
            posterior: Posterior over DAGs
            threshold: Edges with |P(i->j) - 0.5| < threshold are ambiguous
            
        Returns:
            Number of ambiguous edges
        """
        edge_probs = posterior.marginal_edge_probs()
        n_vars = edge_probs.shape[0]
        
        ambiguous_count = 0
        
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                p_forward = edge_probs[i, j]
                p_backward = edge_probs[j, i]
                
                # Edge exists if either direction has probability > threshold
                if (p_forward > 0.1) or (p_backward > 0.1):
                    # Check if it's ambiguous (close to 0.5)
                    if min(p_forward, p_backward) > (0.5 - threshold):
                        ambiguous_count += 1
        
        return ambiguous_count
    
    @staticmethod
    def get_certificate(
        posterior: ParticlePosterior,
        ambiguity_threshold: float = 0.1
    ) -> dict:
        """
        Compute structural identifiability certificate.
        
        Args:
            posterior: Posterior over DAGs
            ambiguity_threshold: Threshold for ambiguity
            
        Returns:
            Dict with identifiability metrics
        """
        fraction_oriented = StructuralIdentifiabilityCertificate.fraction_oriented_edges(posterior)
        n_ambiguous = StructuralIdentifiabilityCertificate.num_ambiguous_edges(
            posterior, threshold=ambiguity_threshold
        )
        
        return {
            "fraction_oriented": fraction_oriented,
            "num_ambiguous_edges": n_ambiguous,
            "is_fully_identifiable": fraction_oriented == 1.0 and n_ambiguous == 0,
            "entropy": posterior.entropy()
        }
