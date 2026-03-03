"""Evaluation metrics for causal discovery experiments."""

import numpy as np
from typing import Dict, Optional, Tuple
from causal_boed.graphs.dag import DAG
from causal_boed.inference.posterior import ParticlePosterior


class StructuralMetrics:
    """Metrics based on graph structure (assuming ground truth is known)."""
    
    @staticmethod
    def structural_hamming_distance(inferred_dag: DAG, true_dag: DAG) -> int:
        """
        Compute Structural Hamming Distance (SHD).
        
        Counts: missing edges + extra edges + reversed edges.
        
        Args:
            inferred_dag: Inferred DAG
            true_dag: Ground truth DAG
            
        Returns:
            SHD score (lower is better)
        """
        inferred = inferred_dag.adjacency
        true = true_dag.adjacency
        
        # Missing edges (in true but not inferred)
        missing = np.sum((true > 0) & (inferred == 0))
        
        # Extra edges (in inferred but not true)
        extra = np.sum((inferred > 0) & (true == 0))
        
        # Reversed edges (j->i in inferred, but i->j in true)
        reversed_edges = np.sum((inferred.T > 0) & (true > 0) & (inferred == 0))
        
        shd = missing + extra + reversed_edges
        
        return int(shd)
    
    @staticmethod
    def orientation_accuracy(inferred_dag: DAG, true_dag: DAG) -> Dict[str, float]:
        """
        Compute orientation metrics for edges both sides agree on.
        
        Returns precision, recall, and F1 for correct edge orientation.
        
        Args:
            inferred_dag: Inferred DAG
            true_dag: Ground truth DAG
            
        Returns:
            Dict with 'precision', 'recall', 'f1' keys
        """
        inferred = inferred_dag.adjacency
        true = true_dag.adjacency
        
        # Edges that exist in true graph
        true_edges = (true > 0)
        inferred_edges = (inferred > 0)
        
        # Correct orientations: i->j in both
        correct = np.sum((true > 0) & (inferred > 0))
        
        # All edges in inferred
        total_inferred = np.sum(inferred > 0)
        
        # All edges in true
        total_true = np.sum(true > 0)
        
        # Precision: of inferred edges, how many are correct?
        precision = correct / total_inferred if total_inferred > 0 else 0.0
        
        # Recall: of true edges, how many did we infer correctly?
        recall = correct / total_true if total_true > 0 else 0.0
        
        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


class IdentifiabilityMetrics:
    """Metrics related to identifiability."""
    
    @staticmethod
    def posterior_entropy(posterior: ParticlePosterior) -> float:
        """
        Compute Shannon entropy of posterior.
        
        Higher entropy = more ambiguity = less identifiable.
        Lower entropy = more confident = more identifiable.
        """
        return posterior.entropy()
    
    @staticmethod
    def posterior_uncertainty_edges(posterior: ParticlePosterior) -> Dict:
        """
        Compute uncertainty in edge directions.
        
        Returns dict with various uncertainty metrics.
        """
        edge_probs = posterior.marginal_edge_probs()
        n = edge_probs.shape[0]
        
        # For each edge, compute how uncertain we are
        uncertainties = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    p = edge_probs[i, j]
                    if p > 0 and p < 1:
                        # Bernoulli entropy
                        ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
                        uncertainties.append(ent)
        
        if not uncertainties:
            return {
                "mean_uncertainty": 0.0,
                "max_uncertainty": 0.0,
                "num_uncertain_edges": 0
            }
        
        return {
            "mean_uncertainty": np.mean(uncertainties),
            "max_uncertainty": np.max(uncertainties),
            "num_uncertain_edges": len(uncertainties)
        }
    
    @staticmethod
    def map_accuracy(inferred_dag: DAG, true_dag: DAG) -> float:
        """Accuracy of MAP DAG (is it correct?)."""
        return 1.0 if inferred_dag == true_dag else 0.0


class ExperimentMetrics:
    """Aggregate metrics over an experiment."""
    
    @staticmethod
    def compute_round_metrics(
        posterior: ParticlePosterior,
        ground_truth_dag: DAG,
        use_posterior_map: bool = True
    ) -> Dict:
        """
        Compute all metrics for a given round.
        
        Args:
            posterior: Current posterior over DAGs
            ground_truth_dag: Ground truth causal structure
            use_posterior_map: If True, use MAP estimate; else use mean structure
            
        Returns:
            Dict with all computed metrics
        """
        # Get point estimate
        if use_posterior_map:
            inferred_dag = posterior.map_dag()
        else:
            # Use mean structure (threshold at 0.5)
            mean_adj = posterior.mean_structure()
            mean_adj = (mean_adj > 0.5).astype(int)
            try:
                inferred_dag = DAG(mean_adj)
            except ValueError:
                # Mean structure has cycles; use MAP instead
                inferred_dag = posterior.map_dag()
        
        metrics = {}
        
        # Structural metrics
        shd = StructuralMetrics.structural_hamming_distance(inferred_dag, ground_truth_dag)
        orientation = StructuralMetrics.orientation_accuracy(inferred_dag, ground_truth_dag)
        
        metrics["shd"] = shd
        metrics["orientation_precision"] = orientation["precision"]
        metrics["orientation_recall"] = orientation["recall"]
        metrics["orientation_f1"] = orientation["f1"]
        
        # Identifiability metrics
        metrics["posterior_entropy"] = IdentifiabilityMetrics.posterior_entropy(posterior)
        metrics["map_accuracy"] = IdentifiabilityMetrics.map_accuracy(inferred_dag, ground_truth_dag)
        
        # Uncertainty
        uncertainty = IdentifiabilityMetrics.posterior_uncertainty_edges(posterior)
        metrics.update({f"uncertainty_{k}": v for k, v in uncertainty.items()})
        
        return metrics
