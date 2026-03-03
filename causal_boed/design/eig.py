"""Expected Information Gain (EIG) computation for intervention design."""

import numpy as np
from typing import Callable, Optional, List
from causal_boed.graphs.dag import DAG
from causal_boed.graphs.interventions import Intervention
from causal_boed.inference.posterior import ParticlePosterior
from causal_boed.sem.linear_gaussian import LinearGaussianSEM
from causal_boed.utils.rng import get_rng


def estimate_eig_monte_carlo(
    intervention: Intervention,
    posterior: ParticlePosterior,
    sem_fn: Callable,
    score_fn: Callable,
    n_posterior_samples: int = 10,
    n_predictive_samples: int = 5,
    seed: Optional[int] = None
) -> float:
    """
    Estimate EIG for an intervention via Monte Carlo.
    
    EIG(I) = E_{G ~ p(G|D)} [ H(p(G|D)) - H(p(G|D, D_I)) ]
    
    where:
    - D_I ~ p(D | G, I) is hypothetical data under intervention I
    - H is Shannon entropy
    
    Algorithm:
    1. Sample G from posterior
    2. For each G, simulate data under intervention
    3. Compute posterior entropy after observing simulated data
    4. Compute entropy reduction
    5. Average over samples
    
    Args:
        intervention: Intervention to evaluate
        posterior: Current posterior over DAGs
        sem_fn: Function that creates SEM given DAG (should have same params across all calls)
        score_fn: Function that scores data under a DAG (returns log p(D|G))
        n_posterior_samples: Number of posterior samples
        n_predictive_samples: Number of hypothetical data samples per posterior sample
        seed: Random seed
        
    Returns:
        Estimated EIG (non-negative)
    """
    rng = get_rng() if seed is None else np.random.default_rng(seed)
    
    # Current entropy
    H_before = posterior.entropy()
    
    # Sample from posterior
    sampled_dags = posterior.sample_from_posterior(n_posterior_samples, seed=seed)
    
    entropy_reductions = []
    
    for dag in sampled_dags:
        # Create SEM for this DAG
        sem = sem_fn(dag)
        
        # Simulate hypothetical interventional data
        X_sim = sem.sample(n_predictive_samples, intervention=intervention, seed=None)
        
        # Compute scores for all particles on simulated data
        log_likelihoods = np.array([
            score_fn(X_sim, p_dag) for p_dag in posterior.particles
        ])
        
        # Updated log-weights: log p(G | D, D_I)
        updated_log_weights = log_likelihoods + posterior.log_weights
        
        # Normalize
        max_log_w = np.max(updated_log_weights)
        log_weights_norm = updated_log_weights - max_log_w
        weights_norm = np.exp(log_weights_norm)
        weights_norm /= np.sum(weights_norm)
        
        # Entropy after intervention
        log_weights_norm = np.log(weights_norm + 1e-10)
        H_after = -np.sum(weights_norm * log_weights_norm)
        
        # Entropy reduction for this sample
        entropy_reduction = max(0, H_before - H_after)
        entropy_reductions.append(entropy_reduction)
    
    # EIG is average entropy reduction
    eig = np.mean(entropy_reductions)
    
    return max(0.0, eig)


def estimate_eig_via_edge_uncertainty(
    intervention: Intervention,
    posterior: ParticlePosterior,
    threshold: float = 0.1
) -> float:
    """
    Estimate EIG using marginal edge probability uncertainty (heuristic).
    
    Intuition: intervening on a node affects edges incident to it.
    EIG is approximated as the sum of entropies (Bernoulli) of edge probabilities
    for edges incident to the intervened variable.
    
    This is much faster than Monte Carlo but less accurate.
    
    Args:
        intervention: Intervention to evaluate
        posterior: Current posterior
        threshold: Unused (for future use)
        
    Returns:
        Estimated EIG via edge uncertainty
    """
    # Marginal edge probabilities
    edge_probs = posterior.marginal_edge_probs()
    
    var = intervention.variable
    
    # Entropy of edges involving this variable
    # Incoming edges: j -> var
    incoming_uncertainty = 0.0
    for parent in range(posterior.particles[0].n_nodes):
        if parent != var:
            p = edge_probs[parent, var]
            incoming_uncertainty += _bernoulli_entropy(p)
    
    # Outgoing edges: var -> j
    outgoing_uncertainty = 0.0
    for child in range(posterior.particles[0].n_nodes):
        if child != var:
            p = edge_probs[var, child]
            outgoing_uncertainty += _bernoulli_entropy(p)
    
    # Total uncertainty reduction
    eig = incoming_uncertainty + outgoing_uncertainty
    
    return eig


def _bernoulli_entropy(p: float) -> float:
    """Entropy of Bernoulli(p): -p log p - (1-p) log(1-p)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))
