"""BGe (Bayesian Gaussian Equivalent) score for linear Gaussian models."""

import numpy as np
from scipy.special import gammaln
from typing import Optional, Tuple
from causal_boed.graphs.dag import DAG


def bge_score(
    X: np.ndarray,
    dag: DAG,
    alpha_mu: float = 1.0,
    alpha_omega: Optional[np.ndarray] = None
) -> float:
    """
    Compute BGe (Bayesian Gaussian Equivalent) score for a DAG.
    
    BGe is a score function for Linear Gaussian models under the assumption
    that the data is drawn from a multivariate normal distribution.
    
    Score = sum over nodes of local scores
    
    Args:
        X: Data matrix (n_samples, n_vars)
        dag: DAG structure
        alpha_mu: Hyperparameter for coefficient prior mean
        alpha_omega: Hyperparameter matrix for precision prior (n_vars, n_vars)
                    If None, use default (weak prior)
    
    Returns:
        BGe score (higher is better)
    """
    n, d = X.shape
    
    if alpha_omega is None:
        # Default weak prior (unit information prior)
        alpha_omega = np.eye(d)
    
    score = 0.0
    
    # Compute score node by node
    for i in range(d):
        parents = dag.get_parents(i)
        
        # Local score for node i given parents
        if not parents:
            # Root node: score based on variance alone
            score_i = _bge_root_score(X[:, i], n, alpha_omega[i, i])
        else:
            # Non-root: condition on parents
            score_i = _bge_conditional_score(
                X[:, i],
                X[:, parents],
                n,
                alpha_mu,
                alpha_omega,
                i,
                parents
            )
        
        score += score_i
    
    return score


def _bge_root_score(x: np.ndarray, n: int, omega_ii: float) -> float:
    """BGe score for a root node (no parents)."""
    # Variance of x
    var_x = np.var(x, ddof=1)
    
    # BGe formula for root nodes
    # Simplified: log p(data | no parents)
    # Using normal-inverse-gamma conjugate prior
    
    T = n / var_x if var_x > 0 else 0.0
    score = 0.5 * np.log(omega_ii / (omega_ii + n)) - 0.5 * n * np.log(var_x + 1e-10)
    
    return score


def _bge_conditional_score(
    x: np.ndarray,
    X_parents: np.ndarray,
    n: int,
    alpha_mu: float,
    alpha_omega: np.ndarray,
    child_idx: int,
    parent_indices: list
) -> float:
    """BGe score for a child given parents (linear regression setup)."""
    # Fit linear regression: x = X_parents @ beta + noise
    
    # Add intercept
    X_aug = np.column_stack([np.ones(n), X_parents])
    
    try:
        # Normal equations: beta_hat = (X^T X)^{-1} X^T y
        XtX = X_aug.T @ X_aug
        Xty = X_aug.T @ x
        
        # Check for singularity
        if np.linalg.cond(XtX) > 1e10:
            # Singular; return weak score
            return 0.0
        
        beta_hat = np.linalg.solve(XtX, Xty)
        
        # Residual variance
        residuals = x - X_aug @ beta_hat
        rss = np.sum(residuals ** 2)
        sigma2_hat = rss / n if n > 0 else 1.0
        
        # BGe score uses precision and prior
        # Simplified: log p(data | parents) under normal-inverse-gamma prior
        p = X_aug.shape[1]  # Including intercept
        
        score = -0.5 * n * np.log(sigma2_hat + 1e-10)
        score += 0.5 * p * np.log(alpha_mu)
        score -= 0.5 * np.log(np.linalg.det(XtX + 1e-10))
        
        return score
    
    except np.linalg.LinAlgError:
        # Numerical issues
        return 0.0


def bic_score(X: np.ndarray, dag: DAG) -> float:
    """
    Compute BIC (Bayesian Information Criterion) score.
    
    BIC = sum_i [n/2 * log(MSE_i) + k_i/2 * log(n)]
    
    where k_i is number of parents of node i.
    
    Lower BIC is better, but we return -BIC so higher is better.
    
    Args:
        X: Data matrix (n_samples, n_vars)
        dag: DAG structure
        
    Returns:
        -BIC score (negated, so higher is better to match log-likelihood convention)
    """
    n, d = X.shape
    bic = 0.0
    
    for i in range(d):
        parents = dag.get_parents(i)
        k_i = len(parents)
        
        if k_i == 0:
            # Root node
            var_i = np.var(X[:, i], ddof=1)
            mse_i = var_i
        else:
            # Regress on parents
            X_parents = X[:, parents]
            
            try:
                # OLS regression
                XtX = X_parents.T @ X_parents
                Xty = X_parents.T @ X[:, i]
                beta = np.linalg.solve(XtX, Xty)
                
                residuals = X[:, i] - X_parents @ beta
                mse_i = np.mean(residuals ** 2)
            except np.linalg.LinAlgError:
                # Singular matrix
                mse_i = np.var(X[:, i], ddof=1)
        
        # BIC contribution for node i
        bic += 0.5 * n * np.log(mse_i + 1e-10) + 0.5 * k_i * np.log(n + 1e-10)
    
    return -bic  # Return negative so higher is better
