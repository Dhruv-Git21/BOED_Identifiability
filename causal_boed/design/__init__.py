"""Design module."""

from causal_boed.design.eig import estimate_eig_monte_carlo, estimate_eig_via_edge_uncertainty
from causal_boed.design.policy_greedy import GreedyEIGPolicy, RandomPolicy, OraclePolicy

__all__ = [
    "estimate_eig_monte_carlo",
    "estimate_eig_via_edge_uncertainty",
    "GreedyEIGPolicy",
    "RandomPolicy",
    "OraclePolicy",
]
