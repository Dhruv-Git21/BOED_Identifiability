"""Intervention specifications and application."""

from dataclasses import dataclass
from typing import Optional, Union, Tuple
import numpy as np


@dataclass
class Intervention:
    """
    Specification of an intervention (do-operation).
    
    Attributes:
        variable: Index of variable to intervene on
        value: Value to set. If None, sample from noise distribution
        intervention_type: "perfect" (deterministic) or "imperfect" (noisy)
        strength: For imperfect interventions, noise level (0-1)
    """
    variable: int
    value: Optional[float] = None
    intervention_type: str = "perfect"
    strength: float = 0.0
    
    def __post_init__(self):
        if self.intervention_type not in ("perfect", "imperfect"):
            raise ValueError(f"Unknown intervention type: {self.intervention_type}")
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError(f"Strength must be in [0, 1], got {self.strength}")
    
    def apply_to_coefficients(self, B: np.ndarray) -> np.ndarray:
        """
        Modify coefficient matrix B to reflect intervention.
        Set incoming edges to intervened variable to 0 (breaks causal dependencies).
        
        Args:
            B: (n, n) coefficient matrix before intervention
            
        Returns:
            Modified B matrix
        """
        B_intervened = B.copy()
        # Remove all incoming edges to intervened variable
        B_intervened[:, self.variable] = 0
        return B_intervened
    
    def __repr__(self) -> str:
        val_str = f"={self.value}" if self.value is not None else ""
        return f"do(X{self.variable}{val_str})"


def apply_intervention(
    X: np.ndarray,
    intervention: Optional[Intervention],
    rng: np.random.Generator
) -> np.ndarray:
    """
    Apply intervention to a sample.
    
    Args:
        X: Sample (n_samples, n_vars) or (n_vars,)
        intervention: Intervention to apply, or None for no intervention
        rng: Random number generator
        
    Returns:
        Modified sample
    """
    if intervention is None:
        return X.copy()
    
    X_intervened = X.copy()
    var = intervention.variable
    
    if intervention.intervention_type == "perfect":
        if intervention.value is not None:
            # Deterministic intervention
            if X_intervened.ndim == 1:
                X_intervened[var] = intervention.value
            else:
                X_intervened[:, var] = intervention.value
        # If value is None, sample from noise (handled externally)
    
    elif intervention.intervention_type == "imperfect":
        # Add noise to make intervention imperfect
        noise_scale = intervention.strength
        if X_intervened.ndim == 1:
            X_intervened[var] += rng.normal(0, noise_scale)
        else:
            X_intervened[:, var] += rng.normal(0, noise_scale, X_intervened[:, var].shape)
    
    return X_intervened
