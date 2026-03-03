"""Random number generation utilities with global seeding."""

import numpy as np
from typing import Optional


class RNG:
    """Encapsulates numpy RNG for reproducibility."""
    
    _rng: Optional[np.random.Generator] = None
    
    @classmethod
    def set_seed(cls, seed: int) -> None:
        """Set the global random seed."""
        cls._rng = np.random.default_rng(seed)
    
    @classmethod
    def get_rng(cls) -> np.random.Generator:
        """Get the current RNG, creating one with default seed if needed."""
        if cls._rng is None:
            cls._rng = np.random.default_rng(42)
        return cls._rng
    
    @classmethod
    def reset(cls) -> None:
        """Reset RNG to None."""
        cls._rng = None


def set_seed(seed: int) -> None:
    """Convenience function to set global seed."""
    RNG.set_seed(seed)


def get_rng() -> np.random.Generator:
    """Convenience function to get global RNG."""
    return RNG.get_rng()
