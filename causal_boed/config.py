"""Configuration system with dataclasses and YAML loading."""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Any, Dict, Union
import yaml


@dataclass
class GraphConfig:
    """Configuration for DAG generation."""
    num_nodes: int = 5
    expected_degree: float = 1.5
    seed: int = 42


@dataclass
class SEMConfig:
    """Configuration for SEM (Structural Equation Model)."""
    sem_type: str = "linear_gaussian"  # "linear_gaussian" or "nonlinear_anm"
    noise_std: float = 1.0
    coeff_scale: float = 1.0
    seed: int = 42


@dataclass
class DataConfig:
    """Configuration for data collection."""
    n_observational: int = 100
    n_interventional_per_round: int = 50
    n_rounds: int = 5
    seed: int = 42


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    n_particles: int = 50
    n_mh_steps: int = 100
    score_type: str = "bge"  # "bge" or "bic"
    use_constraint_based_prescreen: bool = True
    seed: int = 42


@dataclass
class DesignConfig:
    """Configuration for intervention design."""
    policy: str = "greedy_eig"  # "greedy_eig", "random", "oracle"
    n_eig_samples: int = 10
    restrict_to_ambiguous: bool = True
    ambiguity_threshold: float = 0.1
    seed: int = 42


@dataclass
class IdentifiabilityConfig:
    """Configuration for identifiability certificates."""
    structural_enabled: bool = True
    query_enabled: bool = False  # Stub for now
    seed: int = 42


@dataclass
class EvalConfig:
    """Configuration for evaluation metrics."""
    compute_shd: bool = True
    compute_sid: bool = False  # Requires causal-learn or similar
    compute_orientation_accuracy: bool = True
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Master experiment configuration."""
    name: str = "default_experiment"
    graph: GraphConfig = field(default_factory=GraphConfig)
    sem: SEMConfig = field(default_factory=SEMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    design: DesignConfig = field(default_factory=DesignConfig)
    identifiability: IdentifiabilityConfig = field(default_factory=IdentifiabilityConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42
    output_dir: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary (e.g., from YAML)."""
        # Handle nested dataclasses
        if "graph" in data and isinstance(data["graph"], dict):
            data["graph"] = GraphConfig(**data["graph"])
        if "sem" in data and isinstance(data["sem"], dict):
            data["sem"] = SEMConfig(**data["sem"])
        if "data" in data and isinstance(data["data"], dict):
            data["data"] = DataConfig(**data["data"])
        if "inference" in data and isinstance(data["inference"], dict):
            data["inference"] = InferenceConfig(**data["inference"])
        if "design" in data and isinstance(data["design"], dict):
            data["design"] = DesignConfig(**data["design"])
        if "identifiability" in data and isinstance(data["identifiability"], dict):
            data["identifiability"] = IdentifiabilityConfig(**data["identifiability"])
        if "evaluation" in data and isinstance(data["evaluation"], dict):
            data["evaluation"] = EvalConfig(**data["evaluation"])
        
        return cls(**data)


class Config:
    """Convenience class for loading/saving configs."""
    
    @staticmethod
    def load(path: Union[Path, str]) -> ExperimentConfig:
        """Load config from YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return ExperimentConfig.from_dict(data or {})
    
    @staticmethod
    def save(config: ExperimentConfig, path: Union[Path, str]) -> None:
        """Save config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def default() -> ExperimentConfig:
        """Return default config."""
        return ExperimentConfig()


def load_config(path: Union[Path, str]) -> ExperimentConfig:
    """Convenience function to load config."""
    return Config.load(path)
