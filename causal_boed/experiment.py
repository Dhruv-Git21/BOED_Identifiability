"""Core experiment orchestration."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
import json
from datetime import datetime

from causal_boed.config import ExperimentConfig
from causal_boed.graphs.dag import sample_random_dag, DAG
from causal_boed.sem.linear_gaussian import create_linear_gaussian_sem
from causal_boed.sem.nonlinear_anm import create_nonlinear_anm_sem
from causal_boed.inference.score_bge import bge_score, bic_score
from causal_boed.inference.posterior import ParticlePosterior, update_particle_posterior
from causal_boed.inference.mec_prescreen import identify_ambiguous_edges_from_observational
from causal_boed.design.policy_greedy import GreedyEIGPolicy, RandomPolicy, OraclePolicy
from causal_boed.identifiability.structural import StructuralIdentifiabilityCertificate
from causal_boed.eval.metrics import ExperimentMetrics
from causal_boed.utils.rng import set_seed, get_rng
from causal_boed.utils.logging import setup_logging, get_logger


class BOEDExperiment:
    """Single BOED experiment: synthetic data generation and sequential intervention."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize experiment with config."""
        self.config = config
        self.logger = get_logger("causal_boed.experiment")
        
        # Set seeds
        set_seed(config.seed)
        
        # Will be set during run
        self.ground_truth_dag: Optional[DAG] = None
        self.sem = None
        self.data_obs = None
        self.data_interventional = []
        self.posterior: Optional[ParticlePosterior] = None
        self.history = []
    
    def run(self, output_dir: Optional[Path] = None) -> dict:
        """
        Run complete BOED experiment.
        
        Returns:
            Results dict with metrics, history, etc.
        """
        if output_dir is None:
            output_dir = Path("runs") / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Running experiment: {self.config.name}")
        self.logger.info(f"Output dir: {output_dir}")
        
        # Step 1: Generate ground truth
        self.logger.info("Step 1: Generating ground truth DAG and SEM...")
        self._generate_ground_truth()
        
        # Step 2: Generate observational data
        self.logger.info("Step 2: Generating observational data...")
        self._generate_observational_data()
        
        # Step 3: Observational prescreening
        self.logger.info("Step 3: Running observational prescreening...")
        ambiguous_nodes = self._run_prescreening()
        self.logger.info(f"Identified {len(ambiguous_nodes)} ambiguous nodes: {ambiguous_nodes}")
        
        # Step 4: Initialize posterior
        self.logger.info("Step 4: Initializing posterior from observational data...")
        self._initialize_posterior()
        
        # Step 5: Create intervention policy
        self.logger.info(f"Step 5: Creating intervention policy ({self.config.design.policy})...")
        policy = self._create_policy(ambiguous_nodes)
        
        # Step 6: Sequential intervention rounds
        self.logger.info(f"Step 6: Running {self.config.data.n_rounds} intervention rounds...")
        for round_idx in range(self.config.data.n_rounds):
            self.logger.info(f"  Round {round_idx + 1}/{self.config.data.n_rounds}")
            self._intervention_round(policy, ambiguous_nodes, round_idx)
        
        # Step 7: Collect results
        self.logger.info("Step 7: Computing final metrics...")
        results = self._collect_results(output_dir)
        
        self.logger.info("Experiment complete!")
        
        return results
    
    def _generate_ground_truth(self):
        """Generate random DAG and SEM."""
        self.ground_truth_dag = sample_random_dag(
            num_nodes=self.config.graph.num_nodes,
            expected_degree=self.config.graph.expected_degree,
            seed=self.config.graph.seed
        )
        self.logger.info(f"Generated DAG: {self.ground_truth_dag}")
        
        # Create SEM
        if self.config.sem.sem_type == "linear_gaussian":
            self.sem = create_linear_gaussian_sem(
                self.ground_truth_dag,
                coeff_scale=self.config.sem.coeff_scale,
                noise_std=self.config.sem.noise_std,
                seed=self.config.sem.seed
            )
        elif self.config.sem.sem_type == "nonlinear_anm":
            self.sem = create_nonlinear_anm_sem(
                self.ground_truth_dag,
                noise_std=self.config.sem.noise_std,
                seed=self.config.sem.seed
            )
        else:
            raise ValueError(f"Unknown SEM type: {self.config.sem.sem_type}")
    
    def _generate_observational_data(self):
        """Sample observational data (no intervention)."""
        self.data_obs = self.sem.sample(
            n_samples=self.config.data.n_observational,
            intervention=None,
            seed=self.config.data.seed
        )
        self.logger.info(f"Generated {self.data_obs.shape[0]} observational samples")
    
    def _run_prescreening(self) -> List[int]:
        """Run observational prescreening to identify ambiguous nodes."""
        _, ambiguous_nodes = identify_ambiguous_edges_from_observational(
            self.data_obs,
            threshold=self.config.design.ambiguity_threshold
        )
        return ambiguous_nodes
    
    def _initialize_posterior(self):
        """Initialize posterior from observational data."""
        # Sample DAG particles
        n_particles = self.config.inference.n_particles
        particles = []
        
        for _ in range(n_particles):
            dag = sample_random_dag(
                num_nodes=self.config.graph.num_nodes,
                expected_degree=self.config.graph.expected_degree,
                seed=None
            )
            particles.append(dag)
        
        # Score each particle
        score_fn = self._get_score_function()
        log_scores = np.array([score_fn(self.data_obs, dag) for dag in particles])
        
        # Create posterior
        self.posterior = ParticlePosterior(particles, log_scores, normalize=True)
        self.logger.info(f"Initialized posterior with {n_particles} particles")
        self.logger.info(f"Posterior entropy: {self.posterior.entropy():.3f}")
    
    def _intervention_round(self, policy, ambiguous_nodes: List[int], round_idx: int):
        """Execute one intervention round."""
        # Select intervention
        intervention = policy.select_intervention(
            self.posterior,
            ambiguous_nodes=ambiguous_nodes
        )
        self.logger.info(f"  Selected intervention: {intervention}")
        
        # Collect interventional data
        X_int = self.sem.sample(
            n_samples=self.config.data.n_interventional_per_round,
            intervention=intervention,
            seed=None
        )
        
        # Combine with previous data
        combined_data = np.vstack([self.data_obs] + self.data_interventional + [X_int])
        
        # Update posterior
        score_fn = self._get_score_function()
        self.posterior = update_particle_posterior(
            self.posterior,
            score_fn=score_fn,
            X_new=X_int
        )
        
        # Record metrics
        metrics = ExperimentMetrics.compute_round_metrics(
            self.posterior,
            self.ground_truth_dag,
            use_posterior_map=True
        )
        metrics["round"] = round_idx
        metrics["intervention"] = str(intervention)
        
        self.history.append(metrics)
        
        self.logger.info(f"  Posterior entropy: {metrics['posterior_entropy']:.3f}, "
                        f"SHD: {metrics['shd']}")
    
    def _get_score_function(self):
        """Return appropriate scoring function."""
        if self.config.inference.score_type == "bge":
            return bge_score
        elif self.config.inference.score_type == "bic":
            return bic_score
        else:
            raise ValueError(f"Unknown score type: {self.config.inference.score_type}")
    
    def _create_policy(self, ambiguous_nodes: List[int]):
        """Create intervention policy."""
        policy_name = self.config.design.policy
        
        if policy_name == "greedy_eig":
            return GreedyEIGPolicy(
                eig_method="edge_uncertainty",
                restrict_to_ambiguous=self.config.design.restrict_to_ambiguous,
                ambiguity_threshold=self.config.design.ambiguity_threshold,
                n_eig_samples=self.config.design.n_eig_samples,
                seed=self.config.design.seed
            )
        elif policy_name == "random":
            return RandomPolicy()
        elif policy_name == "oracle":
            return OraclePolicy(self.ground_truth_dag)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")
    
    def _collect_results(self, output_dir: Path) -> dict:
        """Collect and save results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute identifiability certificate
        id_cert = StructuralIdentifiabilityCertificate.get_certificate(self.posterior)
        
        # Final metrics
        final_metrics = ExperimentMetrics.compute_round_metrics(
            self.posterior,
            self.ground_truth_dag
        )
        
        # Results dict
        results = {
            "config": self.config.to_dict(),
            "ground_truth_dag": self.ground_truth_dag.adjacency.tolist(),
            "final_metrics": final_metrics,
            "identifiability_cert": id_cert,
            "history": self.history,
            "num_particles": self.config.inference.n_particles,
            "num_rounds": self.config.data.n_rounds
        }
        
        # Save config
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save results
        results_path = output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save history as CSV
        if self.history:
            df = pd.DataFrame(self.history)
            csv_path = output_dir / "history.csv"
            df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Results saved to {output_dir}")
        
        return results
