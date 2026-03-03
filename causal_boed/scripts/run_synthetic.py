"""Run synthetic BOED experiments."""

import typer
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from causal_boed.config import load_config
from causal_boed.experiment import BOEDExperiment
from causal_boed.utils.logging import setup_logging

app = typer.Typer(help="Run BOED experiments on synthetic data")


@app.command()
def main(
    config_path: str = typer.Option(
        "configs/default_linear.yaml",
        "--config",
        "-c",
        help="Path to config file"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: runs/)"
    ),
    num_runs: int = typer.Option(
        1,
        "--runs",
        "-n",
        help="Number of experiment runs"
    ),
    seed_offset: int = typer.Option(
        0,
        "--seed-offset",
        help="Offset for random seed across runs"
    ),
    plot: bool = typer.Option(
        True,
        "--plot/--no-plot",
        help="Generate plots"
    )
):
    """
    Run synthetic BOED experiment.
    
    Example:
        python -m causal_boed.scripts.run_synthetic --config configs/default_linear.yaml
    """
    # Load config
    config = load_config(config_path)
    
    # Setup logging
    if output_dir is None:
        output_dir = "runs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir, level=20)  # INFO level
    logger.info(f"Running {num_runs} experiments with config: {config_path}")
    
    results_all = []
    
    for run_idx in range(num_runs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Run {run_idx + 1}/{num_runs}")
        logger.info(f"{'='*60}")
        
        # Modify seed for this run
        config.seed = config.seed + seed_offset + run_idx
        
        # Run experiment
        exp = BOEDExperiment(config)
        results = exp.run(output_dir / f"run_{run_idx}")
        results_all.append(results)
    
    # Aggregate results
    logger.info(f"\n{'='*60}")
    logger.info("Aggregating results...")
    logger.info(f"{'='*60}")
    
    aggregate_results(results_all, output_dir, logger, plot=plot)
    
    logger.info(f"All results saved to {output_dir}")


def aggregate_results(results_list, output_dir, logger, plot=True):
    """Aggregate results across multiple runs."""
    # Extract final metrics
    final_metrics = []
    for res in results_list:
        metrics = res["final_metrics"]
        metrics["run"] = len(final_metrics)
        final_metrics.append(metrics)
    
    df_final = pd.DataFrame(final_metrics)
    
    # Save summary
    summary_path = output_dir / "summary.csv"
    df_final.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")
    
    # Print summary stats
    logger.info("\nFinal metrics (mean ± std):")
    logger.info(f"  SHD: {df_final['shd'].mean():.2f} ± {df_final['shd'].std():.2f}")
    logger.info(f"  Posterior Entropy: {df_final['posterior_entropy'].mean():.3f} ± {df_final['posterior_entropy'].std():.3f}")
    logger.info(f"  MAP Accuracy: {df_final['map_accuracy'].mean():.3f} ± {df_final['map_accuracy'].std():.3f}")
    
    if plot:
        _plot_results(results_list, output_dir, logger)


def _plot_results(results_list, output_dir, logger):
    """Generate plots from results."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for run_idx, res in enumerate(results_list):
            history = res["history"]
            df = pd.DataFrame(history)
            
            # Entropy over rounds
            axes[0, 0].plot(df["round"], df["posterior_entropy"], marker="o", label=f"Run {run_idx}")
            axes[0, 0].set_xlabel("Round")
            axes[0, 0].set_ylabel("Posterior Entropy")
            axes[0, 0].set_title("Entropy Reduction")
            axes[0, 0].grid(True, alpha=0.3)
            
            # SHD over rounds
            axes[0, 1].plot(df["round"], df["shd"], marker="o", label=f"Run {run_idx}")
            axes[0, 1].set_xlabel("Round")
            axes[0, 1].set_ylabel("SHD")
            axes[0, 1].set_title("Structural Hamming Distance")
            axes[0, 1].grid(True, alpha=0.3)
            
            # Orientation F1 over rounds
            axes[1, 0].plot(df["round"], df["orientation_f1"], marker="o", label=f"Run {run_idx}")
            axes[1, 0].set_xlabel("Round")
            axes[1, 0].set_ylabel("F1 Score")
            axes[1, 0].set_title("Orientation F1")
            axes[1, 0].grid(True, alpha=0.3)
            
            # MAP accuracy over rounds
            axes[1, 1].plot(df["round"], df["map_accuracy"], marker="o", label=f"Run {run_idx}")
            axes[1, 1].set_xlabel("Round")
            axes[1, 1].set_ylabel("Accuracy")
            axes[1, 1].set_title("MAP Accuracy")
            axes[1, 1].grid(True, alpha=0.3)
        
        for ax in axes.flat:
            ax.legend()
        
        plt.tight_layout()
        plot_path = output_dir / "results.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {plot_path}")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")


if __name__ == "__main__":
    app()
