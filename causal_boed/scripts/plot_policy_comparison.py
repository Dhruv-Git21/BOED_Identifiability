"""
Plot per-round trends (mean ± std across runs) for policy comparison.

Reads:
  BOED_Identifiability/runs/sweep_linear_policy_*/run_*/history.csv
and writes:
  BOED_Identifiability/runs/policy_comparison_trends.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def load_history(sweep_dir: Path) -> pd.DataFrame:
    run_dirs = sorted([p for p in sweep_dir.glob("run_*") if p.is_dir()])
    dfs: List[pd.DataFrame] = []

    for rd in run_dirs:
        hist_path = rd / "history.csv"
        if not hist_path.exists():
            continue
        df = pd.read_csv(hist_path)
        df["run_id"] = rd.name
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No history.csv found in {sweep_dir}")

    return pd.concat(dfs, ignore_index=True)


def plot_trend(
    ax,
    histories: Dict[str, pd.DataFrame],
    metric: str,
    y_label: str,
    title: str,
) -> None:
    for label, hist in histories.items():
        grouped = (
            hist.groupby("round")[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        x = grouped["round"].values
        y = grouped["mean"].values
        yerr = grouped["std"].values
        ax.plot(x, y, marker="o", linewidth=2, label=label)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)

    ax.set_xlabel("Round")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()


def main() -> None:
    # Lazy import: plotting dependencies are only needed when running this script.
    import matplotlib.pyplot as plt

    repo_root = Path(__file__).resolve().parents[3]
    runs_root = repo_root / "BOED_Identifiability" / "runs"

    sweep_map: List[Tuple[str, str]] = [
        ("greedy_eig", "sweep_linear_policy_greedy"),
        ("random", "sweep_linear_policy_random"),
        ("oracle", "sweep_linear_policy_oracle"),
    ]

    histories: Dict[str, pd.DataFrame] = {}
    for label, sweep_name in sweep_map:
        sweep_dir = runs_root / sweep_name
        histories[label] = load_history(sweep_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    plot_trend(
        ax=axes[0],
        histories=histories,
        metric="posterior_entropy",
        y_label="Posterior entropy",
        title="Identifiability over rounds (entropy)",
    )
    plot_trend(
        ax=axes[1],
        histories=histories,
        metric="shd",
        y_label="SHD (lower is better)",
        title="Structure recovery over rounds (SHD)",
    )

    plt.tight_layout()
    out_path = runs_root / "policy_comparison_trends.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()

