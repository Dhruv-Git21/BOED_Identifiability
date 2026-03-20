"""
Consolidate per-sweep `summary.csv` files into a single master table.

Each sweep directory is expected to look like:
  BOED_Identifiability/runs/sweep_*/summary.csv
  BOED_Identifiability/runs/sweep_*/run_0/config.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


METRICS: List[str] = ["shd", "posterior_entropy", "orientation_f1", "map_accuracy"]


def _deep_get(d: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_sweep_metadata(sweep_dir: Path) -> Dict[str, Any]:
    """Extract key config values from `run_0/config.json`."""
    cfg_path = sweep_dir / "run_0" / "config.json"
    if not cfg_path.exists():
        return {"has_run_0_config": False}

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # `config.json` is a dictified dataclass; nested fields match `config.py`.
    return {
        "has_run_0_config": True,
        "config_name": cfg.get("name"),
        "sem_type": _deep_get(cfg, ["sem", "sem_type"]),
        "sem_noise_std": _deep_get(cfg, ["sem", "noise_std"]),
        "graph_num_nodes": _deep_get(cfg, ["graph", "num_nodes"]),
        "graph_expected_degree": _deep_get(cfg, ["graph", "expected_degree"]),
        "data_n_observational": _deep_get(cfg, ["data", "n_observational"]),
        "data_n_rounds": _deep_get(cfg, ["data", "n_rounds"]),
        "data_n_interventional_per_round": _deep_get(
            cfg, ["data", "n_interventional_per_round"]
        ),
        "inference_score_type": _deep_get(cfg, ["inference", "score_type"]),
        "design_policy": _deep_get(cfg, ["design", "policy"]),
        "n_particles": _deep_get(cfg, ["inference", "n_particles"]),
    }


def summarize_sweep(sweep_dir: Path) -> Dict[str, Any]:
    summary_path = sweep_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.csv: {summary_path}")

    df = pd.read_csv(summary_path)

    row: Dict[str, Any] = {"sweep_dir": str(sweep_dir), "num_runs": int(len(df))}
    row.update(load_sweep_metadata(sweep_dir))

    for m in METRICS:
        if m in df.columns:
            row[f"{m}_mean"] = float(df[m].mean())
            row[f"{m}_std"] = float(df[m].std())
        else:
            row[f"{m}_mean"] = None
            row[f"{m}_std"] = None

    return row


def iter_sweep_dirs(runs_root: Path) -> List[Path]:
    return sorted([p for p in runs_root.glob("sweep_*") if p.is_dir()])


def main() -> None:
    # File: BOED_Identifiability/causal_boed/scripts/consolidate_sweeps.py
    # parents[0]=scripts, parents[1]=causal_boed, parents[2]=BOED_Identifiability,
    # parents[3]=repo/workspace root.
    workspace_root = Path(__file__).resolve().parents[3]
    runs_root = workspace_root / "BOED_Identifiability" / "runs"
    out_path = runs_root / "master_summary.csv"

    sweep_dirs = iter_sweep_dirs(runs_root)
    if not sweep_dirs:
        raise RuntimeError(f"No sweep_* directories found in {runs_root}")

    rows = [summarize_sweep(s) for s in sweep_dirs]
    master = pd.DataFrame(rows)

    master.to_csv(out_path, index=False)
    print(f"Wrote master summary: {out_path}")
    print(
        master[
            [
                "config_name",
                "design_policy",
                "sem_type",
                "sem_noise_std",
                "data_n_observational",
                "data_n_rounds",
                "graph_num_nodes",
                "shd_mean",
                "posterior_entropy_mean",
                "orientation_f1_mean",
                "map_accuracy_mean",
            ]
        ].sort_values(["shd_mean"], ascending=True)
    )


if __name__ == "__main__":
    main()

