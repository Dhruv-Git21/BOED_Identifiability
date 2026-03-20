#!/usr/bin/env bash
set -euo pipefail

python -m causal_boed.scripts.run_synthetic --config "BOED_Identifiability/configs/ablation_linear_policy_greedy.yaml" --runs 30 --no-plot --output "BOED_Identifiability/runs/sweep_linear_policy_greedy"
python -m causal_boed.scripts.run_synthetic --config "BOED_Identifiability/configs/ablation_linear_policy_random.yaml" --runs 30 --no-plot --output "BOED_Identifiability/runs/sweep_linear_policy_random"
python -m causal_boed.scripts.run_synthetic --config "BOED_Identifiability/configs/ablation_linear_policy_oracle.yaml" --runs 30 --no-plot --output "BOED_Identifiability/runs/sweep_linear_policy_oracle"

python -m causal_boed.scripts.run_synthetic --config "BOED_Identifiability/configs/ablation_linear_noise_0_5.yaml" --runs 30 --no-plot --output "BOED_Identifiability/runs/sweep_linear_noise_0_5"
python -m causal_boed.scripts.run_synthetic --config "BOED_Identifiability/configs/ablation_linear_noise_1_0.yaml" --runs 30 --no-plot --output "BOED_Identifiability/runs/sweep_linear_noise_1_0"
python -m causal_boed.scripts.run_synthetic --config "BOED_Identifiability/configs/ablation_linear_noise_2_0.yaml" --runs 30 --no-plot --output "BOED_Identifiability/runs/sweep_linear_noise_2_0"

python -m causal_boed.scripts.run_synthetic --config "BOED_Identifiability/configs/ablation_linear_nobs_50.yaml" --runs 30 --no-plot --output "BOED_Identifiability/runs/sweep_linear_nobs_50"
python -m causal_boed.scripts.run_synthetic --config "BOED_Identifiability/configs/ablation_linear_nobs_100.yaml" --runs 30 --no-plot --output "BOED_Identifiability/runs/sweep_linear_nobs_100"
python -m causal_boed.scripts.run_synthetic --config "BOED_Identifiability/configs/ablation_linear_nobs_200.yaml" --runs 30 --no-plot --output "BOED_Identifiability/runs/sweep_linear_nobs_200"

python -m causal_boed.scripts.run_synthetic --config "BOED_Identifiability/configs/ablation_linear_nrounds_3.yaml" --runs 30 --no-plot --output "BOED_Identifiability/runs/sweep_linear_nrounds_3"
python -m causal_boed.scripts.run_synthetic --config "BOED_Identifiability/configs/ablation_linear_nrounds_5.yaml" --runs 30 --no-plot --output "BOED_Identifiability/runs/sweep_linear_nrounds_5"
python -m causal_boed.scripts.run_synthetic --config "BOED_Identifiability/configs/ablation_linear_nrounds_7.yaml" --runs 30 --no-plot --output "BOED_Identifiability/runs/sweep_linear_nrounds_7"

python -m causal_boed.scripts.run_synthetic --config "BOED_Identifiability/configs/ablation_nonlinear_policy_greedy.yaml" --runs 20 --no-plot --output "BOED_Identifiability/runs/sweep_nonlinear_greedy"
python -m causal_boed.scripts.run_synthetic --config "BOED_Identifiability/configs/ablation_nonlinear_policy_random.yaml" --runs 20 --no-plot --output "BOED_Identifiability/runs/sweep_nonlinear_random"

python -m causal_boed.scripts.run_synthetic --config "BOED_Identifiability/configs/ablation_dream4_small_policy_greedy.yaml" --runs 20 --no-plot --output "BOED_Identifiability/runs/sweep_dream4_small_greedy"
python -m causal_boed.scripts.run_synthetic --config "BOED_Identifiability/configs/ablation_dream4_small_policy_random.yaml" --runs 20 --no-plot --output "BOED_Identifiability/runs/sweep_dream4_small_random"

python -m causal_boed.scripts.consolidate_sweeps
python -m causal_boed.scripts.plot_policy_comparison
