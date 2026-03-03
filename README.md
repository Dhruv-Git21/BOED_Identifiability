# Causal BOED: Identifiability-Aware Bayesian Optimal Experimental Design

A research-grade Python framework for finite-budget active causal discovery using Bayesian optimal experimental design (BOED).

**Main idea**: Given finite resources, which interventions maximize our ability to identify causal structures?

## Quick Start

### Installation

**Option 1: Using conda**
```bash
conda env create -f env.yml
conda activate causal-boed
```

**Option 2: Using uv (recommended for speed)**
```bash
uv sync
```

**Option 3: Manual pip**
```bash
pip install -e .
```

### Run a Simple Experiment

```bash
# Run default synthetic experiment (5-node linear Gaussian)
python -m causal_boed.scripts.run_synthetic --config configs/default_linear.yaml

# Run 3 experiments and aggregate results
python -m causal_boed.scripts.run_synthetic --config configs/default_linear.yaml --runs 3

# Run nonlinear ANM model
python -m causal_boed.scripts.run_synthetic --config configs/default_nonlinear.yaml
```

Results will be saved to `runs/` directory with metrics, plots, and detailed logs.

### Run Tests

```bash
pytest tests/ -v
```

## How It Works

### The BOED Loop

1. **Observational Phase**: Collect data without interventions
   - Data: $D_{obs} \sim P(X | G^*)$ where $G^*$ is unknown ground truth
   - Learn from conditional independencies (PC algorithm skeleton)

2. **Initial Inference**: Infer posterior over DAGs
   - $P(G | D_{obs})$ via BGe (Bayesian Gaussian Equivalent) score
   - Maintain particle approximation (50-100 DAGs)
   - Compute posterior entropy (measure of uncertainty)

3. **Intervention Design**: Select most informative intervention
   - For each candidate $i$, estimate $\text{EIG}(X_i) = \mathbb{E}[\text{entropy reduction}]$
   - Use fast heuristic: edge uncertainty (incident edge probabilities)
   - Select intervention on variable with highest EIG

4. **Sequential Update**: Collect interventional data and update belief
   - Run intervention $do(X_i = 0)$ or sample from post-intervention distribution
   - Score all particles on new data
   - Update posterior weights via Bayes rule
   - Repeat for B rounds (default: 5 rounds)

5. **Evaluation**: Assess identifiability
   - Structural metrics: SHD (Structural Hamming Distance), orientation F1
   - Identifiability metric: posterior entropy (lower = more identified)
   - MAP accuracy: did we recover ground truth?

### Key Modules

```
causal_boed/
├── graphs/          # DAG generation, interventions
├── sem/             # Linear Gaussian and nonlinear ANM sampling
├── inference/       # BGe/BIC scoring, particle posterior, PC prescreening
├── design/          # EIG computation, greedy policy, oracle upper bound
├── identifiability/ # Structural ID certificates
├── eval/            # Metrics (SHD, entropy, F1)
├── config.py        # Configuration system (YAML + dataclasses)
├── experiment.py    # Main experiment orchestration
└── utils/           # RNG, logging
```

## Configuration

Experiments are configured via YAML files in `configs/`.

**Default config** (`configs/default_linear.yaml`):
```yaml
graph:
  num_nodes: 5
  expected_degree: 1.5
sem:
  sem_type: linear_gaussian
  noise_std: 1.0
data:
  n_observational: 100
  n_interventional_per_round: 50
  n_rounds: 5
inference:
  n_particles: 50
  score_type: bge
design:
  policy: greedy_eig          # Options: random, greedy_eig, oracle
  restrict_to_ambiguous: true # Only intervene on uncertain nodes
```

### Predefined Configs

- `default_linear.yaml`: 5-node linear Gaussian (quick test)
- `default_nonlinear.yaml`: 5-node nonlinear ANM
- `dream4_small.yaml`: 10-node DREAM4-like graph
- `sachs.yaml`: 11-node (Sachs structure) — requires real data

## Understanding the Output

Each experiment produces:

```
runs/experiment_<timestamp>/
├── config.json           # Full configuration used
├── results.json          # Complete results and history
├── history.csv           # Per-round metrics
└── results.png          # Plots of metrics over rounds
```

### Key Metrics

- **Posterior Entropy** (H): Shannon entropy of p(G|D)
  - Higher = more ambiguous = need more interventions
  - Directly optimized by EIG
  - Target: H → 0 (complete identification)

- **SHD** (Structural Hamming Distance): Graph comparison
  - Missing + Extra + Reversed edges
  - Target: SHD = 0 (perfect recovery)

- **Orientation F1**: How well we recover edge directions
  - Only for edges both sides agree exist
  - Target: F1 = 1.0

- **MAP Accuracy**: Did we correctly identify G*?
  - 0 or 1 (binary)
  - Equals 1 if MAP DAG == ground truth

### Example Output

```
Final metrics (mean ± std):
  SHD: 2.34 ± 1.02
  Posterior Entropy: 0.45 ± 0.12
  MAP Accuracy: 0.67 ± 0.47
  Orientation F1: 0.85 ± 0.08
```

## Intervention Policies

### 1. Greedy EIG (Default)

Selects intervention maximizing expected information gain.

```python
from causal_boed.design.policy_greedy import GreedyEIGPolicy

policy = GreedyEIGPolicy(
    eig_method="edge_uncertainty",  # Fast heuristic
    restrict_to_ambiguous=True,      # Only intervene on uncertain nodes
)
```

**Speed**: ~milliseconds per decision (heuristic EIG)

**Accuracy**: Good in practice; true EIG requires MC sampling from posterior predictive

### 2. Random (Baseline)

Uniform random variable selection.

```python
from causal_boed.design.policy_greedy import RandomPolicy
policy = RandomPolicy()
```

**Use case**: Baseline comparison

### 3. Oracle (Upper Bound)

Knows ground truth DAG and intervenes on uncertain edges.

```python
from causal_boed.design.policy_greedy import OraclePolicy
policy = OraclePolicy(ground_truth_dag)
```

**Use case**: Measure gap between greedy and optimal

## Extending the Framework

### Add a New Intervention Policy

```python
# In causal_boed/design/my_policy.py

from causal_boed.graphs.interventions import Intervention
from causal_boed.inference.posterior import ParticlePosterior

class MyPolicy:
    def select_intervention(self, posterior, ambiguous_nodes=None, **kwargs):
        # Your logic here
        var = best_variable(posterior, ambiguous_nodes)
        return Intervention(variable=var, value=0.0)
```

### Add a New Evaluation Metric

```python
# In causal_boed/eval/metrics.py

@staticmethod
def my_metric(inferred_dag, ground_truth_dag):
    # Compute your metric
    return score

# Use in ExperimentMetrics.compute_round_metrics()
```

### Use Full EIG via Monte Carlo

```python
from causal_boed.design.eig import estimate_eig_monte_carlo
from causal_boed.design.policy_greedy import GreedyEIGPolicy

class FullEIGPolicy(GreedyEIGPolicy):
    def select_intervention(self, posterior, sem_fn, score_fn, **kwargs):
        # Use full EIG instead of heuristic
        eig = estimate_eig_monte_carlo(
            intervention, posterior, sem_fn, score_fn,
            n_posterior_samples=20, n_predictive_samples=5
        )
```

**Tradeoff**: Much slower (~seconds per decision) but more accurate.

## Roadmap

### Near-term

- [x] Linear Gaussian SEM + BGe scoring
- [x] Nonlinear ANM sampling
- [x] Particle posterior + sequential updates
- [x] Greedy EIG via edge uncertainty heuristic
- [x] PC-based observational prescreening
- [x] Structural identifiability certificates

### Medium-term

- [ ] Full EIG via Monte Carlo over posterior predictive
- [ ] MCMC over large DAG spaces (>10 nodes)
- [ ] Query identifiability certificates
- [ ] Real data integration (Sachs, DREAM4)
- [ ] Constraint-based methods (GES, FCI)
- [ ] Visualization of DAG uncertainty

### Long-term

- [ ] Amortized policy learning (neural network)
- [ ] Multi-agent BOED (collaborative experimentation)
- [ ] Safety-aware BOED (causal discovery with constraints)
- [ ] Nonlinear discovery (splines, kernels, deep networks)

## Limitations

1. **Linear Gaussian assumptions** (BGe score assumes this)
   - Nonlinear ANM sampling works but scored with BIC (weaker)
   - Extension: implement nonlinear scorers

2. **Small graphs** (enumeration-based, max ~6 nodes)
   - Extension: MCMC sampling for DAG space

3. **EIG approximation** (edge uncertainty heuristic, not true expectation)
   - Extension: Monte Carlo EIG (slower but more accurate)

4. **No amortization** (recompute policy per round)
   - Extension: Learn amortized policy via RL

## Citation

If you use this framework, please cite:

```bibtex
@software{causal_boed_2024,
  title={Causal BOED: Identifiability-Aware Experimental Design},
  author={Your Name},
  year={2024}
}
```

## References

### Key Papers

- **Tong & Koller (2001)**: Active learning for structure learning in DBNs
- **Eberhardt et al. (2012)**: On optimal experimental designs for causal discovery
- **Scutari et al. (2016)**: Identifying Markov boundaries with the PC algorithm
- **Geiger & Heckerman (1994)**: Learning Gaussian networks (BGe score)

### Related Work

- **causal-learn**: Constraint-based discovery (PC, GES, FCI)
- **pyro-ppl**: Bayesian inference for complex models
- **numpyro**: Fast probabilistic programming

## License

MIT

## Contact

For questions or issues, please open an issue on GitHub.
