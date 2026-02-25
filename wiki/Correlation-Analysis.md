# Correlation Analysis

Correlation analysis is the process of finding which reordering algorithm works best for graphs with specific structural features. This page explains the analysis process and how to run it.

## Overview

The correlation analysis:
1. **Benchmarks** multiple algorithms on many graphs
2. **Extracts features** from each graph
3. **Correlates** features with algorithm performance
4. **Generates weights** for the perceptron model

```
Graphs → Benchmark → Features → Correlation → Perceptron Weights
  ↓         ↓           ↓          ↓              ↓
Many     Run all    Extract    Pearson r    JSON config
graphs   algos      metrics    analysis     for C++
```

---

## Quick Start

The unified experiment script handles all correlation analysis:

```bash
# Run full pipeline (includes correlation analysis)
python3 scripts/graphbrew_experiment.py --full --size small

# Train weights with complete pipeline
python3 scripts/graphbrew_experiment.py --train --size small

# Phase-based: run only weights generation
python3 scripts/graphbrew_experiment.py --phase weights --size small
```

---

## How It Works

### Step 1: Benchmark Collection

For each graph, run all candidate algorithms and record execution times:

```
Graph: example.el (N nodes, M edges)
┌────────────────────┬──────────────┬─────────────┐
│ Algorithm          │ Time (sec)   │ Speedup     │
├────────────────────┼──────────────┼─────────────┤
│ ORIGINAL (0)       │ X.XXXXs      │ 1.00x       │
│ HUBCLUSTERDBG (7)  │ X.XXXXs      │ X.XXx       │
│ LeidenOrder (15)   │ X.XXXXs      │ X.XXx       │
│ GraphBrewOrder (12) │ X.XXXXs      │ X.XXx ★     │
│ RCM (11)           │ X.XXXXs      │ X.XXx       │
└────────────────────┴──────────────┴─────────────┘
Best: (depends on graph topology)
```

### Step 2: Feature Extraction

Compute structural features for each graph:

```python
features = {
    # Size features
    "num_nodes": ...,   # from graph
    "num_edges": ...,   # from graph
    "log_nodes": ...,   # log10(num_nodes)
    "log_edges": ...,   # log10(num_edges)
    
    # Density features
    "density": ...,     # edges / max_possible_edges
    "avg_degree": ...,  # 2 * edges / nodes
    
    # Structure features
    "modularity": ...,         # Leiden modularity score
    "degree_variance": ...,    # Variance in degree distribution
    "hub_concentration": ...,  # Edge fraction to top 10% nodes
    "clustering_coeff": ...,   # Average clustering coefficient
}
```

These features are computed automatically by the pipeline. Run `--phase benchmark` on your graphs to populate them.

### Step 3: Build Feature-Performance Matrix

Combine results across all graphs into a matrix mapping graph features to the best-performing algorithm:

```
┌──────────────────┬────────┬─────────┬─────────┬─────────┬───────────────┐
│ Graph            │ ModQ   │ HubConc │ DegVar  │ Density │ Best Algo     │
├──────────────────┼────────┼─────────┼─────────┼─────────┼───────────────┤
│ graph_1          │ ...    │ ...     │ ...     │ ...     │ ...           │
│ graph_2          │ ...    │ ...     │ ...     │ ...     │ ...           │
│ ...              │ ...    │ ...     │ ...     │ ...     │ ...           │
└──────────────────┴────────┴─────────┴─────────┴─────────┴───────────────┘
```

### Step 4: Compute Correlations

Calculate Pearson correlation between each feature and "algorithm X being best":

```
For each algorithm:
┌─────────────────────┬─────────────┬──────────────────────────────────────┐
│ Feature             │ Pearson r   │ Interpretation                       │
├─────────────────────┼─────────────┼──────────────────────────────────────┤
│ modularity          │ +/- X.XX    │ Community structure affinity          │
│ hub_concentration   │ +/- X.XX    │ Hub handling ability                  │
│ degree_variance     │ +/- X.XX    │ Degree skew sensitivity              │
│ log_edges           │ +/- X.XX    │ Scale dependency                     │
│ density             │ +/- X.XX    │ Sparsity preference                  │
└─────────────────────┴─────────────┴──────────────────────────────────────┘
```

### Step 5: Generate Perceptron Weights

Correlations are converted to weights via `compute_weights_from_results()` — a multi-stage pipeline (multi-restart perceptrons → variant-level saving → regret-aware grid search). See [[Perceptron-Weights#multi-restart-training--benchmark-multipliers]] for details.

> **Note:** The older correlation-to-weight approach (`r × scale`) has been superseded by multi-restart perceptron training which produces more robust weights.

---

## Running Analysis

### Using graphbrew_experiment.py (Recommended)

The unified experiment script provides all correlation analysis functionality:

```bash
python3 scripts/graphbrew_experiment.py [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--full` | Run complete pipeline | False |
| `--phase weights` | Run only weights generation | - |
| `--train` | Comprehensive weight training | False |
| `--size` | Graph set: small, medium, large, xlarge, all | all |
| `--skip-cache` | Skip cache simulation (faster) | False |
| `--adaptive-analysis` | Show algorithm distribution | False |

### Examples

#### Quick Test
```bash
python3 scripts/graphbrew_experiment.py --full --size small --skip-cache
```

#### Full Analysis
```bash
# Download and run complete analysis
python3 scripts/graphbrew_experiment.py --full --size medium

# Or run phases separately
python3 scripts/graphbrew_experiment.py --phase benchmark --size medium
python3 scripts/graphbrew_experiment.py --phase weights
```

#### Comprehensive Weight Training
```bash
python3 scripts/graphbrew_experiment.py --train --size all --auto
```

---

## Output

### Console Output

```
======================================================================
GraphBrew Correlation Analysis
======================================================================

Loading graphs from: ./graphs
Found N graphs

Running benchmarks...
  graph1: ORIGINAL=X.XXXs, HUBCLUSTERDBG=X.XXXs, GraphBrewOrder=X.XXXs ★
  graph2: ORIGINAL=X.XXXs, GraphBrewOrder=X.XXXs ★, ...
  ...

Extracting features...
  graph1: mod=X.XX, hc=X.XX, dv=X.X
  ...

Computing correlations...
  ...

----------------------------------------------------------------------
Computing Perceptron Weights
----------------------------------------------------------------------
Perceptron weights saved to: results/models/perceptron/type_0/weights.json
  16 algorithms configured (IDs 0-15; 14 are benchmark-eligible)

Summary:
  Total graphs analyzed: N
  Algorithms benchmarked: M
  Best overall: (depends on your graph set)
```

### Generated Files

1. **results/models/perceptron/type_N/weights.json** - Weights for C++ runtime (per cluster)
2. **results/models/perceptron/registry.json** - Graph → type mappings + centroids
3. **results/benchmark_*.json** - Full benchmark results

---

## Understanding Correlations

### Pearson Correlation Coefficient

```
r = Σ((x - x̄)(y - ȳ)) / (n × σx × σy)

Where:
  x = feature values across graphs
  y = 1 if algorithm was best, 0 otherwise
  r ranges from -1 to +1
```

### Interpreting Results

| Correlation | Meaning |
|-------------|---------|
| r > 0.7 | Strong positive - algorithm excels when feature is high |
| 0.3 < r < 0.7 | Moderate positive |
| -0.3 < r < 0.3 | Weak - feature doesn't predict performance |
| -0.7 < r < -0.3 | Moderate negative |
| r < -0.7 | Strong negative - algorithm struggles when feature is high |

### Example Interpretation

Positive correlation between a feature and an algorithm means the algorithm performs well when that feature is high. For example:
- Community-aware algorithms tend to correlate positively with modularity
- RCM tends to correlate negatively with hub concentration (it optimizes bandwidth, not hub locality)

Run the training pipeline on your data to see the actual correlations.

---

## Feature Definitions

### Size Features

| Feature | Formula | Purpose |
|---------|---------|---------|
| `log_nodes` | log₁₀(|V|) | Graph scale (vertices) |
| `log_edges` | log₁₀(|E|) | Graph scale (edges) |

### Density Features

| Feature | Formula | Purpose |
|---------|---------|---------|
| `density` | 2|E| / (|V|(|V|-1)) | Sparsity measure |
| `avg_degree` | 2|E| / |V| | Average connectivity |

### Structure Features

| Feature | Formula | Purpose |
|---------|---------|---------|
| `modularity` | Leiden Q score | Community structure strength |
| `degree_variance` | Var(degrees) | Degree distribution spread |
| `hub_concentration` | Edges to top 10% / Total | Hub dominance |

---

## Visualizing Results

Results are saved to `results/` and can be analyzed using standard tools.

### Benchmark Results

After running the full pipeline, results are in JSON format:

```bash
# View benchmark results
cat results/benchmark_*.json | python3 -m json.tool

# View generated weights
cat results/models/perceptron/type_0/weights.json | python3 -m json.tool
```

### Correlation Matrix

The weights reflect feature-algorithm correlations. Run `--eval-weights` to see the actual correlation values for your training data.

---

## Updating Weights

Re-run analysis when: adding new graphs, adding algorithms, changing target benchmarks, or noticing poor predictions.

```bash
python3 scripts/graphbrew_experiment.py --train --size medium --auto
```

After training, the system automatically clusters graphs and generates per-cluster weights in `results/models/perceptron/`. See [[Perceptron-Weights]] for weight file format and [[Python-Scripts]] for weight management.

To reset: `rm -rf results/models/perceptron/type_*/weights.json results/models/perceptron/registry.json`

---

## Best Practices

### 1. Use Diverse Graphs

Include graphs with different characteristics:
- Social networks (high modularity, hubs)
- Road networks (low modularity, sparse)
- Web graphs (scale-free, hubs)
- Citation networks (moderate modularity)

### 2. Run Sufficient Trials

```bash
# Use at least 5-10 trials for stable measurements
python3 scripts/graphbrew_experiment.py --full --size medium
```

### 3. Target Your Workload

For specific benchmarks:
```bash
# Focus on specific benchmarks
python3 scripts/graphbrew_experiment.py --phase benchmark --benchmarks pr bfs
```

---

## Troubleshooting

### "No significant correlations found"

**Cause**: Not enough diversity in graphs or algorithms

**Fix**:
```bash
# Download more graphs
python3 scripts/graphbrew_experiment.py --download-only --size medium

# Re-run full experiment
python3 scripts/graphbrew_experiment.py --full --size medium
```

### "Weights look wrong"

**Check benchmark results**:
```bash
python3 -c "
import json
for f in ['results/benchmark_*.json']:
    import glob
    for path in glob.glob(f):
        with open(path) as fp:
            data = json.load(fp)
            print(f'{path}: {len(data)} results')
"
```

---

## Full Correlation Scan

See [[Benchmark-Suite]] for running complete experiments across all graph sizes, and [[Command-Line-Reference]] for all pipeline options.

---

## Mathematical Details

The perceptron score function, multi-stage training pipeline, and validation metrics are documented in [[Perceptron-Weights#score-calculation]] and [[AdaptiveOrder-ML#advanced-training-compute_weights_from_results]].

---

## Next Steps

- [[Perceptron-Weights]] - Understanding and tuning weights
- [[AdaptiveOrder-ML]] - Using the trained model
- [[Benchmark-Suite]] - Running benchmark experiments

---

[← Back to Home](Home) | [Perceptron Weights →](Perceptron-Weights)
