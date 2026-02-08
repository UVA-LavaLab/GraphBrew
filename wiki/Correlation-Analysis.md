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
Graph: facebook.el (4,039 nodes, 88,234 edges)
┌────────────────────┬──────────────┬─────────────┐
│ Algorithm          │ Time (sec)   │ Speedup     │
├────────────────────┼──────────────┼─────────────┤
│ ORIGINAL (0)       │ 0.0523       │ 1.00x       │
│ HUBCLUSTERDBG (7)  │ 0.0412       │ 1.27x       │
│ LeidenOrder (15)   │ 0.0398       │ 1.31x       │
│ LeidenCSR (17)     │ 0.0371       │ 1.41x ★     │
│ RCM (11)           │ 0.0489       │ 1.07x       │
└────────────────────┴──────────────┴─────────────┘
Best: LeidenCSR
```

### Step 2: Feature Extraction

Compute structural features for each graph:

```python
features = {
    # Size features
    "num_nodes": 4039,
    "num_edges": 88234,
    "log_nodes": 3.606,  # log10(4039)
    "log_edges": 4.946,  # log10(88234)
    
    # Density features
    "density": 0.0108,   # edges / max_possible_edges
    "avg_degree": 43.7,  # 2 * edges / nodes
    
    # Structure features
    "modularity": 0.835,        # Leiden modularity score
    "degree_variance": 52.4,    # Variance in degree distribution
    "hub_concentration": 0.42,  # Edge fraction to top 10% nodes
    "clustering_coeff": 0.606,  # Average clustering coefficient
}
```

### Step 3: Build Feature-Performance Matrix

Combine results across all graphs:

```
┌──────────────────┬────────┬─────────┬─────────┬─────────┬───────────────┐
│ Graph            │ ModQ   │ HubConc │ DegVar  │ Density │ Best Algo     │
├──────────────────┼────────┼─────────┼─────────┼─────────┼───────────────┤
│ facebook         │ 0.835  │ 0.42    │ 52.4    │ 0.011   │ LeidenCSR  │
│ twitter          │ 0.721  │ 0.68    │ 891.2   │ 0.002   │ LeidenDendrogram  │
│ roadNet-CA       │ 0.112  │ 0.05    │ 1.2     │ 0.0001  │ RCM           │
│ web-Google       │ 0.654  │ 0.55    │ 234.5   │ 0.008   │ HUBCLUSTERDBG │
│ citation         │ 0.443  │ 0.31    │ 45.6    │ 0.003   │ LeidenOrder   │
│ amazon           │ 0.926  │ 0.18    │ 12.3    │ 0.0004  │ LeidenOrder   │
│ youtube          │ 0.712  │ 0.52    │ 289.1   │ 0.001   │ LeidenCSR  │
│ livejournal      │ 0.758  │ 0.61    │ 567.8   │ 0.0003  │ LeidenDendrogram  │
└──────────────────┴────────┴─────────┴─────────┴─────────┴───────────────┘
```

### Step 4: Compute Correlations

Calculate Pearson correlation between each feature and "algorithm X being best":

```
For LeidenCSR:
┌─────────────────────┬─────────────┬──────────────────────────────────────┐
│ Feature             │ Pearson r   │ Interpretation                       │
├─────────────────────┼─────────────┼──────────────────────────────────────┤
│ modularity          │ +0.78       │ Strong: prefers modular graphs       │
│ hub_concentration   │ +0.45       │ Moderate: handles hubs well          │
│ degree_variance     │ +0.52       │ Moderate: handles degree skew        │
│ log_edges           │ +0.38       │ Weak: slightly better on larger      │
│ density             │ -0.23       │ Weak negative: prefers sparse        │
└─────────────────────┴─────────────┴──────────────────────────────────────┘

For RCM:
┌─────────────────────┬─────────────┬──────────────────────────────────────┐
│ Feature             │ Pearson r   │ Interpretation                       │
├─────────────────────┼─────────────┼──────────────────────────────────────┤
│ modularity          │ -0.65       │ Strong neg: bad on modular graphs    │
│ hub_concentration   │ -0.72       │ Strong neg: doesn't handle hubs      │
│ degree_variance     │ -0.58       │ Moderate neg: uniform degree best    │
│ log_edges           │ -0.15       │ Weak: size doesn't matter much       │
│ density             │ +0.41       │ Moderate: prefers denser graphs      │
└─────────────────────┴─────────────┴──────────────────────────────────────┘
```

### Step 5: Generate Perceptron Weights

The `compute_weights_from_results()` function converts benchmark data into perceptron weights using a multi-stage process:

1. **Multi-restart perceptron training** (`N_RESTARTS=5`, `N_EPOCHS=800` per benchmark)
2. **Z-score feature normalization** for stable SGD gradients
3. **Variant pre-collapse**: keeps only the highest-bias variant per base algorithm
4. **Regret-aware grid search** for per-benchmark multipliers (30 iterations × 32 values)

The final weights reflect both feature correlations and benchmark-specific performance:

```python
weights = compute_weights_from_results(
    benchmark_results=bench_results,
    reorder_results=reorder_results,
    weights_dir="scripts/weights/active",
)
# Produces type_0.json with scoreBase weights + benchmark_weights multipliers
```

**Example result** (simplified):
```json
{
  "LeidenCSR": {
    "bias": 0.85,
    "w_modularity": 0.27,
    "w_hub_concentration": 0.16,
    "w_degree_variance": 0.18,
    "w_density": -0.08,
    "benchmark_weights": {
      "pr": 1.2,
      "bfs": 0.95,
      "cc": 1.1,
      "sssp": 1.05
    }
  }
}
```

> **Note:** The older correlation-to-weight approach (`r × scale`) has been superseded by multi-restart perceptron training which produces more robust weights. The correlation coefficients shown above remain useful for understanding *why* certain algorithms perform well on certain graph types.

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
Found 8 graphs

Running benchmarks...
  facebook.el: ORIGINAL=0.052s, HUBCLUSTERDBG=0.041s, LeidenCSR=0.037s ★
  twitter.el: ORIGINAL=12.3s, LeidenDendrogram=8.1s ★, LeidenCSR=8.4s
  ...

Extracting features...
  facebook.el: mod=0.835, hc=0.42, dv=52.4
  twitter.el: mod=0.721, hc=0.68, dv=891.2
  ...

Computing correlations...
  LeidenCSR × modularity: r=0.78 (strong positive)
  LeidenCSR × hub_concentration: r=0.45 (moderate positive)
  ...

----------------------------------------------------------------------
Computing Perceptron Weights
----------------------------------------------------------------------
Perceptron weights saved to: scripts/weights/active/type_0.json
  18 algorithms configured
  Updated from benchmarks: ORIGINAL, HUBCLUSTERDBG, LeidenCSR, ...

Summary:
  Total graphs analyzed: 8
  Algorithms benchmarked: 11
  Best overall: LeidenCSR (won 4/8 graphs)
  Second best: LeidenDendrogram (won 2/8 graphs)
```

### Generated Files

1. **scripts/weights/active/type_N.json** - Weights for C++ runtime (per cluster)
2. **scripts/weights/active/type_registry.json** - Graph → type mappings + centroids
3. **correlation_matrix.csv** - Raw correlation data
4. **benchmark_results.json** - Full benchmark results

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

```
LeidenCSR:
  r(modularity) = +0.78
  → LeidenCSR works best on highly modular graphs
  → Makes sense: it's designed to exploit community structure

RCM:
  r(hub_concentration) = -0.72
  → RCM performs poorly on hub-dominated graphs
  → Makes sense: RCM optimizes bandwidth, not hub locality
```

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
cat scripts/weights/active/type_0.json | python3 -m json.tool
```

### Correlation Matrix

The weights reflect feature correlations:

```
                  ORIG  HUBCDB  RCM   LeiOrd  LeiHyb
modularity        -0.12  0.34   -0.65  0.56    0.78
hub_concentration -0.08  0.67   -0.72  0.41    0.45
degree_variance    0.02  0.45   -0.58  0.38    0.52
density            0.15 -0.23    0.41 -0.18   -0.23
log_edges         -0.31  0.21   -0.15  0.28    0.38
```

---

## Updating Weights

### When to Re-run Analysis

1. **Added new graphs** to your benchmark set
2. **Added new algorithms** to GraphBrew
3. **Changed target benchmarks** (pr vs bfs vs sssp)
4. **Noticed poor predictions** in production

### Using --train for Comprehensive Updates

If many weight fields are 0 or default 1.0, use the training mode:

```bash
# Complete training: cache impacts, topology features, and per-graph-type weights
python3 scripts/graphbrew_experiment.py \
    --train \
    --size medium \
    --auto
```

This runs phases sequentially:

| Phase | Description | Weight Fields Updated |
|-------|-------------|----------------------|
| **Phase 1** | Reordering | `w_reorder_time` via reorder timings |
| **Phase 2** | Benchmarks | `bias`, win rates from benchmark results |
| **Phase 3** | Cache simulation | `cache_l1_impact`, `cache_l2_impact`, `cache_l3_impact` |
| **Phase 4** | Weight generation | Core weights from correlations |
| **Phase 5** | Fill weights | Update zero fields from results |

**Output files:**
```
scripts/weights/
├── active/                   # Active weights (C++ reads, Python writes)
│   ├── type_registry.json    # Maps graphs → types + cluster centroids
│   ├── type_0.json           # Cluster 0 weights
│   ├── type_1.json           # Cluster 1 weights
│   └── type_N.json           # Additional clusters
├── merged/                   # Accumulated weights from all runs
└── runs/                     # Historical snapshots

results/
└── graph_properties_cache.json  # Cached graph properties for type detection
```

### Automatic Clustering

After each weight update, the system automatically:
1. **Clusters graphs** by feature similarity
2. **Generates per-cluster weights** in `scripts/weights/active/type_N.json`
3. **Updates type registry** with centroids for runtime matching

### Reset to Defaults

```bash
# Regenerate from scratch
rm -rf scripts/weights/active/type_*.json scripts/weights/active/type_registry.json
python3 scripts/graphbrew_experiment.py --train --size small
```

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

For comprehensive benchmarking, use the unified experiment script:

### Running Full Scan

```bash
# Quick test with small graphs
python3 scripts/graphbrew_experiment.py --full --size small

# medium graphs (recommended for development)
python3 scripts/graphbrew_experiment.py --full --size medium

# All graphs with automatic resource management
python3 scripts/graphbrew_experiment.py --full --size all --auto
```

### Key Features

- **Sequential Execution**: One benchmark at a time for full CPU utilization
- **Speedup Baseline**: Uses ORIGINAL for adaptive analysis, RANDOM for general speedup calculations
- **All Algorithms**: Tests IDs 0-17 (typically skips 13=MAP which needs external file)
- **Incremental Save**: Progress saved to allow resumption on interruption

### Results Directory

Results are saved to `./results/` by default:

```
results/
├── mappings/              # Pre-generated label maps
├── reorder_*.json         # Reordering times
├── benchmark_*.json       # All benchmark results
├── cache_*.json           # Cache simulation results
└── logs/                  # Execution logs
```

### Resume Interrupted Scan

If a scan is interrupted, re-run with the same parameters - the script will skip completed work:

```bash
# Resume automatically
python3 scripts/graphbrew_experiment.py --full --size medium
```

### Output Format

The `benchmark_*.json` contains an array of results:

```json
[
  {
    "graph": "facebook",
    "algorithm": "HUBCLUSTERDBG",
    "algorithm_id": 7,
    "benchmark": "pr",
    "time_seconds": 0.037,
    "reorder_time": 0.012,
    "trials": 2,
    "success": true,
    "error": "",
    "extra": {}
  }
]
```

---

## Mathematical Details

### Perceptron Score Function

```
score(algo, community) = bias_algo + Σ(w_feature × feature_value)
```

### Weight Derivation

```python
# compute_weights_from_results() multi-stage pipeline:

# Stage 1: Multi-restart perceptrons (5 restarts × 800 epochs per benchmark)
for bench in ['pr', 'bfs', 'cc', 'sssp']:
    for restart in range(5):
        seed = 42 + restart * 1000 + bench_index * 100
        # Z-score normalize features, SGD with L2 decay (1e-4)
        perceptron = train_perceptron(data, seed, epochs=800)
    avg_weights[bench] = average(all_restarts)

# Stage 2: Average across benchmarks for scoreBase()
scoreBase_weights = average(avg_weights.values())

# Stage 3: Pre-collapse variants (keep highest-bias per base algo)
# Stage 4: Regret-aware grid search for benchmark_weights multipliers
#   30 iterations, 32 log-spaced values [0.1, 10.0]
#   Objective: max(accuracy, min(-mean_regret))

# Bias from win rate (capped at 1.5)
weights[algo]["bias"] = min(1.5, 0.3 + (len(wins) / len(graphs)) * 0.7)

# L2 regularization applied after each SGD update (decay = 1e-4)
```

**Validation:** After training, use `eval_weights.py` to simulate C++ scoring and measure accuracy/regret. Current results: 46.8% accuracy, 2.6% base-aware median regret on 47 graphs × 4 benchmarks.

---

## Next Steps

- [[Perceptron-Weights]] - Understanding and tuning weights
- [[AdaptiveOrder-ML]] - Using the trained model
- [[Benchmark-Suite]] - Running benchmark experiments

---

[← Back to Home](Home) | [Perceptron Weights →](Perceptron-Weights)
