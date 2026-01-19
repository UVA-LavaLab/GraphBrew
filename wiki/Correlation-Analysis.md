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

```bash
# Run correlation analysis with quick test
python3 scripts/analysis/correlation_analysis.py --quick

# Full analysis on real graphs
python3 scripts/analysis/correlation_analysis.py \
    --graphs-dir ./graphs \
    --benchmark pr bfs
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
│ LeidenOrder (12)   │ 0.0398       │ 1.31x       │
│ LeidenHybrid (20)  │ 0.0371       │ 1.41x ★     │
│ RCM (11)           │ 0.0489       │ 1.07x       │
└────────────────────┴──────────────┴─────────────┘
Best: LeidenHybrid
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
│ facebook         │ 0.835  │ 0.42    │ 52.4    │ 0.011   │ LeidenHybrid  │
│ twitter          │ 0.721  │ 0.68    │ 891.2   │ 0.002   │ LeidenDFSHub  │
│ roadNet-CA       │ 0.112  │ 0.05    │ 1.2     │ 0.0001  │ RCM           │
│ web-Google       │ 0.654  │ 0.55    │ 234.5   │ 0.008   │ HUBCLUSTERDBG │
│ citation         │ 0.443  │ 0.31    │ 45.6    │ 0.003   │ LeidenOrder   │
│ amazon           │ 0.926  │ 0.18    │ 12.3    │ 0.0004  │ LeidenOrder   │
│ youtube          │ 0.712  │ 0.52    │ 289.1   │ 0.001   │ LeidenHybrid  │
│ livejournal      │ 0.758  │ 0.61    │ 567.8   │ 0.0003  │ LeidenDFSHub  │
└──────────────────┴────────┴─────────┴─────────┴─────────┴───────────────┘
```

### Step 4: Compute Correlations

Calculate Pearson correlation between each feature and "algorithm X being best":

```
For LeidenHybrid:
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

Convert correlations to weights:

```python
def correlation_to_weight(r, scale=0.35):
    """Convert Pearson r to perceptron weight."""
    # Scale and clip to reasonable range
    return max(-0.3, min(0.3, r * scale))

weights = {
    "LeidenHybrid": {
        "bias": 0.85,  # Base preference (from win rate)
        "w_modularity": 0.78 * 0.35,      # = 0.27
        "w_hub_concentration": 0.45 * 0.35,  # = 0.16
        "w_degree_variance": 0.52 * 0.35,    # = 0.18
        "w_density": -0.23 * 0.35,           # = -0.08
    },
    "RCM": {
        "bias": 0.55,
        "w_modularity": -0.65 * 0.35,     # = -0.23
        "w_hub_concentration": -0.72 * 0.35, # = -0.25
        ...
    }
}
```

---

## Running Analysis

### correlation_analysis.py

```bash
python3 scripts/analysis/correlation_analysis.py [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--graphs-dir` | Directory with graphs | `./graphs` |
| `--benchmark` | Benchmarks to analyze | `pr bfs` |
| `--algorithms` | Algorithm IDs | `0,7,12,15,16-20` |
| `--output` | Output weights file | `scripts/perceptron_weights.json` |
| `--quick` | Quick test with synthetic graphs | False |
| `--no-benchmark` | Use existing results only | False |
| `--verbose` | Show detailed output | False |

### Examples

#### Quick Test
```bash
python3 scripts/analysis/correlation_analysis.py \
    --quick \
    --algorithms "0,7,12,15,20"
```

#### Full Analysis
```bash
# First download graphs
python3 scripts/download/download_graphs.py --size MEDIUM --output-dir ./graphs

# Run analysis
python3 scripts/analysis/correlation_analysis.py \
    --graphs-dir ./graphs \
    --benchmark pr bfs cc \
    --algorithms "0,5,7,11,12,15,16,17,18,19,20" \
    --verbose
```

#### Reanalyze Existing Results
```bash
python3 scripts/analysis/correlation_analysis.py \
    --no-benchmark \
    --output new_weights.json
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
  facebook.el: ORIGINAL=0.052s, HUBCLUSTERDBG=0.041s, LeidenHybrid=0.037s ★
  twitter.el: ORIGINAL=12.3s, LeidenDFSHub=8.1s ★, LeidenHybrid=8.4s
  ...

Extracting features...
  facebook.el: mod=0.835, hc=0.42, dv=52.4
  twitter.el: mod=0.721, hc=0.68, dv=891.2
  ...

Computing correlations...
  LeidenHybrid × modularity: r=0.78 (strong positive)
  LeidenHybrid × hub_concentration: r=0.45 (moderate positive)
  ...

----------------------------------------------------------------------
Computing Perceptron Weights
----------------------------------------------------------------------
Perceptron weights saved to: scripts/perceptron_weights.json
  20 algorithms configured
  Updated from benchmarks: ORIGINAL, HUBCLUSTERDBG, LeidenHybrid, ...

Summary:
  Total graphs analyzed: 8
  Algorithms benchmarked: 11
  Best overall: LeidenHybrid (won 4/8 graphs)
  Second best: LeidenDFSHub (won 2/8 graphs)
```

### Generated Files

1. **perceptron_weights.json** - Weights for C++ runtime
2. **correlation_matrix.csv** - Raw correlation data
3. **benchmark_results.json** - Full benchmark results

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
LeidenHybrid:
  r(modularity) = +0.78
  → LeidenHybrid works best on highly modular graphs
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

### Generate Correlation Heatmap

```bash
python3 scripts/analysis/correlation_analysis.py \
    --graphs-dir ./graphs \
    --plot-heatmap correlation_heatmap.png
```

Output:

```
                  ORIG  HUBCDB  RCM   LeiOrd  LeiHyb
modularity        -0.12  0.34   -0.65  0.56    0.78
hub_concentration -0.08  0.67   -0.72  0.41    0.45
degree_variance    0.02  0.45   -0.58  0.38    0.52
density            0.15 -0.23    0.41 -0.18   -0.23
log_edges         -0.31  0.21   -0.15  0.28    0.38
```

### Generate Performance Chart

```bash
python3 scripts/analysis/correlation_analysis.py \
    --graphs-dir ./graphs \
    --plot-performance performance_chart.png
```

---

## Updating Weights

### When to Re-run Analysis

1. **Added new graphs** to your benchmark set
2. **Added new algorithms** to GraphBrew
3. **Changed target benchmarks** (pr vs bfs vs sssp)
4. **Noticed poor predictions** in production

### Incremental Update

```bash
# Keep existing weights, only update from new data
python3 scripts/analysis/correlation_analysis.py \
    --graphs-dir ./new_graphs \
    --incremental
```

### Reset to Defaults

```bash
# Regenerate from scratch
rm scripts/perceptron_weights.json
python3 scripts/analysis/correlation_analysis.py \
    --graphs-dir ./graphs
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
python3 scripts/analysis/correlation_analysis.py \
    --trials 10
```

### 3. Target Your Workload

If you mostly run PageRank:
```bash
python3 scripts/analysis/correlation_analysis.py \
    --benchmark pr \  # Focus on PageRank
    --algorithms "0,7,12,20"
```

If you run mixed workloads:
```bash
python3 scripts/analysis/correlation_analysis.py \
    --benchmark pr bfs cc sssp \  # All benchmarks
    --algorithms "0,7,12,15,16,17,18,19,20"
```

---

## Troubleshooting

### "No significant correlations found"

**Cause**: Not enough diversity in graphs or algorithms

**Fix**:
```bash
# Add more graphs
python3 scripts/download/download_graphs.py --size MEDIUM --output-dir ./graphs

# Re-run with more algorithms
python3 scripts/analysis/correlation_analysis.py \
    --algorithms "0,1,2,3,4,5,6,7,8,9,10,11,12,15,16,17,18,19,20"
```

### "Weights look wrong"

**Check correlation matrix**:
```bash
cat correlation_matrix.csv
```

**Verify benchmark data**:
```bash
python3 -c "
import json
with open('benchmark_results.json') as f:
    data = json.load(f)
    for g, r in data.items():
        best = min(r.items(), key=lambda x: x[1]['mean'])
        print(f'{g}: {best[0]} ({best[1][\"mean\"]:.4f}s)')
"
```

---

## Full Correlation Scan

For comprehensive benchmarking of all algorithms across all graphs, use the full correlation scan script. This is the recommended way to generate complete benchmark data for perceptron weight training.

### Running Full Scan

```bash
# Quick test with synthetic graphs
python3 scripts/analysis/full_correlation_scan.py --quick

# SMALL graphs only (~12MB)
python3 scripts/analysis/full_correlation_scan.py --small

# SMALL + MEDIUM graphs (~600MB, recommended)
python3 scripts/analysis/full_correlation_scan.py --medium --trials 3

# All graphs including LARGE (~72GB, full experiments)
python3 scripts/analysis/full_correlation_scan.py --full --trials 16
```

### Key Features

- **Sequential Execution**: One benchmark at a time for full CPU utilization
- **RANDOM Baseline**: Uses RANDOM (1) for all speedup calculations
- **All Algorithms**: Tests IDs 0-12, 15-20 (skips 13=GraphBrew, 14=MAP)
- **Incremental Save**: Progress saved to allow resumption on interruption
- **Skip TC**: Triangle counting excluded (reordering doesn't help)

### Results Directory

Results are saved to `./results/` by default:

```
results/
├── logs/
│   └── correlation_scan.log   # Detailed execution log
├── scan_results.json          # All benchmark results (JSON)
├── correlation_matrix.csv     # Feature-algorithm correlations
└── summary_report.txt         # Human-readable summary
```

### Resume Interrupted Scan

If a scan is interrupted, simply re-run with the same parameters:

```bash
# Resume automatically picks up where it left off
python3 scripts/analysis/full_correlation_scan.py --medium --resume
```

### Output Format

The `scan_results.json` contains structured results:

```json
{
  "metadata": {
    "start_time": "2025-01-19T02:00:00",
    "graphs_completed": 20,
    "total_runs": 1900
  },
  "results": {
    "facebook": {
      "pr": {
        "1": {"time": 0.052, "speedup": 1.0},
        "20": {"time": 0.037, "speedup": 1.41}
      }
    }
  },
  "best_algorithms": {
    "pr": {"facebook": "LeidenHybrid", "twitter": "LeidenDFSHub"},
    "bfs": {"facebook": "RABBITORDER", "roadNet": "RCM"}
  }
}
```

---

## Mathematical Details

### Perceptron Score Function

```
score(algo, community) = bias_algo + Σ(w_feature × feature_value)
```

### Weight Derivation

```python
# For each algorithm
for algo in algorithms:
    # Get graphs where this algo was best
    wins = [g for g in graphs if best_algo[g] == algo]
    
    # Compute feature correlations
    for feature in features:
        # Binary: 1 if algo won, 0 otherwise
        y = [1 if g in wins else 0 for g in graphs]
        x = [feature_value[g][feature] for g in graphs]
        
        r = pearson_correlation(x, y)
        weights[algo][f"w_{feature}"] = r * SCALE_FACTOR
    
    # Bias from win rate
    weights[algo]["bias"] = 0.3 + (len(wins) / len(graphs)) * 0.7
```

---

## Next Steps

- [[Perceptron-Weights]] - Understanding and tuning weights
- [[AdaptiveOrder-ML]] - Using the trained model
- [[Benchmark-Suite]] - Running benchmark experiments

---

[← Back to Home](Home) | [Perceptron Weights →](Perceptron-Weights)
