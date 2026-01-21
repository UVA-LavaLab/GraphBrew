# Perceptron Weights

The perceptron weights file controls how AdaptiveOrder selects algorithms for each community. This page explains the weight structure, tuning strategies, and how to customize the model.

## Overview

```
scripts/perceptron_weights.json
```

This JSON file contains weights for each algorithm. When AdaptiveOrder processes a community, it computes a score for each algorithm using these weights and selects the highest-scoring one.

---

## Weight File Location

### Default Location
```
GraphBrew/results/perceptron_weights.json
```

Note: The default path is defined by `DEFAULT_WEIGHTS_FILE` in `graphbrew_experiment.py` as `./results/perceptron_weights.json`.

### Environment Override
```bash
export PERCEPTRON_WEIGHTS_FILE=/path/to/custom_weights.json
./bench/bin/pr -f graph.el -s -o 15 -n 3
```

### Fallback Behavior
If the file doesn't exist, C++ uses hardcoded defaults with conservative weights that favor ORIGINAL for small communities and LeidenHybrid for larger ones.

---

## File Structure

### Complete Example (Enhanced Format)

```json
{
  "Original": {
    "bias": 0.5,
    "w_modularity": 0.1,
    "w_density": 0.05,
    "w_degree_variance": 0.03,
    "w_hub_concentration": 0.05,
    "w_log_nodes": 0.02,
    "w_log_edges": 0.02,
    "w_clustering_coeff": 0.04,
    "w_avg_path_length": 0.02,
    "w_diameter": 0.01,
    "w_community_count": 0.03,
    "benchmark_weights": {
      "pr": 1.0,
      "bfs": 1.0,
      "cc": 1.0,
      "sssp": 1.0,
      "bc": 1.0
    }
  },
  "LeidenDFS": {
    "bias": 3.5,
    "w_modularity": 0.1,
    "w_density": 0.05,
    "w_degree_variance": 0.03,
    "w_hub_concentration": 0.05,
    "w_log_nodes": 0.02,
    "w_log_edges": 0.02,
    "w_clustering_coeff": 0.04,
    "w_avg_path_length": 0.02,
    "w_diameter": 0.01,
    "w_community_count": 0.03,
    "benchmark_weights": {
      "pr": 1.0,
      "bfs": 1.0,
      "cc": 1.0,
      "sssp": 1.0,
      "bc": 1.0
    }
  },
  "_metadata": {
    "enhanced_features": true,
    "last_updated": "2026-01-20T12:00:00"
  }
}
```

Note: Algorithm names in the weights file use the format from `initialize_enhanced_weights()` (e.g., `Original`, `LeidenDFS`, `RabbitOrder`) rather than the C++ uppercase format.

---

## Weight Definitions

### Core Weights

| Weight | Feature | Description |
|--------|---------|-------------|
| `bias` | - | Base preference for algorithm (higher = more likely selected) |
| `w_modularity` | modularity | Leiden community quality score (0-1) |
| `w_log_nodes` | log₁₀(nodes) | Community size (vertices) |
| `w_log_edges` | log₁₀(edges) | Community size (edges) |
| `w_density` | edge density | Edges / max possible edges |
| `w_avg_degree` | avg_degree/100 | Mean vertex degree (normalized) |
| `w_degree_variance` | degree_var/100 | Degree distribution spread |
| `w_hub_concentration` | hub_conc | Edge fraction to top 10% vertices |

### Extended Graph Structure Weights

| Weight | Feature | Description |
|--------|---------|-------------|
| `w_clustering_coeff` | clustering_coefficient | Local clustering coefficient (0-1) |
| `w_avg_path_length` | avg_path_length | Average shortest path length (BFS estimate) |
| `w_diameter` | diameter_estimate | Graph diameter (BFS estimate) |
| `w_community_count` | community_count | Number of sub-communities |

### Per-Benchmark Weights

| Weight | Description |
|--------|-------------|
| `benchmark_weights.pr` | Multiplier for PageRank benchmark |
| `benchmark_weights.bfs` | Multiplier for BFS benchmark |
| `benchmark_weights.cc` | Multiplier for Connected Components benchmark |
| `benchmark_weights.sssp` | Multiplier for SSSP benchmark |
| `benchmark_weights.bc` | Multiplier for Betweenness Centrality benchmark |

### Cache Impact Weights (Optional)

| Weight | Description |
|--------|-------------|
| `cache_l1_impact` | Bonus for algorithms with high L1 hit rates |
| `cache_l2_impact` | Bonus for algorithms with high L2 hit rates |
| `cache_l3_impact` | Bonus for algorithms with high L3 hit rates |
| `cache_dram_penalty` | Penalty for DRAM access (cache misses) |

### Reorder Time Weight (Optional)

| Weight | Description |
|--------|-------------|
| `w_reorder_time` | Penalty for slow reordering (typically negative, e.g., -0.0001) |

### Metadata Fields (Auto-generated)

| Field | Description |
|-------|-------------|
| `win_rate` | Fraction of benchmarks where this algorithm was best |
| `avg_speedup` | Average speedup over ORIGINAL |
| `times_best` | Number of times this algorithm was optimal |
| `sample_count` | Number of graph/benchmark samples |
| `avg_reorder_time` | Average time to reorder (seconds) |
| `avg_l1_hit_rate` | Average L1 cache hit rate (%) |
| `avg_l2_hit_rate` | Average L2 cache hit rate (%) |
| `avg_l3_hit_rate` | Average L3 cache hit rate (%) |

---

## Score Calculation

### Formula

```
score = bias + Σ(w_feature × feature_value)
```

### Example Calculation

Community features:
```
modularity: 0.72
log_nodes: 3.5 (1000 nodes)
log_edges: 4.0 (10000 edges)
density: 0.02
avg_degree: 20 → normalized: 0.2
degree_variance: 45 → normalized: 0.45
hub_concentration: 0.55
```

LeidenHybrid score:
```
= 0.85                      # bias
+ 0.25 × 0.72               # modularity: +0.18
+ 0.10 × 3.5                # log_nodes: +0.35
+ 0.10 × 4.0                # log_edges: +0.40
+ (-0.05) × 0.02            # density: -0.001
+ 0.15 × 0.2                # avg_degree: +0.03
+ 0.15 × 0.45               # degree_variance: +0.0675
+ 0.25 × 0.55               # hub_concentration: +0.1375

= 0.85 + 0.18 + 0.35 + 0.40 - 0.001 + 0.03 + 0.0675 + 0.1375
= 2.01
```

---

## Algorithm Name Mapping

The JSON uses algorithm names, which map to IDs:

| ID | JSON Name | Description |
|----|-----------|-------------|
| 0 | ORIGINAL | No reordering |
| 1 | RANDOM | Random permutation |
| 2 | SORT | Sort by degree |
| 3 | HUBSORT | Hub sorting |
| 4 | HUBCLUSTER | Hub clustering |
| 5 | DBG | Degree-based grouping |
| 6 | HUBSORTDBG | Hub sort + DBG |
| 7 | HUBCLUSTERDBG | Hub cluster + DBG |
| 8 | RABBITORDER | Rabbit Order |
| 9 | GORDER | Gorder |
| 10 | CORDER | Corder |
| 11 | RCM | Reverse Cuthill-McKee |
| 12 | LeidenOrder | Basic Leiden ordering |
| 13 | GraphBrewOrder | GraphBrew composite |
| 15 | AdaptiveOrder | This perceptron model |
| 16 | LeidenDFS | Leiden + DFS |
| 17 | LeidenDFSHub | Leiden + DFS + Hub priority |
| 18 | LeidenDFSSize | Leiden + DFS + Size priority |
| 19 | LeidenBFS | Leiden + BFS |
| 20 | LeidenHybrid | Leiden + Hybrid traversal |

---

## Tuning Strategies

### Strategy 1: Favor One Algorithm

Make LeidenHybrid almost always win:

```json
{
  "LeidenHybrid": {
    "bias": 1.0,
    "w_modularity": 0.0,
    "w_log_nodes": 0.0,
    "w_log_edges": 0.0,
    "w_density": 0.0,
    "w_avg_degree": 0.0,
    "w_degree_variance": 0.0,
    "w_hub_concentration": 0.0
  }
}
```

### Strategy 2: Size-Based Selection

Use simpler algorithms for small communities:

```json
{
  "ORIGINAL": {
    "bias": 0.8,
    "w_log_nodes": -0.3,
    "w_log_edges": -0.3
  },
  "LeidenHybrid": {
    "bias": 0.5,
    "w_log_nodes": 0.2,
    "w_log_edges": 0.2
  }
}
```

Result:
- Small communities (< 100 nodes): ORIGINAL wins
- Large communities (> 1000 nodes): LeidenHybrid wins

### Strategy 3: Structure-Based Selection

Use hub algorithms for hub-heavy graphs:

```json
{
  "HUBCLUSTERDBG": {
    "bias": 0.6,
    "w_hub_concentration": 0.5,
    "w_degree_variance": 0.3
  },
  "RCM": {
    "bias": 0.6,
    "w_hub_concentration": -0.4,
    "w_degree_variance": -0.3
  }
}
```

Result:
- High hub concentration: HUBCLUSTERDBG wins
- Low hub concentration: RCM wins

### Strategy 4: Workload-Specific

For PageRank (benefits from hub locality):

```json
{
  "LeidenHybrid": {
    "bias": 0.85,
    "w_hub_concentration": 0.35
  }
}
```

For BFS (benefits from bandwidth reduction):

```json
{
  "RCM": {
    "bias": 0.75,
    "w_density": 0.2
  }
}
```

---

## Generating Weights

### Automatic: Correlation Analysis

```bash
python3 scripts/analysis/correlation_analysis.py \
    --graphs-dir ./graphs \
    --benchmark pr bfs \
    --output scripts/perceptron_weights.json
```

### Manual: Edit JSON Directly

```bash
# Edit the file
nano scripts/perceptron_weights.json

# Validate JSON
python3 -c "import json; json.load(open('scripts/perceptron_weights.json'))"
```

### Hybrid: Start from Auto-Generated, Then Tune

```bash
# Generate weights from benchmark data
python3 scripts/graphbrew_experiment.py --phase weights

# Backup
cp results/perceptron_weights.json results/perceptron_weights.json.auto

# Edit manually to adjust biases
vim results/perceptron_weights.json
```

---

## Validating Weights

### Check Current Selections

```bash
./bench/bin/pr -f graph.el -s -o 15 -n 1 2>&1 | grep -A 20 "Adaptive Reordering"
```

Output shows what was selected:
```
=== Adaptive Reordering Selection (Depth 0, Modularity: 0.835) ===
Comm    Nodes   Edges   Density DegVar  HubConc Selected
0       3500    45000   0.0073  89.3    0.62    LeidenHybrid
1       2800    28000   0.0071  34.2    0.41    LeidenHybrid
2       600     3200    0.0178  2.1     0.09    ORIGINAL
```

### Debug Score Calculation

Add verbose output:
```bash
./bench/bin/pr -f graph.el -s -o 15 -n 1 -v
```

### Unit Test Weights

```python
#!/usr/bin/env python3
import json

def test_weights(weights_file):
    with open(weights_file) as f:
        weights = json.load(f)
    
    # Test case: large modular community
    features = {
        "modularity": 0.8,
        "log_nodes": 4.0,
        "log_edges": 5.0,
        "density": 0.01,
        "avg_degree": 0.25,
        "degree_variance": 0.5,
        "hub_concentration": 0.4
    }
    
    scores = {}
    for algo, w in weights.items():
        score = w.get("bias", 0.5)
        for feat, val in features.items():
            score += w.get(f"w_{feat}", 0) * val
        scores[algo] = score
    
    best = max(scores.items(), key=lambda x: x[1])
    print(f"Best algorithm: {best[0]} (score: {best[1]:.3f})")
    
    for algo, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {algo}: {score:.3f}")

test_weights("scripts/perceptron_weights.json")
```

---

## Default Weights

If no weights file exists, these defaults are used:

```json
{
  "ORIGINAL": {
    "bias": 0.6,
    "w_modularity": 0.0,
    "w_log_nodes": -0.1,
    "w_log_edges": -0.1,
    "w_density": 0.1,
    "w_avg_degree": 0.0,
    "w_degree_variance": 0.0,
    "w_hub_concentration": 0.0
  },
  "LeidenHybrid": {
    "bias": 0.85,
    "w_modularity": 0.25,
    "w_log_nodes": 0.1,
    "w_log_edges": 0.1,
    "w_density": -0.05,
    "w_avg_degree": 0.15,
    "w_degree_variance": 0.15,
    "w_hub_concentration": 0.25
  }
}
```

These defaults:
- Favor ORIGINAL for small communities
- Favor LeidenHybrid for large, modular, hub-heavy communities

---

## Troubleshooting

### "Algorithm X never gets selected"

Check if bias is too low:
```bash
cat scripts/perceptron_weights.json | grep -A 1 '"AlgorithmX"'
```

Increase bias or relevant weights.

### "Wrong algorithm selected"

1. Check feature values:
```bash
./bench/bin/pr -f graph.el -s -o 15 -n 1 2>&1 | grep "Comm"
```

2. Manually calculate scores with those features

3. Adjust weights accordingly

### "Weights file not loading"

Check file exists and is valid JSON:
```bash
ls -la scripts/perceptron_weights.json
python3 -c "import json; json.load(open('scripts/perceptron_weights.json'))"
```

Check environment variable:
```bash
echo $PERCEPTRON_WEIGHTS_FILE
```

---

## Best Practices

### 1. Start Conservative

Begin with moderate biases (0.5-0.7) and small weights (±0.1-0.2).

### 2. Validate Empirically

Always benchmark after tuning:
```bash
# Before tuning
./bench/bin/pr -f graph.el -s -o 15 -n 10

# After tuning
./bench/bin/pr -f graph.el -s -o 15 -n 10
```

### 3. Keep Backups

```bash
cp scripts/perceptron_weights.json scripts/perceptron_weights.json.backup
```

### 4. Document Changes

Add comments (JSON doesn't support comments, but you can use a README):
```bash
echo "2026-01-18: Increased LeidenHybrid bias for PageRank workload" >> scripts/weights_changelog.txt
```

---

## Next Steps

- [[Correlation-Analysis]] - Automatic weight generation
- [[AdaptiveOrder-ML]] - Using the perceptron model
- [[Adding-New-Algorithms]] - Add algorithms to the perceptron

---

[← Back to Home](Home) | [GraphBrewOrder →](GraphBrewOrder)
