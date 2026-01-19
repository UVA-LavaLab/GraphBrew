# AdaptiveOrder: ML-Powered Algorithm Selection

AdaptiveOrder (algorithm 15) uses a **machine learning perceptron** to automatically select the best reordering algorithm for each community in your graph. This page explains how it works and how to train it.

## Overview

Instead of using one reordering algorithm for the entire graph, AdaptiveOrder:
1. Detects communities using Leiden
2. Computes features for each community
3. Uses a trained perceptron to predict the best algorithm
4. Applies different algorithms to different communities

```
Graph → Leiden → Communities → Features → Perceptron → Per-Community Algorithms
```

## Why Per-Community Selection?

Different parts of a graph have different structures:
- **Hub communities**: Dense cores with high-degree vertices → HUBCLUSTERDBG works well
- **Sparse communities**: Road-network-like → RCM or ORIGINAL may be better
- **Hierarchical communities**: Tree-like → LeidenDFS variants excel

AdaptiveOrder selects the best algorithm for each community's characteristics.

## How to Use

### Basic Usage

```bash
# Let AdaptiveOrder choose automatically
./bench/bin/pr -f graph.el -s -o 15 -n 3
```

### Output Explained

```
=== Adaptive Reordering Selection (Depth 0, Modularity: 0.0301) ===
Comm    Nodes   Edges   Density DegVar  HubConc Selected
131     1662    16151   0.0117  1.9441  0.5686  LeidenHybrid
272     103     149     0.0284  2.8547  0.4329  Original
489     178     378     0.0240  1.3353  0.3968  HUBCLUSTERDBG
...

=== Algorithm Selection Summary ===
Original: 846 communities
LeidenHybrid: 3 communities
HUBCLUSTERDBG: 2 communities
```

This shows:
- Each community's features
- Which algorithm was selected for each
- Summary of selections

---

## The Perceptron Model

### What is a Perceptron?

A perceptron is a simple linear classifier that computes a weighted sum of features:

```
score(algorithm) = bias + w1×feature1 + w2×feature2 + ...
```

The algorithm with the highest score wins.

### Features Used

| Feature | Description | Range |
|---------|-------------|-------|
| `modularity` | Community cohesion | 0.0 - 1.0 |
| `log_nodes` | log10(num_nodes) | 0 - 10 |
| `log_edges` | log10(num_edges) | 0 - 15 |
| `density` | edges / max_edges | 0.0 - 1.0 |
| `avg_degree` | mean degree / 100 | 0.0 - 1.0 |
| `degree_variance` | degree distribution spread | 0.0 - 5.0 |
| `hub_concentration` | fraction of edges to top 10% | 0.0 - 1.0 |

### Weight Structure

Each algorithm has weights for each feature:

```json
{
  "LeidenHybrid": {
    "bias": 0.85,
    "w_modularity": 0.25,
    "w_log_nodes": 0.1,
    "w_log_edges": 0.1,
    "w_density": -0.05,
    "w_avg_degree": 0.15,
    "w_degree_variance": 0.15,
    "w_hub_concentration": 0.25
  },
  ...
}
```

### Score Calculation Example

```
Community features:
- modularity: 0.5
- log_nodes: 4.0
- log_edges: 5.0
- density: 0.01
- avg_degree: 0.2
- degree_variance: 1.5
- hub_concentration: 0.4

LeidenHybrid score:
= 0.85                    # bias
+ 0.25 × 0.5              # modularity
+ 0.1 × 4.0               # log_nodes
+ 0.1 × 5.0               # log_edges
+ (-0.05) × 0.01          # density
+ 0.15 × 0.2              # avg_degree
+ 0.15 × 1.5              # degree_variance
+ 0.25 × 0.4              # hub_concentration
= 0.85 + 0.125 + 0.4 + 0.5 - 0.0005 + 0.03 + 0.225 + 0.1
= 2.23
```

---

## Training the Perceptron

### Automatic Training via Correlation Analysis

The easiest way to train is using the correlation analysis script:

```bash
# Run benchmarks and generate weights
python3 scripts/analysis/correlation_analysis.py \
    --graphs-dir ./graphs \
    --benchmark pr bfs \
    --algorithms 0 5 7 12 16 17 18 19 20
```

This:
1. Runs all specified algorithms on all graphs
2. Identifies which algorithm is best for each graph
3. Correlates graph features with best algorithm
4. Generates `scripts/perceptron_weights.json`

### Quick Training (Synthetic Graphs)

```bash
# Quick test with generated graphs
python3 scripts/analysis/correlation_analysis.py --quick
```

### Understanding the Training Output

```
----------------------------------------------------------------------
Computing Perceptron Weights
----------------------------------------------------------------------
Loaded existing weights from: scripts/perceptron_weights.json
Backup created: scripts/perceptron_weights.json.backup

Perceptron weights saved to: scripts/perceptron_weights.json
  20 algorithms configured (all 0-20)
  Updated from benchmarks: ORIGINAL, LeidenOrder, LeidenHybrid
  C++ will automatically load these weights at runtime
```

---

## Weight File Format

### Location

```
GraphBrew/scripts/perceptron_weights.json
```

### Structure

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
    ...
  },
  ...
}
```

### Algorithm Name Mapping

| ID | Name in JSON |
|----|--------------|
| 0 | ORIGINAL |
| 1 | RANDOM |
| 2 | SORT |
| 3 | HUBSORT |
| 4 | HUBCLUSTER |
| 5 | DBG |
| 6 | HUBSORTDBG |
| 7 | HUBCLUSTERDBG |
| 8 | RABBITORDER |
| 9 | GORDER |
| 10 | CORDER |
| 11 | RCM |
| 12 | LeidenOrder |
| 13 | GraphBrewOrder |
| 15 | AdaptiveOrder |
| 16 | LeidenDFS |
| 17 | LeidenDFSHub |
| 18 | LeidenDFSSize |
| 19 | LeidenBFS |
| 20 | LeidenHybrid |

---

## Manual Weight Tuning

### When to Tune Manually

- You have domain knowledge about your graphs
- Automated training doesn't have enough data
- You want to favor certain algorithms

### Tuning Guidelines

**Bias**: Base preference for algorithm (0.3 - 1.0)
- Higher = algorithm selected more often
- Start around 0.5-0.7

**w_modularity**: How algorithm performs on modular graphs
- Positive = better on high-modularity graphs
- Community algorithms should have positive values (0.1 - 0.3)

**w_log_nodes / w_log_edges**: Scale effects
- Positive = better on larger graphs
- Negative = better on smaller graphs

**w_density**: Sparse vs dense preference
- Positive = better on dense graphs
- Negative = better on sparse graphs

**w_hub_concentration**: Hub structure preference
- Positive = better when hubs dominate
- Hub algorithms (HUBSORT, HUBCLUSTER) should have positive values

### Example: Favoring LeidenHybrid for Large Graphs

```json
{
  "LeidenHybrid": {
    "bias": 0.9,
    "w_modularity": 0.3,
    "w_log_nodes": 0.2,
    "w_log_edges": 0.2,
    "w_density": -0.1,
    "w_avg_degree": 0.15,
    "w_degree_variance": 0.15,
    "w_hub_concentration": 0.25
  }
}
```

---

## How C++ Loads Weights

### Automatic Loading

The C++ code automatically loads weights from:
1. `PERCEPTRON_WEIGHTS_FILE` environment variable (if set)
2. `scripts/perceptron_weights.json` (default)
3. Hardcoded defaults (if file not found)

### Environment Variable Override

```bash
# Use custom weights file
export PERCEPTRON_WEIGHTS_FILE=/path/to/my_weights.json
./bench/bin/pr -f graph.el -s -o 15 -n 3
```

### Fallback to Defaults

If the weights file doesn't exist or is invalid:
- C++ uses hardcoded defaults
- Warning is printed (in verbose mode)
- System continues to work

---

## Recursive AdaptiveOrder

AdaptiveOrder can recursively apply itself to large communities:

```
Level 0: Full graph → 10 communities
Level 1: Large community → 5 sub-communities
Level 2: Sub-community → final ordering
```

This creates a hierarchical ordering that respects structure at multiple scales.

### Controlling Recursion

The recursion depth is controlled by:
- Community size threshold (min nodes for recursion)
- Maximum depth limit

---

## Debugging AdaptiveOrder

### Verbose Output

```bash
# See detailed selection process
./bench/bin/pr -f graph.el -s -o 15 -n 1 2>&1 | head -50
```

### Check Which Weights Are Loaded

Look for output like:
```
Perceptron: Loaded 20 algorithm weights from scripts/perceptron_weights.json
```

### Verify Weights File

```bash
# Check JSON is valid
python3 -c "import json; json.load(open('scripts/perceptron_weights.json'))"

# View contents
cat scripts/perceptron_weights.json | python3 -m json.tool
```

---

## Performance Considerations

### When AdaptiveOrder Helps Most

✅ Graphs with diverse community structures
✅ Large graphs where wrong algorithm choice is costly
✅ Unknown graphs in automated pipelines

### When to Use Fixed Algorithm Instead

❌ Small graphs (overhead not worth it)
❌ Graphs you know well (just use the best algorithm)
❌ Graphs with uniform structure (all communities similar)

### Overhead

- Leiden community detection: ~5-10% of total time
- Feature computation: ~1-2% of total time
- Perceptron inference: < 1% of total time

Total overhead is usually recovered through better algorithm selection.

---

## Next Steps

- [[Perceptron-Weights]] - Detailed weight file documentation
- [[Correlation-Analysis]] - Understanding the training process
- [[Adding-New-Algorithms]] - Add algorithms to the perceptron

---

[← Back to Home](Home) | [Reordering Algorithms →](Reordering-Algorithms)
