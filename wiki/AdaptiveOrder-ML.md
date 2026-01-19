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

---

## How Correlation Analysis Works

### Step 1: Benchmark All Algorithms

For each graph in your dataset, we run every algorithm and measure execution time:

```
Graph: facebook.el
┌────────────────────┬──────────────┬─────────┐
│ Algorithm          │ Time (sec)   │ Rank    │
├────────────────────┼──────────────┼─────────┤
│ ORIGINAL (0)       │ 0.0523       │ 5       │
│ HUBCLUSTERDBG (7)  │ 0.0412       │ 3       │
│ LeidenOrder (12)   │ 0.0398       │ 2       │
│ LeidenHybrid (20)  │ 0.0371       │ 1 ★     │
│ RCM (11)           │ 0.0489       │ 4       │
└────────────────────┴──────────────┴─────────┘
Winner: LeidenHybrid (fastest)
```

### Step 2: Extract Graph Features

For each graph, we compute structural features:

```
Graph: facebook.el
┌─────────────────────┬────────────┐
│ Feature             │ Value      │
├─────────────────────┼────────────┤
│ num_nodes           │ 4,039      │
│ num_edges           │ 88,234     │
│ modularity          │ 0.835      │  ← High (strong communities)
│ density             │ 0.0108     │  ← Sparse
│ avg_degree          │ 43.7       │
│ degree_variance     │ 52.4       │  ← High (hubs exist)
│ hub_concentration   │ 0.42       │  ← Moderate hub dominance
└─────────────────────┴────────────┘
```

### Step 3: Build the Correlation Table

We create a table linking features to winning algorithms:

```
┌──────────────────┬────────────┬─────────┬─────────┬─────────┬────────────────┐
│ Graph            │ Modularity │ Density │ HubConc │ DegVar  │ Best Algorithm │
├──────────────────┼────────────┼─────────┼─────────┼─────────┼────────────────┤
│ facebook.el      │ 0.835      │ 0.011   │ 0.42    │ 52.4    │ LeidenHybrid   │
│ twitter.el       │ 0.721      │ 0.002   │ 0.68    │ 891.2   │ LeidenDFSHub   │
│ roadNet-CA.el    │ 0.112      │ 0.0001  │ 0.05    │ 1.2     │ RCM            │
│ web-Google.el    │ 0.654      │ 0.008   │ 0.55    │ 234.5   │ HUBCLUSTERDBG  │
│ citation.el      │ 0.443      │ 0.003   │ 0.31    │ 45.6    │ LeidenOrder    │
└──────────────────┴────────────┴─────────┴─────────┴─────────┴────────────────┘
```

### Step 4: Compute Correlations

We calculate Pearson correlation between each feature and algorithm performance:

```
Feature Correlations with "LeidenHybrid being best":
┌─────────────────────┬─────────────┬────────────────────────────────────┐
│ Feature             │ Correlation │ Interpretation                     │
├─────────────────────┼─────────────┼────────────────────────────────────┤
│ modularity          │ +0.78       │ Strong positive → use on modular   │
│ hub_concentration   │ +0.45       │ Moderate positive → helps with hubs│
│ density             │ -0.23       │ Weak negative → prefers sparse     │
│ degree_variance     │ +0.52       │ Moderate positive → handles skew   │
└─────────────────────┴─────────────┴────────────────────────────────────┘
```

### Step 5: Convert Correlations to Perceptron Weights

The correlations become the perceptron weights:

```python
# Simplified weight derivation
weights["LeidenHybrid"] = {
    "bias": 0.5 + (win_rate * 0.5),      # Base preference from win rate
    "w_modularity": correlation_modularity * scale,      # +0.78 → +0.25
    "w_hub_concentration": correlation_hubconc * scale,  # +0.45 → +0.15
    "w_density": correlation_density * scale,            # -0.23 → -0.08
    "w_degree_variance": correlation_degvar * scale,     # +0.52 → +0.17
}
```

The scaling factor normalizes correlations to useful weight ranges (typically 0.1-0.3).

---

## How Perceptron Scores Feed Into Ordering

### Complete Example: Ordering a Social Network

Let's walk through ordering `social_network.el` with AdaptiveOrder:

#### Phase 1: Community Detection

```
Input Graph: 10,000 nodes, 150,000 edges

Leiden detects 5 communities:
┌─────────┬───────┬────────┬──────────┬─────────┬──────────┐
│ Comm ID │ Nodes │ Edges  │ Density  │ HubConc │ DegVar   │
├─────────┼───────┼────────┼──────────┼─────────┼──────────┤
│ C0      │ 3,500 │ 45,000 │ 0.0073   │ 0.62    │ 89.3     │
│ C1      │ 2,800 │ 28,000 │ 0.0071   │ 0.41    │ 34.2     │
│ C2      │ 1,900 │ 22,000 │ 0.0122   │ 0.28    │ 12.1     │
│ C3      │ 1,200 │ 8,500  │ 0.0118   │ 0.15    │ 5.4      │
│ C4      │ 600   │ 3,200  │ 0.0178   │ 0.09    │ 2.1      │
└─────────┴───────┴────────┴──────────┴─────────┴──────────┘
```

#### Phase 2: Feature Computation for Each Community

For community C0:
```
Features (normalized):
- modularity: 0.72 (from Leiden quality)
- log_nodes: 3.54 (log10(3500))
- log_edges: 4.65 (log10(45000))
- density: 0.0073
- avg_degree: 25.7 / 100 = 0.257
- degree_variance: 89.3 / 100 = 0.893
- hub_concentration: 0.62
```

#### Phase 3: Perceptron Scoring for Community C0

```
Algorithm Scores for C0 (high hubs, high variance):

LeidenHybrid:
  = 0.85 + (0.25×0.72) + (0.1×3.54) + (0.1×4.65) + (-0.05×0.007)
    + (0.15×0.257) + (0.15×0.893) + (0.25×0.62)
  = 0.85 + 0.18 + 0.354 + 0.465 - 0.0004 + 0.039 + 0.134 + 0.155
  = 2.18 ★ WINNER

HUBCLUSTERDBG:
  = 0.75 + (0.1×0.72) + (0.05×3.54) + (0.05×4.65) + (0.0×0.007)
    + (0.2×0.257) + (0.2×0.893) + (0.35×0.62)
  = 0.75 + 0.072 + 0.177 + 0.233 + 0 + 0.051 + 0.179 + 0.217
  = 1.68

ORIGINAL:
  = 0.6 + (0.0×0.72) + (-0.1×3.54) + (-0.1×4.65) + (0.1×0.007)
    + (0.0×0.257) + (0.0×0.893) + (0.0×0.62)
  = 0.6 + 0 - 0.354 - 0.465 + 0.0007 + 0 + 0 + 0
  = -0.22

RCM:
  = 0.65 + (0.05×0.72) + (0.0×3.54) + (0.0×4.65) + (0.15×0.007)
    + (0.1×0.257) + (-0.1×0.893) + (-0.05×0.62)
  = 0.65 + 0.036 + 0 + 0 + 0.001 + 0.026 - 0.089 - 0.031
  = 0.59
```

**Result**: C0 gets **LeidenHybrid** (score 2.18)

#### Phase 4: Scoring All Communities

```
┌─────────┬──────────────────┬────────────────────────────────────────┐
│ Comm    │ Selected Algo    │ Reasoning                              │
├─────────┼──────────────────┼────────────────────────────────────────┤
│ C0      │ LeidenHybrid     │ High hub_conc (0.62), high deg_var     │
│ C1      │ LeidenHybrid     │ Moderate hub_conc, good modularity     │
│ C2      │ LeidenOrder      │ Lower hub_conc, still modular          │
│ C3      │ HUBCLUSTERDBG    │ Small community, moderate structure    │
│ C4      │ ORIGINAL         │ Very small, overhead not worth it      │
└─────────┴──────────────────┴────────────────────────────────────────┘
```

#### Phase 5: Apply Per-Community Ordering

```
Final Vertex Relabeling:

Original IDs → New IDs (after per-community reordering)

Community C0 (LeidenHybrid applied):
  Vertices 0-3499 → reordered by hub-aware DFS within C0
  New IDs: 0-3499

Community C1 (LeidenHybrid applied):
  Vertices 3500-6299 → reordered by hub-aware DFS within C1
  New IDs: 3500-6299

Community C2 (LeidenOrder applied):
  Vertices 6300-8199 → simple community ordering
  New IDs: 6300-8199

Community C3 (HUBCLUSTERDBG applied):
  Vertices 8200-9399 → hub clustering with DBG
  New IDs: 8200-9399

Community C4 (ORIGINAL - no change):
  Vertices 9400-9999 → kept as-is
  New IDs: 9400-9999
```

#### Visual Representation

```
Before AdaptiveOrder:
┌─────────────────────────────────────────────────────┐
│ Memory layout: vertices scattered, poor locality    │
│ [v892][v12][v5601][v234][v8923][v45]...            │
└─────────────────────────────────────────────────────┘

After AdaptiveOrder:
┌─────────────────────────────────────────────────────┐
│ C0 (LeidenHybrid)    │ C1 (LeidenHybrid)    │ ...  │
│ [hub1][hub2][n1][n2] │ [hub1][n1][n2][n3]   │      │
│ Hubs clustered first │ Hubs clustered first │      │
└─────────────────────────────────────────────────────┘

Result: When PageRank processes C0, all hub vertices
        are in adjacent cache lines → fewer cache misses
```

---

## Why Different Algorithms for Different Communities?

### The Key Insight

Not all communities have the same structure:

```
Community C0: Social cluster (influencers + followers)
  → High hub concentration (0.62)
  → LeidenHybrid groups influencers together
  → Their followers are adjacent in memory

Community C4: Small tight-knit group
  → Low hub concentration (0.09)
  → Few vertices (600)
  → Reordering overhead > benefit
  → ORIGINAL keeps natural ordering
```

### Performance Impact

```
PageRank on Community C0:
┌────────────────────┬──────────────┬─────────────────┐
│ Algorithm          │ Cache Misses │ Time (ms)       │
├────────────────────┼──────────────┼─────────────────┤
│ ORIGINAL           │ 145,000      │ 12.3            │
│ HUBCLUSTERDBG      │ 98,000       │ 8.7             │
│ LeidenHybrid       │ 67,000       │ 6.2 ★           │
└────────────────────┴──────────────┴─────────────────┘

PageRank on Community C4:
┌────────────────────┬──────────────┬─────────────────┐
│ Algorithm          │ Cache Misses │ Time (ms)       │
├────────────────────┼──────────────┼─────────────────┤
│ ORIGINAL           │ 1,200        │ 0.4 ★           │
│ LeidenHybrid       │ 1,150        │ 0.5 (+ overhead)│
└────────────────────┴──────────────┴─────────────────┘

AdaptiveOrder picks the best for EACH community!
```

---

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
