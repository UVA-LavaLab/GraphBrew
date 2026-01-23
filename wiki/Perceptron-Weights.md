# Perceptron Weights

The perceptron weights control how AdaptiveOrder selects algorithms for each community. This page explains the weight structure, tuning strategies, and how to customize the model.

## Overview

```
scripts/weights/type_N.json   # Auto-clustered type weights
```

Each JSON file contains weights for each algorithm. When AdaptiveOrder processes a community, it computes a score for each algorithm using these weights and selects the highest-scoring one.

## How Perceptron Scoring Works

```
INPUTS (Features)         WEIGHTS              OUTPUT
=================         =======              ======

modularity: 0.72  --*---> w_mod: 0.28 ---+
                                         |
density: 0.001    --*---> w_den: -0.15 --+
                                         |
degree_var: 2.1   --*---> w_dv: 0.18 ----+
                                         |
hub_conc: 0.45    --*---> w_hc: 0.22 ----+----> SUM ---> SCORE
                                         |      (+bias)
cluster_coef: 0.3 --*---> w_cc: 0.12 ----+
                                         |
avg_path: 5.2     --*---> w_ap: 0.08 ----+
                                         |
diameter: 16      --*---> w_di: 0.05 ----+
                                         |
...               --*---> ...       -----+


ALGORITHM SELECTION:
====================
RABBITORDER:    score = 2.31  <-- WINNER
LeidenDFS:      score = 2.18
HubClusterDBG:  score = 1.95
GORDER:         score = 1.82
ORIGINAL:       score = 0.50
```

**Key Insight:** Each weight encodes "When this feature is high, how well does this algorithm perform?"

---

## Weight File Location

### Default Location
```
GraphBrew/scripts/weights/
├── type_registry.json    # Maps graphs → types + centroids
├── type_0.json           # Cluster 0 weights
├── type_1.json           # Cluster 1 weights
└── type_N.json           # Additional clusters

```

### Automatic Clustering and Storage

When weights are saved, the script automatically:
1. Creates type-based weight files in `scripts/weights/` (e.g., `type_0.json`, `type_1.json`)
2. Updates the **type registry** (`scripts/weights/type_registry.json`) with cluster centroids

This ensures weights are never accidentally overwritten and the best cluster is selected at runtime.

### Auto-Clustering Type System

AdaptiveOrder uses an automatic clustering system that groups graphs by feature similarity instead of predefined categories. This allows the system to scale to any number of graph types.

**How It Works:**
1. **Feature Extraction:** For each graph, compute 9 features: modularity, log_nodes, log_edges, density, avg_degree, degree_variance, hub_concentration, clustering_coefficient, community_count
2. **Clustering:** Group similar graphs using k-means-like clustering
3. **Per-Cluster Training:** Train optimized weights for each cluster
4. **Runtime Matching:** Select best cluster based on Euclidean distance to centroid

**Type Files:**
```
scripts/weights/
├── type_registry.json    # {graph: type_N, centroids: {...}}
├── type_0.json           # Weights for cluster 0
├── type_1.json           # Weights for cluster 1
└── type_N.json           # Additional clusters as needed
```

**Loading Order:**
1. Environment variable `PERCEPTRON_WEIGHTS_FILE` (if set)
2. Best matching type file from `scripts/weights/type_N.json`
3. Semantic type fallback (if type files don't exist)
4. Hardcoded defaults

**Example Output:**
```
Finding best type match for features: mod=0.4521, deg_var=0.8012, hub=0.3421...
Best matching type: type_0 (distance: 0.12)
Loaded 21 weights from scripts/weights/type_0.json
```

### Environment Override
```bash
export PERCEPTRON_WEIGHTS_FILE=/path/to/custom_weights.json
./bench/bin/pr -f graph.el -s -o 15 -n 3
```

### Fallback Behavior
If no type files exist, C++ uses hardcoded defaults with conservative weights that favor ORIGINAL for small communities and LeidenHybrid for larger ones.

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
| `benchmark_weights.tc` | Multiplier for Triangle Counting benchmark |

**Usage:**
- When `BENCH_GENERIC` (default), multiplier = 1.0 (no adjustment)
- When a specific benchmark is passed, the score is multiplied by the corresponding weight
- This allows algorithms to score differently for different workloads

```cpp
// C++ Usage:
SelectReorderingPerceptron(features);           // BENCH_GENERIC (multiplier = 1.0)
SelectReorderingPerceptron(features, BENCH_PR); // Uses benchmark_weights.pr
SelectReorderingPerceptron(features, "bfs");    // Uses benchmark_weights.bfs
```

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
base_score = bias + Σ(w_feature × feature_value)
final_score = base_score × benchmark_weights[benchmark_type]
```

For `BENCH_GENERIC` (default when no benchmark is specified), the multiplier is 1.0.

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
    --output scripts/weights/type_0.json
```

### Manual: Edit JSON Directly

```bash
# Edit the file
nano scripts/weights/type_0.json

# Validate JSON
python3 -c "import json; json.load(open('scripts/weights/type_0.json'))"
```

### Hybrid: Start from Auto-Generated, Then Tune

```bash
# Generate weights from benchmark data
python3 scripts/graphbrew_experiment.py --phase weights

# Backup
cp scripts/weights/type_0.json scripts/weights/type_0.json.backup

# Edit manually to adjust biases
vim scripts/weights/type_0.json
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

test_weights("scripts/weights/type_0.json")
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

## Understanding Algorithm Biases

The **bias** is the most important weight - it's the base preference for an algorithm before any graph features are considered. Higher bias = more likely to be selected.

### How Bias is Calculated

When training weights, the bias is computed from benchmark performance:

```
bias = 0.5 × avg_speedup_vs_RANDOM
```

Where:
- **RANDOM (ID 1)** is the baseline - represents worst-case random node ordering
- `avg_speedup` = RANDOM_time / Algorithm_time

### Example Bias Values

After training on 87 graphs, typical biases look like:

| Algorithm | Bias | Interpretation |
|-----------|------|----------------|
| HUBSORT | ~26.0 | **26x faster** than random on average |
| SORT | ~9.5 | 9.5x faster than random |
| HUBSORTDBG | ~8.5 | 8.5x faster than random |
| LeidenDFSHub | ~2.4 | 2.4x faster than random |
| LeidenBFS | ~2.3 | 2.3x faster than random |
| LeidenOrder | ~1.8 | 1.8x faster than random |
| ORIGINAL | 0.5 | Baseline (no reordering) |
| RANDOM | 0.5 | Baseline (1.00x) |
| DBG | ~0.44 | Slightly worse than random |
| RABBITORDER | ~0.40 | Slower than random on small graphs |

### Why Some Algorithms Have Low Bias

Algorithms like `RABBITORDER`, `DBG`, and `LeidenHybrid` may show biases < 0.5, meaning they're slower than RANDOM on the training graphs. This happens because:

1. **Reordering overhead** - Complex algorithms have high setup cost
2. **Small graph penalty** - Sophisticated ordering doesn't help small graphs
3. **Graph type mismatch** - Algorithm designed for different graph structures

### Adjusting Biases Manually

To favor an algorithm regardless of graph features:

```json
{
  "LeidenHybrid": {
    "bias": 3.0,  // Force higher preference
    "w_modularity": 0.1,
    ...
  }
}
```

---

## Step-by-Step Terminal Training Guide

### Training Pipeline Diagram

```
+=====================================================+
|            TRAINING PIPELINE (Python)               |
+=====================================================+

+-----------+    +-----------+    +-----------+
|  Graph 1  |    |  Graph 2  |    |  Graph N  |
+-----+-----+    +-----+-----+    +-----+-----+
      |               |               |
      +-------+-------+-------+-------+
              |
              v
      +-------+-------+
      | PHASE 1:      |
      | Reorder each  |
      | graph with    |
      | all 18 algos  |
      +-------+-------+
              |
              v
      +-------+-------+
      | PHASE 2:      |
      | Run benchmarks|
      | PR,BFS,CC,    |
      | SSSP,BC       |
      +-------+-------+
              |
              v
      +-------+-------+
      | For each      |
      | result:       |
      | 1.Get speedup |
      | 2.Get features|
      | 3.Find type   |
      | 4.Update wts  |
      +-------+-------+
              |
              v
+-------------+-------------+
|  type_0.json | type_1.json |
|  (social)    | (road)      |
+--------------+-------------+
```

### Complete Training Workflow

#### Step 1: Clean Previous Results (Optional)

```bash
cd /path/to/GraphBrew

# Backup current weights
cp -r scripts/weights/ scripts/weights.backup/

# Clean results but keep graphs
python3 scripts/graphbrew_experiment.py --clean
```

#### Step 2: Generate Reorderings for All Graphs

```bash
# Small graphs only (~5 minutes)
python3 scripts/graphbrew_experiment.py \
    --graphs-dir ./results/graphs \
    --graphs small \
    --phase reorder \
    --generate-maps

# All graphs including large (~1-2 hours)
python3 scripts/graphbrew_experiment.py \
    --graphs-dir ./results/graphs \
    --max-graphs 50 \
    --phase reorder \
    --generate-maps
```

#### Step 3: Run Benchmarks (RANDOM Baseline)

```bash
# All benchmarks on all graphs
python3 scripts/graphbrew_experiment.py \
    --graphs-dir ./results/graphs \
    --max-graphs 50 \
    --phase benchmark \
    --benchmarks pr bfs cc sssp bc \
    --trials 2 \
    --use-maps
```

#### Step 4: Generate New Weights

```bash
# Generate weights from benchmark results
python3 scripts/graphbrew_experiment.py \
    --graphs-dir ./results/graphs \
    --phase weights
```

#### Step 5: Run Brute-Force Validation

```bash
# Test all 20 algorithms vs adaptive choice
python3 scripts/graphbrew_experiment.py \
    --graphs-dir ./results/graphs \
    --max-graphs 50 \
    --brute-force \
    --bf-benchmark pr \
    --trials 2
```

#### Step 6: View Results

```bash
# Check type weight files
ls -la scripts/weights/

# Check algorithm biases in type_0.json
cat scripts/weights/type_0.json | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('Algorithm Biases (sorted by speedup):')
items = [(k, v.get('bias', 0)) for k, v in d.items() if not k.startswith('_')]
for k, b in sorted(items, key=lambda x: -x[1])[:15]:
    print(f'  {k:20s}: {b:.3f}')
"

# Check validation accuracy
cat results/brute_force_*.json | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    print(f'Accuracy: {d.get(\"overall_accuracy\", \"N/A\")}')
" 2>/dev/null || echo "Run brute-force first"
```

### Quick Training (Minimal)

For fast iteration during development:

```bash
# Quick training on 3 small graphs, 1 trial each
python3 scripts/graphbrew_experiment.py \
    --graphs-dir ./results/graphs \
    --graphs small \
    --max-graphs 3 \
    --trials 1 \
    --skip-cache \
    2>&1 | tee /tmp/training.log
```

### Fill All Weight Fields (Comprehensive)

If many weight fields show as 0 or default 1.0, use `--fill-weights` to populate ALL fields:

```bash
# Fill ALL weight fields including cache impacts and topology features
python3 scripts/graphbrew_experiment.py \
    --fill-weights \
    --graphs-dir ./results/graphs \
    --graphs small \
    --max-graphs 5 \
    --trials 2
```

This runs all 6 phases sequentially:
1. **Phase 1 (Reorderings)**: Fills `w_reorder_time`
2. **Phase 2 (Benchmarks)**: Fills `bias`, `w_log_edges`, `w_avg_degree`  
3. **Phase 3 (Cache Sim)**: Fills `cache_l1_impact`, `cache_l2_impact`, `cache_l3_impact`
4. **Phase 4 (Base Weights)**: Fills `w_density`, `w_degree_variance`, `w_hub_concentration`
5. **Phase 5 (Topology)**: Fills `w_clustering_coeff`, `w_avg_path_length`, `w_diameter`
6. **Phase 6 (Benchmark Weights)**: Fills `benchmark_weights.{pr,bfs,cc,sssp,bc}`

---

## Gradient Update Rule: Online Learning

### Error Signal Computation

```python
error = (speedup - 1.0) - current_score
# Positive error: algorithm performed better than expected
# Negative error: algorithm performed worse than expected
```

### Weight Update (Stochastic Gradient Descent)

```python
learning_rate = 0.01

# Core feature weights
w_modularity      += lr * error * modularity
w_density         += lr * error * density  
w_degree_variance += lr * error * degree_variance
w_hub_concentration += lr * error * hub_concentration

# Extended topology weights
w_avg_path_length += lr * error * (avg_path_length / 10)
w_diameter        += lr * error * (diameter / 50)
w_clustering_coeff += lr * error * clustering_coeff

# Cache impact weights (when simulation data available)
w_cache_l1_impact += lr * error * l1_hit_rate
w_cache_dram_penalty += lr * error * dram_penalty
```

### When Algorithm Performs Well (speedup > 1.0)

```python
error = speedup - 1.0 - current_score  # positive

# Increase weights for features that predicted this success
w_modularity += learning_rate * error * modularity
w_hub_concentration += learning_rate * error * hub_concentration
# ... etc for all features
```

**Effect**: Algorithm gets higher scores for similar graphs in future.

### When Algorithm Performs Poorly (speedup < 1.0)

```python
error = speedup - 1.0 - current_score  # negative

# Decrease weights (error is negative)
w_modularity += learning_rate * error * modularity
```

**Effect**: Algorithm gets lower scores for similar graphs in future.

---

### Full Training (Production)

For production-quality weights:

```bash
# Full training on all graphs with validation
python3 scripts/graphbrew_experiment.py \
    --graphs-dir ./results/graphs \
    --max-graphs 50 \
    --benchmarks pr bfs cc sssp bc \
    --trials 3 \
    --generate-maps \
    --use-maps \
    2>&1 | tee results/training_full.log

# Follow with brute-force validation
python3 scripts/graphbrew_experiment.py \
    --graphs-dir ./results/graphs \
    --max-graphs 50 \
    --brute-force \
    --bf-benchmark pr \
    --trials 2 \
    2>&1 | tee results/validation.log
```

### Monitoring Long Runs

```bash
# In another terminal, monitor progress
tail -f results/training_full.log

# Check if still running
ps aux | grep graphbrew

# Kill if needed
pkill -f graphbrew_experiment
```

---

## Troubleshooting

### "Algorithm X never gets selected"

Check if bias is too low:
```bash
cat scripts/weights/type_0.json | grep -A 1 '"AlgorithmX"'
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
ls -la scripts/weights/type_*.json
python3 -c "import json; json.load(open('scripts/weights/type_0.json'))"
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
cp -r scripts/weights/ scripts/weights.backup/
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
