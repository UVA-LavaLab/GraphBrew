# Configuration Files

Guide to GraphBrew configuration and data files.

---

## Overview

GraphBrew uses several configuration and data files:

```
scripts/
├── graphbrew_experiment.py    # Main script (uses CLI args)
├── requirements.txt           # Python dependencies
│
├── weights/                   # Perceptron weight files
│   ├── active/                # C++ runtime reads from here
│   │   ├── type_registry.json # Graph clusters + centroids
│   │   ├── type_0.json        # Cluster 0 algorithm weights
│   │   └── type_N.json        # Additional cluster weights
│   ├── merged/                # Accumulated weights from runs
│   └── runs/                  # Historical run snapshots
│
└── lib/                       # Python library modules
    ├── utils.py               # ALGORITHMS dict, constants
    ├── benchmark.py           # Benchmark execution
    ├── cache.py               # Cache simulation
    ├── weights.py             # Weight management
    ├── features.py            # Graph feature extraction
    └── ...                    # Additional modules

results/
├── perceptron_weights.json    # Combined weights
├── graph_properties_cache.json # Cached graph features
├── benchmark_*.json           # Benchmark result files
├── cache_*.json               # Cache simulation results
└── reorder_*.json             # Reorder timing results
```

---

## Command-Line Configuration

GraphBrew uses command-line arguments instead of JSON config files:

### Basic Usage

```bash
# Full pipeline with defaults
python3 scripts/graphbrew_experiment.py --full --download-size SMALL

# Customize experiments via command line
python3 scripts/graphbrew_experiment.py \
    --phase benchmark \
    --graphs small \
    --benchmarks pr bfs cc \
    --trials 5
```

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--full` | Run complete pipeline | - |
| `--download-size` | SMALL, MEDIUM, LARGE, ALL | - |
| `--phase` | all, reorder, benchmark, cache, weights | all |
| `--graphs` | small, medium, large, all, custom | all |
| `--benchmarks` | pr, bfs, cc, sssp, bc, tc | all 6 |
| `--trials` | Number of benchmark trials | 2 |
| `--key-only` | Only test key algorithms | false |
| `--skip-cache` | Skip cache simulation | false |

### Memory and Disk Limits

```bash
# Auto-detect available resources
python3 scripts/graphbrew_experiment.py --full --auto-memory --auto-disk

# Set explicit limits
python3 scripts/graphbrew_experiment.py --full --max-memory 32 --max-disk 100
```

---

## Perceptron Weight Files

### Type Registry (type_registry.json)

Maps graph types to clusters with centroid feature vectors:

```json
{
  "type_0": {
    "centroid": [0.811, 0.263, 0.420, 0.054, 0.512, 6.2e-05, 4.8e-05],
    "graph_count": 2,
    "algorithms": [],
    "graphs": []
  },
  "type_1": {
    "centroid": [0.479, 0.357, 0.426, 0.025, 0.002, 6.2e-05, 4.8e-05],
    "graph_count": 4
  }
}
```

Centroid array indices: [modularity, degree_variance, hub_concentration, avg_degree, clustering_coeff, avg_path_length, diameter]
```

### Type Weight Files (type_N.json)

Per-algorithm weights for each cluster:

```json
{
  "RABBITORDER": {
    "bias": 2.5,
    "w_modularity": 0.15,
    "w_log_nodes": 0.02,
    "w_density": -0.05,
    "w_hub_concentration": 0.12,
    "benchmark_weights": {
      "pr": 1.2,
      "bfs": 0.9,
      "cc": 1.0
    }
  },
  "LeidenCSR": {
    "bias": 3.2,
    "w_modularity": 0.20,
    "w_log_nodes": 0.01
  }
}
```

### Weight Field Descriptions

| Field | Description | Impact |
|-------|-------------|--------|
| `bias` | Base preference (higher = more likely selected) | Algorithm's inherent quality |
| `w_modularity` | Weight for graph modularity | Positive → good for modular graphs |
| `w_log_nodes` | Weight for log(node count) | Positive → scales better |
| `w_log_edges` | Weight for log(edge count) | Positive → handles large edge sets |
| `w_density` | Weight for edge density | Positive → good for dense graphs |
| `w_avg_degree` | Weight for average degree | Connectivity effect |
| `w_hub_concentration` | Weight for hub concentration | Positive → good for hub-heavy graphs |
| `w_degree_variance` | Weight for degree variance | Positive → handles skewed degrees |
| `w_clustering_coeff` | Weight for clustering coefficient | Local clustering effect |
| `w_avg_path_length` | Weight for average path length | Graph diameter sensitivity |
| `w_diameter` | Weight for graph diameter | Diameter effect |
| `w_community_count` | Weight for community count | Sub-community complexity |
| `cache_l1_impact` | Bonus for high L1 cache hit rate | Cache locality |
| `cache_l2_impact` | Bonus for high L2 cache hit rate | Cache locality |
| `cache_l3_impact` | Bonus for high L3 cache hit rate | Cache locality |
| `cache_dram_penalty` | Penalty for DRAM accesses | Memory bandwidth |
| `w_reorder_time` | Penalty for slow reordering | Negative → prefers fast algorithms |

---

## Results Files

### Graph Properties Cache (graph_properties_cache.json)

Cached graph features to avoid recomputation:

```json
{
  "email-Enron": {
    "modularity": 0.586,
    "degree_variance": 8.234,
    "hub_concentration": 0.412,
    "clustering_coefficient": 0.497,
    "avg_degree": 10.02,
    "avg_path_length": 4.25,
    "diameter": 10.0,
    "community_count": 45.0,
    "nodes": 36692,
    "edges": 183831,
    "graph_type": "social"
  }
}
```

### Benchmark Results (benchmark_*.json)

JSON array of benchmark results:

```json
[
  {
    "graph": "email-Enron",
    "algorithm": "HUBCLUSTERDBG",
    "algorithm_id": 7,
    "benchmark": "pr",
    "time_seconds": 0.0189,
    "reorder_time": 0.032,
    "trials": 2,
    "success": true,
    "error": "",
    "extra": {}
  }
]
```

---

## Environment Variables

### Perceptron Weights Override

```bash
# Use custom weights file (overrides type matching)
export PERCEPTRON_WEIGHTS_FILE=/path/to/custom_weights.json
./bench/bin/pr -f graph.el -s -o 14 -n 3
```

### OpenMP Threads

```bash
# Control parallelism
export OMP_NUM_THREADS=8
./bench/bin/pr -f graph.el -s -n 5
```

---

## Example Workflows

### Quick Test

```bash
# Test on small graphs with few algorithms
python3 scripts/graphbrew_experiment.py \
    --phase benchmark \
    --graphs small \
    --key-only \
    --trials 1
```

### Full Benchmark

```bash
# Complete benchmark with all algorithms
python3 scripts/graphbrew_experiment.py \
    --full \
    --download-size MEDIUM \
    --trials 5 \
    --auto-memory
```

### Weight Training Only

```bash
# Train perceptron weights from existing results
python3 scripts/graphbrew_experiment.py \
    --fill-weights \
    --skip-cache \
    --graphs small
```

### Adaptive Validation

```bash
# Compare AdaptiveOrder vs all algorithms
python3 scripts/graphbrew_experiment.py \
    --brute-force \
    --graphs small
```

---

## Troubleshooting

### "Weight file not found"

```bash
# Check weight files exist
ls -la scripts/weights/active/

# Regenerate weights
python3 scripts/graphbrew_experiment.py --fill-weights --graphs small
```

### "No graphs found"

```bash
# Download graphs first
python3 scripts/graphbrew_experiment.py --download-only --download-size SMALL

# Check graphs directory
ls -la graphs/
```

### "Invalid JSON"

```bash
# Validate JSON files
python3 -m json.tool scripts/weights/active/type_0.json
```

---

## Next Steps

- [[Running-Benchmarks]] - Command-line benchmark usage
- [[Python-Scripts]] - Full script documentation
- [[Perceptron-Weights]] - Weight system details
- [[AdaptiveOrder-ML]] - ML algorithm selection

---

[← Back to Home](Home) | [Python Scripts →](Python-Scripts)
