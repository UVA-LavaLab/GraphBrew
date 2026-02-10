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
    ├── utils.py               # ⭐ Single source of truth for constants:
    │                          #    ALGORITHMS, BENCHMARKS, SIZE_*, TIMEOUT_*,
    │                          #    GRAPHBREW_VARIANTS, etc.
    ├── benchmark.py           # Benchmark execution
    ├── cache.py               # Cache simulation
    ├── weights.py             # Weight management
    ├── features.py            # Graph feature extraction
    ├── graph_data.py          # Per-graph data storage
    └── ...                    # Additional modules

results/
├── graphs/                    # Static per-graph features
│   └── {graph_name}/
│       └── features.json      # Graph topology (nodes, edges, modularity, etc.)
│
├── logs/                      # Run-specific data and command logs
│   └── {graph_name}/
│       ├── runs/{timestamp}/  # Timestamped experiment data
│       │   ├── benchmarks/    # Per-algorithm benchmark results
│       │   ├── reorder/       # Reorder times and mapping info
│       │   └── weights/       # Computed perceptron weights
│       └── *.log              # Individual operation logs
│
├── graph_properties_cache.json # Cached graph features
├── benchmark_*.json           # Aggregate benchmark result files
├── cache_*.json               # Aggregate cache simulation results
└── reorder_*.json             # Aggregate reorder timing results
```

---

## Command-Line Configuration

GraphBrew uses command-line arguments instead of JSON config files:

### Basic Usage

```bash
# Full pipeline with defaults
python3 scripts/graphbrew_experiment.py --full --size small

# Customize experiments via command line
python3 scripts/graphbrew_experiment.py \
    --phase benchmark \
    --size small \
    --benchmarks pr bfs cc \
    --trials 5
```

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--full` | Run complete pipeline | - |
| `--size` | small, medium, large, xlarge, all | all |
| `--phase` | all, reorder, benchmark, cache, weights, adaptive | all |
| `--benchmarks` | pr, bfs, cc, sssp, bc, tc | all 6 |
| `--trials` | Number of benchmark trials | 2 |
| `--quick` | Only test key algorithms | false |
| `--skip-cache` | Skip cache simulation | false |
| `--train` | Complete training pipeline | false |

### Memory and Disk Limits

```bash
# Auto-detect available resources
python3 scripts/graphbrew_experiment.py --full --auto

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
    "centroid": [0.479, 0.357, 0.426, 0.025, 0.5, 0.6, 0.7],
    "graph_count": 4
  }
}
```

Centroid array indices: [modularity, degree_variance, hub_concentration, avg_degree, clustering_coefficient, log_nodes, log_edges]
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
  "GraphBrewOrder": {
    "bias": 3.2,
    "w_modularity": 0.20,
    "w_log_nodes": 0.01
  }
}
```

### Weight Field Descriptions

See [[Perceptron-Weights#weight-definitions]] for the complete 24-field reference including core weights, cache impacts, quadratic cross-terms, and per-benchmark multipliers.

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

See [[Command-Line-Reference#environment-variables]] for `PERCEPTRON_WEIGHTS_FILE`, `OMP_NUM_THREADS`, and NUMA binding.

---

## Example Workflows

```bash
python3 scripts/graphbrew_experiment.py --phase benchmark --size small --quick --trials 1  # Quick test
python3 scripts/graphbrew_experiment.py --full --size medium --trials 5 --auto              # Full benchmark
python3 scripts/graphbrew_experiment.py --train --skip-cache --size small                   # Weight training
python3 scripts/graphbrew_experiment.py --brute-force --size small                          # Validation
```

See [[Command-Line-Reference]] for all options.

---

## Single Source of Truth (SSOT) Constants

All tunable constants are defined in **one location** to ensure consistency between C++ and Python.

### Python Constants (scripts/lib/utils.py)

```python
# Unified Reorder Configuration (match C++ reorder::ReorderConfig in reorder_types.h)
# Used by: GraphBrew, Leiden, GraphBrew, RabbitOrder, Adaptive
REORDER_DEFAULT_RESOLUTION = 1.0          # Modularity resolution (auto-computed from graph)
REORDER_DEFAULT_TOLERANCE = 1e-2          # Convergence tolerance (0.01)
REORDER_DEFAULT_AGGREGATION_TOLERANCE = 0.8
REORDER_DEFAULT_TOLERANCE_DROP = 10.0     # Tolerance reduction per pass
REORDER_DEFAULT_MAX_ITERATIONS = 10       # Max iterations per pass
REORDER_DEFAULT_MAX_PASSES = 10           # Max aggregation passes

# Backward-compatible aliases
LEIDEN_DEFAULT_RESOLUTION = REORDER_DEFAULT_RESOLUTION
LEIDEN_DEFAULT_MAX_ITERATIONS = REORDER_DEFAULT_MAX_ITERATIONS
LEIDEN_MODULARITY_MAX_ITERATIONS = 20     # Quality-focused mode
LEIDEN_MODULARITY_MAX_PASSES = 20

# Weight Computation Normalization
WEIGHT_PATH_LENGTH_NORMALIZATION = 10.0   # Normalize avg_path_length
WEIGHT_REORDER_TIME_NORMALIZATION = 10.0  # Normalize reorder_time penalty
WEIGHT_AVG_DEGREE_DEFAULT = 10.0          # Default avg_degree fallback

# Timeout Constants (seconds)
TIMEOUT_REORDER = 43200      # 12 hours (GOrder can be slow)
TIMEOUT_BENCHMARK = 600      # 10 minutes
TIMEOUT_SIM = 1200           # 20 minutes
TIMEOUT_SIM_HEAVY = 3600     # 1 hour (bc, sssp simulations)
```

### C++ Constants (bench/include/graphbrew/reorder/reorder_types.h)

```cpp
namespace reorder {
// Unified defaults for ALL community-based reordering
constexpr double DEFAULT_RESOLUTION = 1.0;           // Modularity (auto-computed)
constexpr double DEFAULT_TOLERANCE = 1e-2;           // Convergence
constexpr double DEFAULT_AGGREGATION_TOLERANCE = 0.8;
constexpr double DEFAULT_TOLERANCE_DROP = 10.0;      // Per-pass reduction
constexpr int DEFAULT_MAX_ITERATIONS = 10;           // Per-pass limit
constexpr int DEFAULT_MAX_PASSES = 10;               // Total passes

struct ReorderConfig {
    ResolutionMode resolutionMode = ResolutionMode::AUTO;
    double resolution = DEFAULT_RESOLUTION;
    int maxIterations = DEFAULT_MAX_ITERATIONS;
    int maxPasses = DEFAULT_MAX_PASSES;
    OrderingStrategy ordering = OrderingStrategy::HIERARCHICAL;
    // ... full config with FromOptions(), applyAutoResolution()
};
}
```

> ⚠️ **Important**: When changing defaults, update `reorder_types.h` (single source of truth). `reorder_leiden.h` and other headers reference these unified constants.

---

## Troubleshooting

### "Weight file not found"

```bash
# Check weight files exist
ls -la scripts/weights/active/

# Regenerate weights
python3 scripts/graphbrew_experiment.py --train --size small
```

### "No graphs found"

```bash
# Download graphs first
python3 scripts/graphbrew_experiment.py --download-only --size small

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
