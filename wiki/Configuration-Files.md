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
└── lib/                       # Python library (5 sub-packages)
    ├── core/                  # Constants, logging, data stores
    │   ├── utils.py           # ⭐ SSOT: ALGORITHMS, BENCHMARKS, canonical_algo_key(), etc.
    │   ├── graph_types.py     # GraphInfo dataclass
    │   ├── datastore.py       # BenchmarkStore, GraphPropsStore
    │   └── graph_data.py      # Per-graph data storage
    ├── pipeline/              # Experiment execution stages
    │   ├── benchmark.py       # Benchmark execution
    │   ├── cache.py           # Cache simulation
    │   ├── reorder.py         # Vertex reordering
    │   ├── phases.py          # Phase orchestration
    │   └── ...                # build, download, progress
    ├── ml/                    # ML scoring & training
    │   ├── weights.py         # PerceptronWeight, compute_weights_from_results
    │   ├── eval_weights.py    # Data loading, evaluation
    │   ├── training.py        # Iterative/batched training
    │   └── ...                # adaptive_emulator, oracle, features
    ├── analysis/              # Post-run analysis
    │   ├── adaptive.py        # Adaptive analysis, A/B testing
    │   ├── metrics.py         # Amortization & end-to-end
    │   └── figures.py         # Plot generation
    └── tools/                 # Standalone CLI utilities
        ├── check_includes.py  # Scan C++ headers for legacy includes
        └── regen_features.py  # Regenerate graph features via C++ binary

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
├── data/                      # Runtime model storage (auto-created by ensure_prerequisites)
│   ├── adaptive_models.json   # Unified model store (DT, hybrid, kNN)
│   ├── benchmarks.json        # Aggregated benchmark data
│   └── graph_properties.json  # Graph feature database
│
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
| `--benchmarks` | pr, pr_spmv, bfs, cc, cc_sv, sssp, bc, tc | all 8 |
| `--trials` | Number of benchmark trials | 2 |
| `--quick` | Only test key algorithms | false |
| `--skip-cache` | Skip cache simulation | false |
| `--pregenerate-sg` | Pre-generate reordered `.sg` per algorithm (eliminates runtime reorder overhead) | true |
| `--no-pregenerate-sg` | Disable `.sg` pre-generation; reorder at runtime | false |
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

All trained perceptron weights are stored in `results/data/adaptive_models.json` under the `perceptron` key. The C++ runtime loads these via `LoadPerceptronWeightsFromDB()` in `reorder_database.h`.

Per-algorithm weights contain bias, feature weights, benchmark multipliers, and metadata. Run `--train` to generate actual weight values.

### Weight Field Descriptions

See [[Perceptron-Weights#weight-definitions]] for the complete 24-field reference including core weights, cache impacts, quadratic cross-terms, and per-benchmark multipliers.

---

## Results Files

### Graph Properties (data/graph_properties.json)

Graph feature vectors used by kNN and perceptron scoring. Managed by `GraphPropsStore` in `scripts/lib/core/datastore.py`. Populated automatically when C++ binaries run with `InitSelfRecording()`.

```json
{
  "graph_name": {
    "modularity": ...,
    "degree_variance": ...,
    "hub_concentration": ...,
    "clustering_coefficient": ...,
    "avg_degree": ...,
    "avg_path_length": ...,
    "diameter": ...,
    "community_count": ...,
    "nodes": ...,
    "edges": ...,
    "graph_type": "..."
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

### Unified Algorithm Naming (scripts/lib/core/utils.py)

Every subsystem that needs an algorithm name — weight files, `.sg` filenames, result JSON,
benchmark display — **MUST** use the canonical naming API:

```python
from scripts.lib.core.utils import canonical_algo_key, algo_converter_opt

# canonical_algo_key(algo_id, variant=None) → string key
canonical_algo_key(0)             # → "ORIGINAL"
canonical_algo_key(8)             # → "RABBITORDER_csr"      (default variant)
canonical_algo_key(8, "boost")    # → "RABBITORDER_boost"
canonical_algo_key(12, "leiden")  # → "GraphBrewOrder_leiden"
canonical_algo_key(11)            # → "RCM_default"

# algo_converter_opt(algo_id, variant=None) → "-o" argument for C++ binaries
algo_converter_opt(0)             # → "0"
algo_converter_opt(8, "boost")    # → "8:boost"
algo_converter_opt(12, "leiden")  # → "12:leiden"
```

These two functions are always used as a **pair**: `canonical_algo_key()` for the
human-readable key and `algo_converter_opt()` for the C++ command-line argument.
For variant algorithms (RabbitOrder, RCM, GOrder, GraphBrewOrder), the variant
suffix is **always** included — omitting the variant uses the registered default.

**Where the canonical key appears:**

| Context | Example |
|---------|-----------|
| Weight JSON key | `"RABBITORDER_csr": { "bias": 2.5, ... }` |
| `.sg` filename | `email-Enron_RABBITORDER_csr.sg` |
| `.lo` mapping file | `email-Enron_RABBITORDER_csr.lo` |
| Benchmark result field | `{ "algorithm": "RABBITORDER_csr", ... }` |
| Per-graph data dirs | `results/graphs/email-Enron/RABBITORDER_csr/` |

### Python Constants (scripts/lib/core/utils.py)

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

> ⚠️ **Important**: When changing defaults, update `reorder_types.h` (single source of truth). `reorder_graphbrew.h` and other headers reference these unified constants.

---

## Troubleshooting

### "Weight file not found"

```bash
# Check model store exists
ls -la results/data/adaptive_models.json

# Regenerate weights
python3 scripts/graphbrew_experiment.py --train --size small
```

### "No graphs found"

```bash
# Download graphs first
python3 scripts/graphbrew_experiment.py --download-only --size small

# Check graphs directory
ls -la results/graphs/
```

### "Invalid JSON"

```bash
# Validate JSON files
python3 -m json.tool results/data/adaptive_models.json
```

---

## Next Steps

- [[Running-Benchmarks]] - Command-line benchmark usage
- [[Python-Scripts]] - Full script documentation
- [[Perceptron-Weights]] - Weight system details
- [[AdaptiveOrder-ML]] - ML algorithm selection

---

[← Back to Home](Home) | [Python Scripts →](Python-Scripts)
