# Benchmark Suite

The GraphBrew Benchmark Suite provides automated tools for running comprehensive experiments across multiple graphs, algorithms, and benchmarks.

## Overview

```
scripts/
├── graphbrew_experiment.py     # ⭐ MAIN: One-click unified pipeline
│                                #    Downloads, builds, benchmarks, analyzes
├── requirements.txt            # Python dependencies
├── lib/                        # 📦 Core modules (all functionality)
│   ├── download.py             # Graph downloading
│   ├── benchmark.py            # Benchmark execution
│   ├── cache.py                # Cache simulation
│   ├── weights.py              # Weight management
│   ├── training.py             # ML training
│   ├── features.py             # Graph feature extraction
│   └── ...                     # Other modules
└── weights/                    # Auto-clustered type weights
    ├── active/                 # C++ reads from here
    ├── merged/                 # Accumulated weights
    └── runs/                   # Historical snapshots
```

---

## 🚀 Quick Start (One-Click Recommended)

The unified script handles everything automatically:

```bash
# Full pipeline: download → build → benchmark → analyze → train
python3 scripts/graphbrew_experiment.py --full --download-size SMALL

# Just run benchmarks on existing graphs
python3 scripts/graphbrew_experiment.py --phase benchmark

# Quick test with key algorithms
python3 scripts/graphbrew_experiment.py --size small --quick

# Skip cache simulations (faster)
python3 scripts/graphbrew_experiment.py --phase all --skip-cache

# Run brute-force validation
python3 scripts/graphbrew_experiment.py --brute-force

# Complete training pipeline: reorder → benchmark → cache sim → update weights
python3 scripts/graphbrew_experiment.py --train --size small --max-graphs 5
```

### Download Size Options

| Size | Graphs | Total Size | Use Case |
|------|--------|------------|----------|
| `SMALL` | 16 | ~62 MB | Quick testing |
| `MEDIUM` | 28 | ~1.1 GB | Standard experiments |
| `LARGE` | 37 | ~25 GB | Full evaluation |
| `XLARGE` | 6 | ~63 GB | Massive-scale testing |
| `ALL` | 87 | ~89 GB | Complete benchmark |

### Graph Categories

The graph catalog includes diverse graph types:

| Category | Examples | Count |
|----------|----------|-------|
| Mesh | mesh networks, finite element | 16 |
| Web | uk-2002, webbase-2001, web-Google | 15 |
| Collaboration | hollywood-2009, com-DBLP, ca-AstroPh | 11 |
| Social | soc-LiveJournal, com-Orkut | 10 |
| Synthetic | RMAT, uniform random | 8 |
| Road | roadNet-CA, europe-osm, USA-road-d | 7 |
| Citation | cit-Patents, cit-HepPh | 5 |
| Commerce | amazon0601, com-Amazon | 5 |
| P2P | p2p-Gnutella24/25/30/31 | 4 |
| Communication | email-Enron, wiki-Talk | 3 |
| Infrastructure | as-Skitter | 3 |

All results are saved to `./results/`:
- `reorder_*.json` - Reordering times
- `benchmark_*.json` - Execution times  
- `cache_*.json` - Cache miss rates (L1/L2/L3)
- `mappings/` - Pre-generated label maps for consistent reordering
- `graph_properties_cache.json` - Cached graph properties for type detection

Auto-clustered weights are saved to `./scripts/weights/active/`:
- `type_registry.json` - Maps graph names to type IDs + cluster centroids
- `type_N.json` - Per-cluster trained ML weights (type_0.json, type_1.json, etc.)

---

## Alternative Approach (Manual)

If you prefer more control, you can use specific phases of the unified script:

### 1. Download Graphs

```bash
# List available graphs
python3 scripts/graphbrew_experiment.py --download-only --download-size SMALL

# Download specific size category
python3 scripts/graphbrew_experiment.py --download-only --download-size MEDIUM

# Force re-download
python3 scripts/graphbrew_experiment.py --download-only --force-download
```

### 2. Generate Label Maps (Consistent Reordering)

```bash
# Pre-generate label maps for all algorithms
python3 scripts/graphbrew_experiment.py --generate-maps --graphs small
```

### 3. Run Specific Phase

```bash
# Just reordering phase (measure reorder times)
python3 scripts/graphbrew_experiment.py --phase reorder --graphs small

# Just benchmarks (skip cache simulation)
python3 scripts/graphbrew_experiment.py --phase benchmark --graphs small --skip-cache

# Just cache simulation
python3 scripts/graphbrew_experiment.py --phase cache --graphs small

# Just weight generation (from existing results)
python3 scripts/graphbrew_experiment.py --phase weights
```

### 4. Custom Configuration

```bash
# Specify graph size range
python3 scripts/graphbrew_experiment.py --min-size 10 --max-size 500

# Limit number of graphs
python3 scripts/graphbrew_experiment.py --max-graphs 10

# Custom trials
python3 scripts/graphbrew_experiment.py --trials 10

# Key algorithms only (faster)
python3 scripts/graphbrew_experiment.py --quick
```

---

## Output Format

### Benchmark Results (benchmark_*.json)

```json
[
  {
    "graph": "email-Enron",
    "algorithm": "LeidenCSR",
    "algorithm_id": 17,
    "benchmark": "pr",
    "time_seconds": 0.0234,
    "reorder_time": 0.0,
    "trials": 2,
    "success": true,
    "error": "",
    "extra": {}
  }
]
```

### Cache Results (cache_*.json)

```json
[
  {
    "graph": "email-Enron",
    "algorithm_id": 17,
    "algorithm_name": "LeidenCSR",
    "benchmark": "pr",
    "l1_hit_rate": 85.2,
    "l2_hit_rate": 92.1,
    "l3_hit_rate": 98.5,
    "l1_misses": 0,
    "l2_misses": 0,
    "l3_misses": 0,
    "success": true,
    "error": ""
  }
]
```

### Reorder Results (reorder_*.json)

```json
[
  {
    "graph": "email-Enron",
    "algorithm_id": 17,
    "algorithm_name": "LeidenCSR",
    "time_seconds": 0.145,
    "mapping_file": "results/mappings/email-Enron/LeidenCSR.lo",
    "success": true,
    "error": ""
  }
]
```

### Perceptron Weights (scripts/weights/active/type_N.json)

```json
{
  "LeidenCSR": {
    "bias": 0.85,
    "w_modularity": 0.25,
    "w_log_nodes": 0.1,
    "w_log_edges": 0.1,
    "w_density": -0.05,
    "w_avg_degree": 0.15,
    "w_degree_variance": 0.15,
    "w_hub_concentration": 0.25,
    "cache_l1_impact": 0.1,
    "cache_l2_impact": 0.05,
    "cache_l3_impact": 0.02,
    "cache_dram_penalty": -0.1,
    "w_reorder_time": -0.0001,
    "_metadata": {
      "win_rate": 0.85,
      "avg_speedup": 2.34,
      "times_best": 42,
      "sample_count": 50,
      "avg_reorder_time": 1.234,
      "avg_l1_hit_rate": 85.2,
      "avg_l2_hit_rate": 92.1,
      "avg_l3_hit_rate": 98.5
    }
  }
}
```

---

## PageRank Convergence Analysis

Analyze how reordering affects PageRank convergence.

### Usage

Run PageRank directly via the binary with verbose output:

```bash
# Run PR with verbose convergence output
./bench/bin/pr -f graph.mtx -s -o 7 -n 5
```

Or include in the experiment pipeline:

```bash
# Run benchmarks (includes convergence data in results)
python3 scripts/graphbrew_experiment.py --phase benchmark --graphs small
```

### Example Output

PageRank convergence varies by reordering algorithm:

```
Graph: facebook.el
┌────────────────────┬────────────┬──────────────┐
│ Algorithm          │ Iterations │ Final Error  │
├────────────────────┼────────────┼──────────────┤
│ ORIGINAL (0)       │ 18         │ 9.2e-7       │
│ HUBCLUSTERDBG (7)  │ 16         │ 8.8e-7       │
│ LeidenOrder (15)   │ 15         │ 9.1e-7       │
│ LeidenCSR (17)     │ 14         │ 8.5e-7       │
└────────────────────┴────────────┴──────────────┘
```

---

## Experiment Workflow

### Complete Reproducible Experiment (One Command)

```bash
# One-click full experiment
python3 scripts/graphbrew_experiment.py --full --size medium
```

This automatically:
1. Downloads graphs from SuiteSparse
2. Builds binaries
3. Generates label maps for consistent reordering
4. Runs all benchmarks with all algorithms
5. Runs cache simulations
6. Trains perceptron weights with cache + reorder time features

### Custom Experiment Workflow

```bash
#!/bin/bash
# Custom experiment workflow

cd GraphBrew

# 1. Download specific size graphs
python3 scripts/graphbrew_experiment.py --download-only --download-size MEDIUM

# 2. Generate label maps once
python3 scripts/graphbrew_experiment.py --generate-maps --graphs medium

# 3. Run benchmarks using pre-generated maps
python3 scripts/graphbrew_experiment.py --use-maps --phase benchmark --graphs medium --trials 10

# 4. Run cache simulations
python3 scripts/graphbrew_experiment.py --use-maps --phase cache --graphs medium

# 5. Generate perceptron weights
python3 scripts/graphbrew_experiment.py --phase weights

# 6. Run brute-force validation
python3 scripts/graphbrew_experiment.py --brute-force --graphs medium
```

---

## Parallel Execution

### Thread Control

```bash
# Control OpenMP threads
export OMP_NUM_THREADS=8
python3 scripts/graphbrew_experiment.py --full --download-size SMALL
```

### Skip Heavy Operations

```bash
# Skip slow algorithms on large graphs
python3 scripts/graphbrew_experiment.py --skip-slow --size large

# Skip expensive cache simulations (BC, SSSP)
python3 scripts/graphbrew_experiment.py --skip-expensive --phase cache
```

---

## Troubleshooting

### "Graph not found"

```bash
# Check downloaded graphs
ls results/graphs/

# Re-download with force flag
python3 scripts/graphbrew_experiment.py --download-only --force-download
```

### Timeout Issues

```bash
# Increase timeouts
python3 scripts/graphbrew_experiment.py \
    --timeout-reorder 86400 \
    --timeout-benchmark 7200 \
    --timeout-sim 14400
```

### Memory Issues

```bash
# Use smaller graphs
python3 scripts/graphbrew_experiment.py --graphs small

# Skip large graphs
python3 scripts/graphbrew_experiment.py --max-size 500
```

### Clean Start

```bash
# Clean results only (keep graphs)
python3 scripts/graphbrew_experiment.py --clean

# Full reset (remove everything including downloaded graphs)
python3 scripts/graphbrew_experiment.py --clean-all
```

---

## Next Steps

- [[Correlation-Analysis]] - Analyze benchmark results
- [[AdaptiveOrder-ML]] - Train the perceptron
- [[Running-Benchmarks]] - Manual benchmark commands
- [[Python-Scripts]] - Full script documentation

---

[← Back to Home](Home) | [Correlation Analysis →](Correlation-Analysis)
