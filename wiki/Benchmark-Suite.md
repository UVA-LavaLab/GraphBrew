# Benchmark Suite

The GraphBrew Benchmark Suite provides automated tools for running comprehensive experiments across multiple graphs, algorithms, and benchmarks.

## Overview

```
scripts/
├── benchmark/
│   ├── run_benchmark.py              # Main benchmark runner
│   └── run_pagerank_convergence.py   # PageRank iteration analysis
├── download/
│   └── download_graphs.py            # Graph downloader
└── analysis/
    └── correlation_analysis.py       # Results analysis
```

---

## Quick Start

### 1. Download Graphs

```bash
# List available graphs
python3 scripts/download/download_graphs.py --list

# Download medium-sized graphs (~600MB)
python3 scripts/download/download_graphs.py --size MEDIUM --output-dir ./graphs
```

### 2. Run Benchmarks

```bash
# Quick test
python3 scripts/benchmark/run_benchmark.py --quick --benchmark pr bfs

# Full benchmark
python3 scripts/benchmark/run_benchmark.py \
    --graphs-config ./graphs/graphs.json \
    --benchmark pr bfs cc \
    --algorithms 0,7,12,15,20 \
    --trials 5
```

### 3. Analyze Results

```bash
python3 scripts/analysis/correlation_analysis.py \
    --graphs-dir ./graphs \
    --benchmark pr bfs
```

---

## run_benchmark.py

The main benchmark runner for comprehensive experiments.

### Basic Usage

```bash
python3 scripts/benchmark/run_benchmark.py [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--graphs-config` | Path to graphs.json | Auto-detect |
| `--graphs-dir` | Directory with graph files | `./graphs` |
| `--benchmark` | Benchmarks to run | `pr` |
| `--algorithms` | Algorithm IDs (comma-separated) | `0,7,12,20` |
| `--trials` | Number of trials per run | `5` |
| `--output` | Output JSON file | `results.json` |
| `--quick` | Quick test with synthetic graphs | False |
| `--timeout` | Timeout per run (seconds) | `3600` |

### Examples

#### Quick Validation Test
```bash
python3 scripts/benchmark/run_benchmark.py \
    --quick \
    --benchmark pr \
    --algorithms 0,7,20 \
    --trials 1
```

#### Full Experiment Suite
```bash
python3 scripts/benchmark/run_benchmark.py \
    --graphs-config ./graphs/graphs.json \
    --benchmark pr bfs cc sssp bc tc \
    --algorithms 0,1,2,3,4,5,6,7,8,9,10,11,12,15,16,17,18,19,20 \
    --trials 16 \
    --output ./bench/results/full_benchmark.json
```

#### Specific Algorithms on Specific Graphs
```bash
python3 scripts/benchmark/run_benchmark.py \
    --graphs-dir ./graphs \
    --benchmark pr bfs \
    --algorithms 0,12,20 \
    --trials 10
```

### Output Format

Results are saved as JSON:

```json
{
  "metadata": {
    "date": "2026-01-18",
    "trials": 5,
    "benchmarks": ["pr", "bfs"],
    "algorithms": [0, 7, 12, 20]
  },
  "results": {
    "facebook": {
      "pr": {
        "0": {"mean": 0.0523, "std": 0.002, "times": [0.051, 0.053, ...]},
        "7": {"mean": 0.0412, "std": 0.001, "times": [0.041, 0.042, ...]},
        "20": {"mean": 0.0371, "std": 0.001, "times": [0.037, 0.038, ...]}
      },
      "bfs": {
        "0": {"mean": 0.0089, "std": 0.001, "mteps": 9923.4},
        ...
      }
    }
  }
}
```

---

## download_graphs.py

Download benchmark graphs from SuiteSparse Matrix Collection.

### Usage

```bash
python3 scripts/download/download_graphs.py [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--list` | List available graphs |
| `--size` | Size category (SMALL, MEDIUM, LARGE, ALL) |
| `--output-dir` | Output directory |
| `--validate` | Validate downloaded graphs |
| `--graph` | Download specific graph by name |

### Size Categories

| Category | Graphs | Download Size | Description |
|----------|--------|---------------|-------------|
| SMALL | 4 | ~12MB | Quick testing |
| MEDIUM | 17 | ~600MB | Development |
| MID_LARGE | 8 | ~4GB | Serious testing |
| LARGE | 5 | ~72GB | Full experiments |
| XL | 3 | ~150GB | Large-scale |

### Examples

```bash
# List all available graphs
python3 scripts/download/download_graphs.py --list

# Download small graphs for testing
python3 scripts/download/download_graphs.py --size SMALL --output-dir ./graphs

# Download specific graph
python3 scripts/download/download_graphs.py --graph twitter --output-dir ./graphs

# Validate downloads
python3 scripts/download/download_graphs.py --validate --output-dir ./graphs
```

### graphs.json

After download, a `graphs.json` config is auto-generated:

```json
{
  "graphs": {
    "facebook": {
      "path": "./graphs/facebook/graph.el",
      "nodes": 4039,
      "edges": 88234,
      "format": "el",
      "symmetric": true
    },
    "twitter": {
      "path": "./graphs/twitter/graph.mtx",
      "nodes": 41652230,
      "edges": 1468365182,
      "format": "mtx",
      "symmetric": false
    }
  }
}
```

---

## run_pagerank_convergence.py

Analyze how reordering affects PageRank convergence.

### Usage

```bash
python3 scripts/benchmark/run_pagerank_convergence.py \
    --graphs-config ./graphs/graphs.json \
    --algorithms 0,7,12,20
```

### Output

Shows iteration counts per algorithm:

```
PageRank Convergence Analysis
=============================

Graph: facebook.el
┌────────────────────┬────────────┬──────────────┐
│ Algorithm          │ Iterations │ Final Error  │
├────────────────────┼────────────┼──────────────┤
│ ORIGINAL (0)       │ 18         │ 9.2e-7       │
│ HUBCLUSTERDBG (7)  │ 16         │ 8.8e-7       │
│ LeidenOrder (12)   │ 15         │ 9.1e-7       │
│ LeidenHybrid (20)  │ 14         │ 8.5e-7       │
└────────────────────┴────────────┴──────────────┘
```

---

## Experiment Workflow

### Complete Reproducible Experiment

```bash
#!/bin/bash
# Full experiment workflow

# 1. Setup
cd GraphBrew
source .venv/bin/activate

# 2. Download graphs
python3 scripts/download/download_graphs.py \
    --size MEDIUM \
    --output-dir ./graphs

# 3. Run benchmarks
python3 scripts/benchmark/run_benchmark.py \
    --graphs-config ./graphs/graphs.json \
    --benchmark pr bfs cc tc \
    --algorithms 0,7,12,15,16,17,18,19,20 \
    --trials 10 \
    --output ./bench/results/experiment_$(date +%Y%m%d).json

# 4. Analyze results
python3 scripts/analysis/correlation_analysis.py \
    --graphs-dir ./graphs \
    --benchmark pr bfs

# 5. Generate summary
python3 -c "
import json
with open('./bench/results/experiment_*.json') as f:
    data = json.load(f)
    for graph, results in data['results'].items():
        print(f'\n{graph}:')
        for bench, algos in results.items():
            best = min(algos.items(), key=lambda x: x[1]['mean'])
            print(f'  {bench}: Best={best[0]} ({best[1][\"mean\"]:.4f}s)')
"
```

### Multi-Source Benchmarks

For BFS, SSSP, and BC, the suite automatically tests multiple source vertices:

```bash
python3 scripts/benchmark/run_benchmark.py \
    --benchmark bfs sssp bc \
    --trials 16  # 16 different source vertices
```

Output includes:
- Mean time across all sources
- Standard deviation
- MTEPS (Million Traversed Edges Per Second) for BFS

---

## Configuration Files

### Experiment Config (optional)

Create custom experiment configs:

```json
{
  "name": "leiden_comparison",
  "description": "Compare Leiden variants",
  "graphs": ["facebook", "twitter", "web-Google"],
  "benchmarks": ["pr", "bfs"],
  "algorithms": [0, 12, 16, 17, 18, 19, 20],
  "trials": 10,
  "options": {
    "symmetrize": true,
    "timeout": 3600
  }
}
```

Run with:
```bash
python3 scripts/benchmark/run_benchmark.py --config experiment.json
```

---

## Parallel Execution

### Running on Multiple Machines

Split workload across machines:

```bash
# Machine 1: Small/Medium graphs
python3 scripts/benchmark/run_benchmark.py \
    --graphs-dir ./graphs \
    --size SMALL,MEDIUM \
    --output results_small.json

# Machine 2: Large graphs
python3 scripts/benchmark/run_benchmark.py \
    --graphs-dir ./graphs \
    --size LARGE \
    --output results_large.json
```

### Thread Control

```bash
# Control OpenMP threads
export OMP_NUM_THREADS=8
python3 scripts/benchmark/run_benchmark.py ...
```

---

## Troubleshooting

### "Graph not found"

```bash
# Check graphs.json exists
cat ./graphs/graphs.json

# Regenerate config
python3 scripts/download/download_graphs.py --validate --output-dir ./graphs
```

### Timeout Issues

```bash
# Increase timeout for large graphs
python3 scripts/benchmark/run_benchmark.py \
    --timeout 7200 \
    ...
```

### Memory Issues

```bash
# Skip large graphs
python3 scripts/benchmark/run_benchmark.py \
    --size SMALL,MEDIUM \
    ...
```

---

## Next Steps

- [[Correlation-Analysis]] - Analyze benchmark results
- [[AdaptiveOrder-ML]] - Train the perceptron
- [[Running-Benchmarks]] - Manual benchmark commands

---

[← Back to Home](Home) | [Correlation Analysis →](Correlation-Analysis)
