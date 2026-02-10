# GraphBrew Examples

This folder contains example scripts demonstrating how to use the GraphBrew library
for graph reordering experiments.

## Quick Start

### 1. Quick Test (`quick_test.py`)

The simplest way to test GraphBrew on a single graph:

```bash
# Basic usage
python scripts/examples/quick_test.py graphs/email-Enron

# With specific benchmarks
python scripts/examples/quick_test.py graphs/email-Enron --benchmarks pr bfs cc

# With specific algorithms
python scripts/examples/quick_test.py graphs/email-Enron --algorithms 0 1 8 15
```

### 2. Custom Pipeline (`custom_pipeline.py`)

Build customized experiment pipelines:

```bash
# Run full pipeline on a single graph
python scripts/examples/custom_pipeline.py --graph graphs/email-Enron

# Quick mode (fewer algorithms, iterations)
python scripts/examples/custom_pipeline.py --graph graphs/email-Enron --quick

# Compare specific algorithms
python scripts/examples/custom_pipeline.py --graph graphs/email-Enron --compare 8 15 17

# Select specific phases
python scripts/examples/custom_pipeline.py --graph graphs/email-Enron --phases reorder benchmark
```

### 3. Batch Processing (`batch_process.py`)

Process multiple graphs in batch:

```bash
# Process all graphs
python scripts/examples/batch_process.py graphs/ --output results/batch

# Only graphs matching a pattern
python scripts/examples/batch_process.py graphs/ --pattern "web-*"

# Resume interrupted processing
python scripts/examples/batch_process.py graphs/ --resume
```

### 4. Algorithm Comparison (`compare_algorithms.py`)

Compare reordering algorithms across graphs:

```bash
# Compare default algorithms on all graphs
python scripts/examples/compare_algorithms.py

# Specific graphs
python scripts/examples/compare_algorithms.py --graphs email-Enron web-Stanford

# Specific algorithms
python scripts/examples/compare_algorithms.py --algorithms 0 1 8 15 17

# Export results to CSV
python scripts/examples/compare_algorithms.py --output comparison.csv
```

## Algorithm Reference

| ID | Name | Description |
|----|------|-------------|
| 0 | ORIGINAL | No reordering (baseline) |
| 1 | RANDOM | Random permutation |
| 2 | SORT | Sort by degree |
| 3 | HUBSORT | Hub-based sorting |
| 4 | HUBCLUSTER | Hub clustering |
| 5 | DBG | Degree-based grouping |
| 6 | HUBSORTDBG | HubSort + DBG hybrid |
| 7 | HUBCLUSTERDBG | HubCluster + DBG (general-purpose) |
| 8 | RABBITORDER | Rabbit ordering (community-aware) |
| 9 | GORDER | Graph reordering |
| 10 | CORDER | Cache-optimized ordering |
| 11 | RCM | Reverse Cuthill-McKee |
| 12 | GraphBrewOrder | Per-community reordering |
| 13 | MAP | Load from external file |
| 14 | AdaptiveOrder | ML-powered algorithm selection |
| 15 | LeidenOrder | GVE-Leiden baseline reference |

## Benchmark Reference

| Name | Full Name | Description |
|------|-----------|-------------|
| pr | PageRank | Iterative graph ranking |
| bfs | Breadth-First Search | Level-synchronous traversal |
| cc | Connected Components | Component labeling |
| sssp | Single-Source Shortest Path | Dijkstra's algorithm |
| bc | Betweenness Centrality | Node importance metric |
| tc | Triangle Counting | Triangle enumeration |

## Writing Your Own Scripts

All examples import from the `lib` module:

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.phases import PhaseConfig, run_reorder_phase, run_benchmark_phase
from lib.progress import ProgressTracker
from lib import ALGORITHMS, BENCHMARKS
```

### Basic Pattern

```python
# 1. Create configuration
config = PhaseConfig(
    graph_path="graphs/my-graph/my-graph.mtx",
    output_dir="results/my-experiment",
    algorithms=[0, 8, 15],  # original, gorder, rcm
    benchmarks=['pr', 'bfs'],
    iterations=5,
    warmup=2
)

# 2. Run phases
run_reorder_phase(config)    # Generate reorderings
results = run_benchmark_phase(config)  # Run benchmarks

# 3. Process results
for r in results:
    print(f"{r.algorithm_name}: {r.avg_time:.4f}s")
```

## See Also

- [Python Scripts Documentation](../../wiki/Python-Scripts.md)
- [Phases Architecture](../../scripts/lib/phases.py)
- [Type Definitions](../../scripts/lib/graph_types.py)
