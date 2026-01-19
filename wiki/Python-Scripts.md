# Python Scripts Guide

Documentation for all Python tools in the GraphBrew framework.

## Overview

```
scripts/
├── graph_brew.py             # Main benchmark runner
├── graph_create.py           # Graph generation utilities
├── requirements.txt          # Python dependencies
├── perceptron_weights.json   # ML model weights
│
├── analysis/                 # Analysis tools
│   └── correlation_analysis.py
│
├── brew/                     # Experiment framework
│   └── run_experiment.py
│
├── config/                   # Configuration files
│   ├── brew/
│   ├── gap/
│   └── test/
│
└── test/                     # Test utilities
    └── run_experiment.py
```

---

## Installation

### Requirements

```bash
cd scripts
pip install -r requirements.txt
```

### requirements.txt Contents

```
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
scipy>=1.6.0
networkx>=2.5
tqdm>=4.50.0
```

---

## graph_brew.py - Main Runner

The primary script for running benchmark experiments.

### Basic Usage

```bash
python3 graph_brew.py --help
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input graph file(s) | Required |
| `--benchmark` | Benchmark(s) to run | `pr` |
| `--algorithms` | Algorithm IDs | `0 7` |
| `--trials` | Number of trials | `3` |
| `--output` | Output CSV file | `results.csv` |
| `--bin-dir` | Binary directory | `../bench/bin` |
| `--symmetrize` | Symmetrize graphs | `False` |
| `--verbose` | Verbose output | `False` |

### Examples

```bash
# Single graph, single benchmark
python3 graph_brew.py \
    --input ../graphs/facebook.el \
    --benchmark pr \
    --algorithms 0 7 12 20 \
    --trials 5

# Multiple graphs and benchmarks
python3 graph_brew.py \
    --input ../graphs/*.el \
    --benchmark pr bfs cc \
    --algorithms 0 7 12 15 20 \
    --trials 3 \
    --output results.csv

# With all options
python3 graph_brew.py \
    --input ../graphs/large.el \
    --benchmark pr bfs cc sssp bc tc \
    --algorithms 0 1 2 3 4 5 6 7 8 9 10 11 12 15 16 17 18 19 20 \
    --trials 10 \
    --symmetrize \
    --verbose \
    --output full_benchmark.csv
```

### Output Format

```csv
graph,benchmark,algorithm,trial,time,metric
facebook.el,pr,0,1,0.0234,
facebook.el,pr,0,2,0.0231,
facebook.el,pr,7,1,0.0189,
facebook.el,bfs,0,1,0.0012,76.9
```

---

## graph_create.py - Graph Generation

Generate synthetic graphs for testing.

### Usage

```bash
python3 graph_create.py --help
```

### Graph Types

| Type | Description | Parameters |
|------|-------------|------------|
| `random` | Erdős–Rényi random | `--nodes`, `--prob` |
| `barabasi` | Preferential attachment | `--nodes`, `--edges-per-node` |
| `watts` | Small-world | `--nodes`, `--k`, `--prob` |
| `tree` | Random tree | `--nodes` |
| `grid` | 2D grid | `--rows`, `--cols` |

### Examples

```bash
# Random graph with 1000 nodes
python3 graph_create.py \
    --type random \
    --nodes 1000 \
    --prob 0.01 \
    --output random_1k.el

# Scale-free (Barabási-Albert)
python3 graph_create.py \
    --type barabasi \
    --nodes 10000 \
    --edges-per-node 5 \
    --output scalefree_10k.el

# Small-world (Watts-Strogatz)
python3 graph_create.py \
    --type watts \
    --nodes 5000 \
    --k 10 \
    --prob 0.1 \
    --output smallworld_5k.el
```

---

## analysis/correlation_analysis.py

Analyze benchmark results and compute perceptron weights.

### Main Functions

#### 1. Load and Process Results

```python
from correlation_analysis import load_results, compute_graph_features

# Load benchmark results
results = load_results('results.csv')

# Compute graph features
features = compute_graph_features(graph_path)
```

#### 2. Compute Correlations

```python
from correlation_analysis import compute_correlations

# Analyze which features correlate with performance
correlations = compute_correlations(results, features)
print(correlations)
```

#### 3. Compute Perceptron Weights

```python
from correlation_analysis import compute_perceptron_weights

# Generate/update weights file
compute_perceptron_weights(
    results_df=results,
    weights_file='perceptron_weights.json'
)
```

### Command-Line Usage

```bash
# Full analysis
python3 correlation_analysis.py \
    --results ../results.csv \
    --graphs-dir ../graphs \
    --output-weights perceptron_weights.json

# Quick analysis
python3 correlation_analysis.py \
    --quick \
    --output-weights perceptron_weights.json
```

### Output: perceptron_weights.json

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

---

## brew/run_experiment.py

Structured experiment runner with configuration files.

### Usage

```bash
cd scripts/brew
python3 run_experiment.py --config ../config/brew/run.json
```

### Configuration File

```json
{
  "graphs_dir": "../graphs",
  "output_dir": "../results",
  "benchmarks": ["pr", "bfs", "cc"],
  "algorithms": [0, 7, 12, 20],
  "trials": 5,
  "options": {
    "symmetrize": true,
    "verify": false,
    "timeout": 3600
  }
}
```

### Features

- **Auto-discovery**: Finds all graphs in directory
- **Parallel execution**: Multiple graphs simultaneously
- **Progress tracking**: Shows completion status
- **Error handling**: Skips failed runs, continues
- **Result aggregation**: Merges all results

### Example Workflow

```bash
# 1. Setup config
cat > my_experiment.json << EOF
{
  "graphs_dir": "./my_graphs",
  "benchmarks": ["pr", "bfs"],
  "algorithms": [0, 7, 20],
  "trials": 3
}
EOF

# 2. Run experiment
python3 run_experiment.py --config my_experiment.json

# 3. Analyze results
python3 ../analysis/correlation_analysis.py \
    --results ./results/results.csv
```

---

## Configuration Files

### Location

```
scripts/config/
├── brew/           # GraphBrew experiments
│   ├── convert.json    # Format conversion
│   ├── label.json      # Graph labeling
│   └── run.json        # Benchmark runs
│
├── gap/            # GAP-compatible
│   └── ...
│
└── test/           # Quick tests
    └── ...
```

### config/brew/run.json

```json
{
  "description": "Standard GraphBrew benchmark configuration",
  "graphs_dir": "../../graphs",
  "output_dir": "../../results",
  
  "benchmarks": ["pr", "bfs", "cc", "tc"],
  "algorithms": [0, 5, 7, 12, 15, 20],
  "trials": 5,
  
  "options": {
    "symmetrize": true,
    "verify": false,
    "timeout": 3600,
    "root_vertex": -1
  },
  
  "filters": {
    "min_nodes": 100,
    "max_nodes": 100000000,
    "extensions": [".el", ".mtx"]
  }
}
```

### config/brew/convert.json

```json
{
  "description": "Graph format conversion settings",
  "input_formats": ["csv", "tsv", "mtx"],
  "output_format": "el",
  
  "transformations": {
    "remove_self_loops": true,
    "remove_multi_edges": true,
    "to_zero_indexed": true
  }
}
```

---

## Common Tasks

### Task 1: Run Quick Benchmark

```bash
cd scripts
python3 graph_brew.py \
    --input ../test/graphs/4.el \
    --benchmark pr \
    --algorithms 0 7 \
    --trials 1 \
    --symmetrize
```

### Task 2: Full Experiment Suite

```bash
cd scripts/brew

# Create config
cat > full_test.json << EOF
{
  "graphs_dir": "../../graphs",
  "benchmarks": ["pr", "bfs", "cc", "sssp", "bc", "tc"],
  "algorithms": [0, 7, 12, 15, 20],
  "trials": 10
}
EOF

# Run
python3 run_experiment.py --config full_test.json

# Analyze
python3 ../analysis/correlation_analysis.py \
    --results ../../results/results.csv \
    --output-weights ../perceptron_weights.json
```

### Task 3: Train Perceptron Weights

```bash
cd scripts/analysis

# Run comprehensive benchmark
python3 correlation_analysis.py \
    --graphs-dir ../../graphs \
    --benchmarks pr bfs cc \
    --algorithms 0 5 7 8 9 10 11 12 16 17 18 19 20 \
    --output-weights ../perceptron_weights.json
```

### Task 4: Generate Synthetic Graphs

```bash
cd scripts

# Create test graphs
for nodes in 1000 10000 100000; do
    python3 graph_create.py \
        --type barabasi \
        --nodes $nodes \
        --edges-per-node 10 \
        --output ../graphs/synthetic_${nodes}.el
done
```

### Task 5: Convert Graph Format

```bash
# MTX to EL
python3 -c "
import sys
with open('$1', 'r') as f:
    for line in f:
        if line.startswith('%'): continue
        if line.strip():
            parts = line.split()
            if len(parts) >= 2:
                print(int(parts[0])-1, int(parts[1])-1)
" > output.el
```

---

## Writing Custom Scripts

### Template

```python
#!/usr/bin/env python3
"""
Description of your script.
"""

import argparse
import subprocess
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='My script')
    parser.add_argument('--input', required=True, help='Input file')
    parser.add_argument('--output', default='output.csv', help='Output file')
    return parser.parse_args()

def run_benchmark(graph, algorithm, benchmark, trials=3):
    """Run a single benchmark."""
    cmd = [
        '../bench/bin/' + benchmark,
        '-f', graph,
        '-o', str(algorithm),
        '-n', str(trials),
        '-s'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return parse_output(result.stdout)

def parse_output(output):
    """Parse benchmark output."""
    # Extract timing information
    for line in output.split('\n'):
        if 'Average' in line:
            return float(line.split()[-2])
    return None

def main():
    args = parse_args()
    
    # Your logic here
    results = []
    for algo in [0, 7, 12, 20]:
        time = run_benchmark(args.input, algo, 'pr')
        results.append({'algorithm': algo, 'time': time})
    
    # Save results
    import pandas as pd
    pd.DataFrame(results).to_csv(args.output, index=False)

if __name__ == '__main__':
    main()
```

### Using the Modules

```python
# Import from correlation_analysis
import sys
sys.path.append('scripts/analysis')
from correlation_analysis import compute_graph_features, load_results

# Compute features for a graph
features = compute_graph_features('graphs/my_graph.el')
print(f"Nodes: {features['num_nodes']}")
print(f"Edges: {features['num_edges']}")
print(f"Density: {features['density']:.6f}")
print(f"Modularity: {features['modularity']:.4f}")
```

---

## Troubleshooting

### Import Errors

```bash
# Install missing packages
pip install -r requirements.txt

# Check Python version
python3 --version  # Should be 3.6+
```

### Binary Not Found

```bash
# Check binaries exist
ls -la ../bench/bin/

# Build if missing
cd ../bench && make all
```

### Permission Denied

```bash
chmod +x ../bench/bin/*
```

### Slow Execution

```bash
# Use fewer trials for testing
--trials 1

# Use smaller graphs
--input test/graphs/4.el
```

### Memory Issues

```python
# Process graphs one at a time
for graph in graphs:
    result = process(graph)
    save(result)
    del result  # Free memory
```

---

## Next Steps

- [[AdaptiveOrder-ML]] - ML perceptron details
- [[Running-Benchmarks]] - Command-line usage
- [[Code-Architecture]] - Overall structure

---

[← Back to Home](Home) | [AdaptiveOrder ML →](AdaptiveOrder-ML)
