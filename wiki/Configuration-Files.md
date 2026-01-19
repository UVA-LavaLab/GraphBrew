# Configuration Files

Guide to GraphBrew configuration files for automated experiments.

---

## Overview

Configuration files control automated experiments via Python scripts:

```
scripts/config/
├── brew/               # Standard GraphBrew configs
│   ├── run.json        # Benchmark run settings
│   ├── convert.json    # Format conversion
│   └── label.json      # Graph labeling
│
├── gap/                # GAP suite compatible
│   └── ...
│
└── test/               # Quick testing
    └── ...
```

---

## run.json - Benchmark Configuration

The main configuration for running experiments.

### Full Structure

```json
{
  "description": "Experiment description",
  
  "paths": {
    "graphs_dir": "../../graphs",
    "output_dir": "../../results",
    "bin_dir": "../../bench/bin"
  },
  
  "benchmarks": ["pr", "bfs", "cc", "sssp", "bc", "tc"],
  
  "algorithms": [0, 7, 12, 15, 20],
  
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
    "extensions": [".el", ".mtx", ".gr"]
  },
  
  "output": {
    "format": "csv",
    "include_raw": true,
    "include_summary": true
  }
}
```

### Field Descriptions

#### paths

| Field | Description | Default |
|-------|-------------|---------|
| `graphs_dir` | Directory containing graph files | Required |
| `output_dir` | Where to save results | `./results` |
| `bin_dir` | Location of benchmark binaries | `../../bench/bin` |

#### benchmarks

List of benchmarks to run:

```json
"benchmarks": ["pr", "bfs", "cc", "sssp", "bc", "tc"]
```

#### algorithms

List of reordering algorithm IDs:

```json
"algorithms": [0, 7, 12, 15, 20]
```

See [[Command-Line-Reference]] for all IDs.

#### options

| Option | Description | Default |
|--------|-------------|---------|
| `symmetrize` | Make graphs undirected | `true` |
| `verify` | Verify results | `false` |
| `timeout` | Max seconds per run | `3600` |
| `root_vertex` | BFS/SSSP root (-1 = random) | `-1` |

#### filters

| Filter | Description | Default |
|--------|-------------|---------|
| `min_nodes` | Skip small graphs | `0` |
| `max_nodes` | Skip huge graphs | `∞` |
| `extensions` | File types to include | `[".el"]` |

---

## Example Configurations

### Minimal Config

```json
{
  "graphs_dir": "./graphs",
  "benchmarks": ["pr"],
  "algorithms": [0, 7],
  "trials": 3
}
```

### Quick Test Config

```json
{
  "description": "Quick functionality test",
  "paths": {
    "graphs_dir": "../../test/graphs"
  },
  "benchmarks": ["pr", "bfs"],
  "algorithms": [0, 7, 12],
  "trials": 1,
  "options": {
    "symmetrize": true,
    "verify": true,
    "timeout": 60
  }
}
```

### Full Benchmark Config

```json
{
  "description": "Complete benchmark suite",
  "paths": {
    "graphs_dir": "../../graphs",
    "output_dir": "../../results/full_benchmark"
  },
  "benchmarks": ["pr", "bfs", "cc", "sssp", "bc", "tc"],
  "algorithms": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20],
  "trials": 10,
  "options": {
    "symmetrize": true,
    "verify": false,
    "timeout": 7200
  },
  "filters": {
    "min_nodes": 1000,
    "extensions": [".el", ".mtx"]
  }
}
```

### Leiden-Only Config

```json
{
  "description": "Test Leiden-based algorithms",
  "benchmarks": ["pr", "tc"],
  "algorithms": [0, 12, 16, 17, 18, 19, 20],
  "trials": 5
}
```

---

## convert.json - Format Conversion

Settings for converting between graph formats.

```json
{
  "description": "Graph format conversion settings",
  
  "input": {
    "formats": ["csv", "tsv", "mtx", "gr"],
    "delimiter": "auto"
  },
  
  "output": {
    "format": "el",
    "dir": "./converted"
  },
  
  "transformations": {
    "remove_self_loops": true,
    "remove_multi_edges": true,
    "to_zero_indexed": true,
    "symmetrize": false
  }
}
```

### Transformation Options

| Option | Description | Default |
|--------|-------------|---------|
| `remove_self_loops` | Remove edges v→v | `true` |
| `remove_multi_edges` | Keep only first edge | `true` |
| `to_zero_indexed` | Convert 1-indexed to 0-indexed | `true` |
| `symmetrize` | Add reverse edges | `false` |

---

## label.json - Graph Labeling

Settings for generating graph metadata.

```json
{
  "description": "Graph labeling configuration",
  
  "metrics": [
    "num_nodes",
    "num_edges",
    "density",
    "avg_degree",
    "max_degree",
    "modularity",
    "diameter",
    "clustering_coefficient"
  ],
  
  "output": {
    "format": "json",
    "file": "graph_metadata.json"
  },
  
  "options": {
    "compute_modularity": true,
    "sample_diameter": true,
    "sample_size": 100
  }
}
```

---

## Using Configurations

### With run_experiment.py

```bash
cd scripts/brew
python3 run_experiment.py --config ../config/brew/run.json
```

### Overriding Config Options

```bash
# Override trials
python3 run_experiment.py \
    --config ../config/brew/run.json \
    --trials 3

# Override output
python3 run_experiment.py \
    --config ../config/brew/run.json \
    --output ./my_results
```

### Multiple Configs

```bash
# Run test first
python3 run_experiment.py --config ../config/test/run.json

# Then full benchmark
python3 run_experiment.py --config ../config/brew/run.json
```

---

## Environment Variables

Configs can reference environment variables:

```json
{
  "paths": {
    "graphs_dir": "${GRAPHS_DIR}",
    "output_dir": "${OUTPUT_DIR:-./results}"
  }
}
```

Usage:
```bash
export GRAPHS_DIR=/data/large_graphs
python3 run_experiment.py --config run.json
```

---

## Creating Custom Configs

### Step 1: Copy Template

```bash
cp config/brew/run.json config/my_experiment.json
```

### Step 2: Edit Settings

```bash
# Edit with your favorite editor
vim config/my_experiment.json
```

### Step 3: Validate JSON

```bash
python3 -c "import json; json.load(open('config/my_experiment.json'))"
```

### Step 4: Run

```bash
python3 run_experiment.py --config config/my_experiment.json
```

---

## Output Files

### Results CSV

```csv
graph,benchmark,algorithm,trial,time,metric,reorder_time
facebook.el,pr,0,1,0.0234,,0
facebook.el,pr,0,2,0.0231,,0
facebook.el,pr,7,1,0.0189,,0.045
facebook.el,bfs,0,1,0.0012,76.9,0
```

### Summary JSON

```json
{
  "experiment": "Full Benchmark",
  "date": "2024-01-15",
  "config": "run.json",
  "results": {
    "facebook.el": {
      "pr": {
        "0": {"mean": 0.0232, "std": 0.0002},
        "7": {"mean": 0.0189, "std": 0.0003}
      }
    }
  }
}
```

---

## Troubleshooting

### "Config file not found"

```bash
# Check path
ls -la config/brew/run.json

# Use absolute path
python3 run_experiment.py --config /full/path/to/run.json
```

### "Invalid JSON"

```bash
# Validate and find error
python3 -m json.tool config/my_config.json
```

### "No graphs found"

Check `graphs_dir` path and `filters.extensions`.

---

## Next Steps

- [[Running-Benchmarks]] - Command-line usage
- [[Python-Scripts]] - Script documentation
- [[Command-Line-Reference]] - All options

---

[← Back to Home](Home) | [Python Scripts →](Python-Scripts)
