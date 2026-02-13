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

## 🚀 Quick Start

```bash
python3 scripts/graphbrew_experiment.py --full --size small          # Full pipeline
python3 scripts/graphbrew_experiment.py --train --size small         # Training pipeline
python3 scripts/graphbrew_experiment.py --size small --quick         # Quick test
python3 scripts/graphbrew_experiment.py --brute-force               # Validation
```

Sizes: `small` (16 graphs, 62MB) · `medium` (28, 1.1GB) · `large` (37, 25GB) · `xlarge` (6, 63GB) · `all` (87, 89GB). Categories include mesh, web, social, road, citation, P2P, and synthetic graphs.

Results saved to `./results/` (`reorder_*.json`, `benchmark_*.json`, `cache_*.json`) and weights to `./results/weights/` (`registry.json`, `type_N/weights.json`).

---

## Running Individual Phases

```bash
python3 scripts/graphbrew_experiment.py --phase reorder --size small
python3 scripts/graphbrew_experiment.py --phase benchmark --size small --skip-cache
python3 scripts/graphbrew_experiment.py --phase cache --size small
python3 scripts/graphbrew_experiment.py --phase weights
```

See [[Command-Line-Reference]] for all options including `--min-mb`, `--max-graphs`, `--trials`, `--quick`.

---

## Output Format

Results are JSON arrays. See [[Configuration-Files]] for the complete schema of `benchmark_*.json`, `cache_*.json`, `reorder_*.json`, and `type_N.json` weight files.

### Amortization Analysis

After benchmarking, derive end-to-end metrics from existing result files:

```bash
python3 scripts/graphbrew_experiment.py --phase all  # Amortization computed automatically
python3 -m scripts.lib.metrics  # Standalone amortization analysis
```

See [[Python-Scripts#-amortization--end-to-end-evaluation---phase-all]] for full details on amortization iterations, E2E speedup, and head-to-head comparisons.

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
python3 scripts/graphbrew_experiment.py --phase benchmark --size small
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
<!-- LeidenCSR (16) deprecated — GraphBrew (12) subsumes it -->
└────────────────────┴────────────┴──────────────┘
```

---

## Experiment Workflow

```bash
# One-click full experiment
python3 scripts/graphbrew_experiment.py --full --size medium
```

For step-by-step control, see [[Running-Benchmarks]] for manual execution and [[Command-Line-Reference]] for all options.

---

## Troubleshooting

See [[Troubleshooting]] for common issues. Quick fixes:
- Missing graphs: `--download-only --force-download`
- Memory issues: `--size small` or `--max-mb 500`
- Timeouts: `--skip-slow --skip-expensive`

---

## Next Steps

- [[Correlation-Analysis]] - Analyze benchmark results
- [[AdaptiveOrder-ML]] - Train the perceptron
- [[Running-Benchmarks]] - Manual benchmark commands
- [[Python-Scripts]] - Full script documentation

---

[← Back to Home](Home) | [Correlation Analysis →](Correlation-Analysis)
