# Benchmark Suite

The GraphBrew Benchmark Suite provides automated tools for running comprehensive experiments across multiple graphs, algorithms, and benchmarks.

## Overview

```
scripts/
├── graphbrew_experiment.py     # ⭐ MAIN: One-click unified pipeline
├── requirements.txt            # Python dependencies
└── lib/                        # 📦 5 sub-packages (see lib/README.md)
    ├── core/                   # Constants, logging, data stores
    ├── pipeline/               # Experiment execution stages
    ├── ml/                     # ML scoring & training (fallback)
    ├── analysis/               # Post-run analysis & visualisation
    └── tools/                  # Standalone CLI utilities
```

Weight files are stored under `results/data/adaptive_models.json` (not `scripts/`).

---

## 🚀 Quick Start

```bash
python3 scripts/graphbrew_experiment.py --full --size small          # Full pipeline
python3 scripts/graphbrew_experiment.py --train --size small         # Training pipeline
python3 scripts/graphbrew_experiment.py --size small --quick         # Quick test
python3 scripts/graphbrew_experiment.py --brute-force               # Validation
```

Sizes: `small` (16 graphs, 62MB) · `medium` (28, 1.1GB) · `large` (37, 25GB) · `xlarge` (6, 63GB) · `all` (87, 89GB). Categories include mesh, web, social, road, citation, P2P, and synthetic graphs.

Results saved to `./results/` (`reorder_*.json`, `benchmark_*.json`, `cache_*.json`) and weights to `./results/data/adaptive_models.json`.

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

Results are JSON arrays. See [[Configuration-Files]] for the complete schema of `benchmark_*.json`, `cache_*.json`, and `reorder_*.json`. Weight data is consolidated in `results/data/adaptive_models.json`.

### Amortization Analysis

After benchmarking, the pipeline automatically computes amortization metrics:

- **Break-even N\*** = `reorder_overhead / time_saved_per_iteration` — iterations before reordering pays off
- **E2E Speedup@N** = `N × baseline_time / (reorder_overhead + N × reordered_time)` — end-to-end speedup
- **MinN@95%** — smallest N where reorder overhead < 5% of total cost

```bash
python3 scripts/graphbrew_experiment.py --phase all  # Amortization computed automatically
python3 -m scripts.lib.analysis.metrics  # Standalone amortization analysis
```

> **Note:** Experiments default to 7 benchmarks (`EXPERIMENT_BENCHMARKS` — TC excluded). After RANDOM baseline `.sg` conversion, the pipeline pre-generates reordered `.sg` for each of the 12 reorder algorithms (`--pregenerate-sg`, default ON). At benchmark time, pre-generated `.sg` files are loaded with `-o 0` — no runtime reorder overhead. The reorder phase runs 12 algorithms (baselines ORIGINAL/RANDOM skipped). Benchmarking runs all 14 eligible algorithms.

See [[Python-Scripts#-amortization--end-to-end-evaluation---phase-all]] for full details.

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

PageRank convergence can vary by reordering algorithm. Run with `--benchmarks pr` to see iteration counts and final error for each algorithm on your graphs.

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
