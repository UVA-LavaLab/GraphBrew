# scripts/experiments/ — Paper Experiment Suites

Two self-contained experiment runners for two papers, plus shared configuration.

## VLDB 2026 — Multilayered Graph Reordering

| File | Purpose |
|------|---------|
| `vldb_config.py` | Shared configuration: 16 baselines, 10 GraphBrew variants, 11 graphs, 5 chains |
| `vldb_paper_experiments.py` | Main runner: 8 experiments with auto-build/download/convert |
| `vldb_generate_figures.py` | LaTeX table & PNG figure generation from JSON results |
| `vldb_experiments.py` | Full lab experiment suite |
| `vldb_experiments_small.py` | Lightweight preview version |

```bash
# Run all VLDB experiments
python3 scripts/experiments/vldb_paper_experiments.py --all --graph-dir /data/graphs

# Preview mode (smaller graphs)
python3 scripts/experiments/vldb_paper_experiments.py --all --preview
```

## ECG/GrAPL — Graph-Aware Cache Replacement Policies

| File | Purpose |
|------|---------|
| `ecg_config.py` | Configuration: 9 cache policies, 11 reorder×policy pairs, 6 graphs |
| `ecg_paper_experiments.py` | 6 experiments: policy comparison, reorder interaction, cache sweep, fat-ID analysis |

```bash
# Run all ECG experiments
python3 scripts/experiments/ecg_paper_experiments.py --all --graph-dir /data/graphs

# Analytical only (no simulation needed)
python3 scripts/experiments/ecg_paper_experiments.py --exp 6

# Preview mode
python3 scripts/experiments/ecg_paper_experiments.py --exp 1 --preview --dry-run
```

### ECG Experiments

| # | Experiment | Input | Output |
|---|-----------|-------|--------|
| 1 | Policy Comparison | 6 graphs × 7 benchmarks × 9 policies | Miss rate table |
| 2 | Reorder Interaction | 6 × 7 × 11 pairs | DBG requirement proof |
| 3 | Cache Size Sweep | 6 × 2 × 5 × 12 sizes | Miss rate curves |
| 4 | Algorithm Analysis | Derived from Exp1 | Iterative vs traversal |
| 5 | Graph Sensitivity | Derived from Exp1 | Topology-specific results |
| 6 | Fat-ID Analysis | Analytical | Bit allocation table |

## Other Files

| File | Purpose |
|------|---------|
| `exp3_model_ablation.py` | ML model ablation experiments |
| `__init__.py` | Package marker |

## Output

Results are saved as JSON in `results/` with timestamps:
- VLDB: `results/vldb_experiments/`
- ECG: `results/ecg_experiments/`
