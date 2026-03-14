# scripts/

All Python tooling for GraphBrew lives here.

## Quick Start

```bash
# One-command: download 150 graphs/size, benchmark, train, evaluate
python3 scripts/graphbrew_experiment.py --target-graphs 150

# Preview what would run (no execution)
python3 scripts/graphbrew_experiment.py --target-graphs 150 --dry-run

# Skip ML evaluation at end
python3 scripts/graphbrew_experiment.py --target-graphs 150 --skip-eval
```

`--target-graphs N` auto-enables `--full`, `--catalog-size N`, `--auto`, `--all-variants`.

## Structure

```
scripts/
├── graphbrew_experiment.py        ← SINGLE entry point (the only top-level .py)
├── requirements.txt
├── experiments/                   ← Paper experiment runners
│   ├── vldb_config.py             ← VLDB paper: graph/algorithm configuration
│   ├── vldb_paper_experiments.py  ← VLDB paper: 8-experiment suite
│   ├── vldb_generate_figures.py   ← VLDB paper: figure/table generation
│   ├── vldb_experiments.py        ← VLDB: full lab suite
│   ├── vldb_experiments_small.py  ← VLDB: lightweight preview
│   ├── ecg_config.py              ← ECG paper: cache policy configuration
│   └── ecg_paper_experiments.py   ← ECG paper: 6-experiment suite
├── lib/                           ← modular library, 5 sub-packages (see lib/README.md)
│   ├── core/                      ← constants, logging, data stores
│   ├── pipeline/                  ← experiment execution stages
│   ├── ml/                        ← ML scoring & training (fallback)
│   ├── analysis/                  ← post-run analysis & visualisation
│   ├── tools/                     ← standalone CLI utilities
│   └── __init__.py                ← re-exports every public name
└── test/                          ← pytest tests  (run: pytest scripts/test/)
```

Weight and model data live under `results/data/` (not `scripts/`), managed by
`lib/core/utils.py` constants:

```
results/data/                           ← runtime data (gitignored)
├── benchmarks.json                     ← streaming benchmark database (primary)
├── graph_properties.json               ← graph feature vectors
└── adaptive_models.json                ← perceptron weights + DT/hybrid models (fallback)
```

## Rules for AI agents

1. **One top-level Python file: `graphbrew_experiment.py`.**
   It is the single pipeline entry point.
   Do NOT create other `.py` files at `scripts/` root.
   Analysis tools (e.g., `evaluate_all_modes.py`) live in `lib/tools/`.
   Everything else belongs in `lib/` or `experiments/`.

2. **Library code goes in `lib/`.**  Modules there are imported by
   `graphbrew_experiment.py`; they are never executed directly.

3. **Experiment runners go in `experiments/`.**
   Two paper suites: VLDB 2026 (graph reordering) and ECG/GrAPL (cache policies).
   Both output JSON results to `results/`.

4. **Weight/model paths** use helpers in `lib/core/utils.py`:
   `results_data_dir()`, `benchmarks_db_path()`, `graph_properties_path()`,
   `adaptive_models_path()`.
   Never hard-code paths like `benchmarks.json` or `adaptive_models.json`.

5. **Tests** go in `test/`.  Run with `pytest scripts/test/ -x -q`.

6. **Algorithm naming** uses SSOT functions in `lib/core/utils.py`:
   `canonical_algo_key()`, `algo_converter_opt()`,
   `canonical_name_from_converter_opt()`, `chain_canonical_name()`,
   `get_algo_variants()`.
   Never hard-code algorithm names — always derive them from these functions.
   `CHAINED_ORDERINGS` is auto-populated at import time.
