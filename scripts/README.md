# scripts/

All Python tooling for GraphBrew lives here.

## Structure

```
scripts/
├── graphbrew_experiment.py   ← SINGLE entry point (the only top-level .py)
├── requirements.txt
├── lib/                      ← modular library, 5 sub-packages (see lib/README.md)
│   ├── core/                 ← constants, logging, data stores
│   ├── pipeline/             ← experiment execution stages
│   ├── ml/                   ← ML scoring & training (fallback)
│   ├── analysis/             ← post-run analysis & visualisation
│   ├── tools/                ← standalone CLI utilities
│   └── __init__.py           ← re-exports every public name
└── test/                     ← pytest tests  (run: pytest scripts/test/)
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

1. **Only ONE top-level Python file exists: `graphbrew_experiment.py`.**
   Do NOT create other `.py` files at `scripts/` root.
   Everything else belongs in `lib/`.

2. **Library code goes in `lib/`.**  Modules there are imported by
   `graphbrew_experiment.py`; they are never executed directly.

3. **Weight/model paths** use helpers in `lib/core/utils.py`:
   `results_data_dir()`, `benchmarks_db_path()`, `graph_properties_path()`,
   `adaptive_models_path()`.
   Never hard-code paths like `benchmarks.json` or `adaptive_models.json`.

4. **Tests** go in `test/`.  Run with `pytest scripts/test/ -x -q`.

5. **Algorithm naming** uses SSOT functions in `lib/core/utils.py`:
   `canonical_algo_key()`, `algo_converter_opt()`,
   `canonical_name_from_converter_opt()`, `chain_canonical_name()`,
   `get_algo_variants()`.
   Never hard-code algorithm names — always derive them from these functions.
   `CHAINED_ORDERINGS` is auto-populated at import time.
