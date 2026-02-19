# scripts/

All Python tooling for GraphBrew lives here.

## Structure

```
scripts/
├── graphbrew_experiment.py   ← SINGLE entry point (the only top-level .py)
├── requirements.txt
├── lib/                      ← internal library modules (import, never run directly)
│   ├── utils.py              ← paths, constants, shared helpers
│   ├── weights.py            ← weight training, type registry, LOGO CV
│   ├── eval_weights.py       ← evaluation / accuracy reports
│   ├── adaptive_emulator.py  ← Python mirror of C++ adaptive selector
│   ├── perceptron.py         ← interactive perceptron experimentation
│   ├── weight_merger.py      ← merge weights across experiment runs
│   ├── features.py           ← graph feature computation
│   ├── benchmark.py          ← benchmark orchestration
│   ├── ...                   ← (see lib/ for full listing)
│   └── __init__.py
├── test/                     ← pytest tests  (run: pytest scripts/test/)
└── examples/                 ← standalone example scripts
```

Weight files live under `results/weights/` (not `scripts/`), managed by
`lib/utils.py` constants (`WEIGHTS_DIR`, `ACTIVE_WEIGHTS_DIR`):

```
results/weights/              ← runtime weight data (gitignored)
├── registry.json
├── type_0/
│   ├── weights.json
│   ├── pr.json
│   └── ...
├── type_1/
│   └── weights.json
└── type_2/
    └── weights.json
```

## Rules for AI agents

1. **Only ONE top-level Python file exists: `graphbrew_experiment.py`.**
   Do NOT create other `.py` files at `scripts/` root.
   Everything else belongs in `lib/`.

2. **Library code goes in `lib/`.**  Modules there are imported by
   `graphbrew_experiment.py`; they are never executed directly.

3. **Weight file paths** are built by helpers in `lib/utils.py`:
   `weights_registry_path()`, `weights_type_path()`, `weights_bench_path()`.
   Never hard-code paths like `type_0.json` or `type_registry.json`.

4. **Tests** go in `test/`.  Run with `pytest scripts/test/ -x -q`.
