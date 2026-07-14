# scripts/lib/tools/ — Standalone CLI Utilities

Analysis and diagnostic tools invocable directly or via `graphbrew_experiment.py` flags.

| File | Flag | Purpose |
|------|------|---------|
| `evaluate_all_modes.py` | `--evaluate` | Model × Criterion evaluation with LOGO cross-validation |
| `check_includes.py` | `--check-includes` | Scan C++ sources for legacy include paths |
| `regen_features.py` | `--regen-features` | Regenerate features.json for all .sg graphs via C++ binary |
