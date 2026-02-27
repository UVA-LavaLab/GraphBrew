# scripts/test/ — Test Suite

Pytest-based tests for the GraphBrew Python tooling and C++ integration.

| Test File | What It Tests |
|-----------|---------------|
| `test_algorithm_variants.py` | All algorithm variants execute correctly (70+ parametrized cases) |
| `test_cache_simulation.py` | Cache simulation correctness |
| `test_experiment_validation.py` | End-to-end pipeline: convert → benchmark → datastore → weights → oracle |
| `test_fill_adaptive.py` | AdaptiveOrder fill logic |
| `test_fill_weights_variants.py` | Weight computation from benchmark results for all 21 trainable variants |
| `test_graphbrew_experiment.py` | Experiment pipeline phases |
| `test_multilayer_validity.py` | GraphBrewOrder multi-layer config validation |
| `test_self_recording.py` | C++ self-recording (--db-dir / GRAPHBREW_DB_DIR) integration |
| `test_weight_flow.py` | Weight training → export flow |

## Running

```bash
cd /path/to/GraphBrew
python3 -m pytest scripts/test/ -v            # All tests
python3 -m pytest scripts/test/ -k gorder -v  # Just GOrder tests
```

Test fixtures (tiny graphs) live in `graphs/tiny/`.
