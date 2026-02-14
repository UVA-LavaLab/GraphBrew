# scripts/test/ — Test Suite

Pytest-based tests for the GraphBrew Python tooling and C++ integration.

| Test File | What It Tests |
|-----------|---------------|
| `test_algorithm_variants.py` | All algorithm variants execute correctly (70+ parametrized cases) |
| `test_graphbrew_experiment.py` | Experiment pipeline phases |
| `test_cache_simulation.py` | Cache simulation correctness |
| `test_fill_adaptive.py` | AdaptiveOrder fill logic |
| `test_weight_flow.py` | Weight training → export flow |
| `test_weight_merger.py` | Multi-run weight merging |
| `test_multilayer_validity.py` | GraphBrewOrder multi-layer config validation |

## Running

```bash
cd /path/to/GraphBrew
python3 -m pytest scripts/test/ -v            # All tests
python3 -m pytest scripts/test/ -k gorder -v  # Just GOrder tests
```

Test fixtures (tiny graphs) live in `graphs/tiny/`.
