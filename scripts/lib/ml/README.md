# scripts/lib/ml/ — Machine Learning & Scoring

Perceptron-based adaptive algorithm selection and model evaluation.

| File | Purpose |
|------|---------|
| `weights.py` | Compute perceptron weights from benchmark data (win-rate scoring) |
| `eval_weights.py` | Evaluate weight quality: accuracy, regret, per-graph breakdown |
| `training.py` | Iterative and batched weight training loops |
| `model_tree.py` | Decision tree and hybrid (DT + perceptron) model training |
| `adaptive_emulator.py` | Emulate C++ AdaptiveOrder scoring in Python for validation |
| `oracle.py` | Oracle analysis: optimal selection, confusion matrices |
| `features.py` | Graph feature extraction (degree stats, working set ratio, modularity) |

The ML pipeline produces `adaptive_models.json` consumed by the C++ `AdaptiveOrder` (algorithm 14).
