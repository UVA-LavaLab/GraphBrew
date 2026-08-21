# Retired adaptive-model ablation

`exp3_model_ablation.py` is retained to reproduce a historical offline model
comparison. It is not used by the validated deterministic runtime selector.

```bash
python3 scripts/experiments/adaptive_ml/exp3_model_ablation.py
```

Shared offline-model implementations live under
[`scripts/lib/ml/`](../../lib/ml/). New deployment claims must use the frozen
`allkernel-lowreuse-rule` evidence instead of this ablation.
