# `ecg/` — ECG / GrAPL paper

Graph-aware cache replacement policies. Self-contained: config + runner.

| File | Purpose |
|---|---|
| [`config.py`](config.py) | 9 cache policies, 11 reorder×policy pairs, 6 graphs |
| [`runner.py`](runner.py) | 6 experiments (policy comparison, reorder interaction, cache sweep, fat-ID) |

## Run

```bash
python3 scripts/experiments/ecg/runner.py --all --graph-dir results/graphs
python3 scripts/experiments/ecg/runner.py --exp 6                # analytical only
python3 scripts/experiments/ecg/runner.py --exp 1 --preview --dry-run
```

Outputs land in `results/ecg_experiments/`.
