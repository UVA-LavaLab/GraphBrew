# Specialized experiment runners

Reusable logic belongs in `scripts/lib/`. This directory contains isolated,
restartable study runners that consume those shared contracts.

Use the top-level orchestrator for normal development:

```bash
source .venv/bin/activate
python3 scripts/graphbrew_experiment.py \
  --full --quick --size small --trials 1 --skip-cache
```

Campaign-specific entry points are implementation details until their
documentation is deliberately released. Do not duplicate shared download,
build, mapping, verification, or result-store logic inside a campaign.

See [scripts/README.md](../README.md) for the canonical paths table.
