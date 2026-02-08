# Leiden Optimization Agent Pack

Explore, benchmark, and iteratively improve Leiden-based graph reordering
variants to beat RabbitOrder on cache quality and end-to-end performance.

## Quick Navigation

| File | Purpose |
|------|---------|
| [01_MISSION.md](01_MISSION.md) | Goal: beat RabbitOrder, constraints, success criteria |
| [02_REPO_MAP.md](02_REPO_MAP.md) | Key files for Leiden, VIBE, cache sim, and benchmark infra |
| [03_WORKFLOW.md](03_WORKFLOW.md) | Iterate: hypothesis → implement → measure → commit or revert |
| [04_CHECKLISTS.md](04_CHECKLISTS.md) | Correctness, performance, and quality gates |
| [05_EXPERIMENTS.md](05_EXPERIMENTS.md) | Benchmark tiers, graph categories, comparison protocol |
| [06_VARIANT_LANDSCAPE.md](06_VARIANT_LANDSCAPE.md) | Complete map of every Leiden/VIBE variant and its knobs |
| [07_TUNING_PLAYBOOK.md](07_TUNING_PLAYBOOK.md) | Parameter tuning guide: resolution, refinement, ordering strategies |
| [08_OUTPUT_FORMATS.md](08_OUTPUT_FORMATS.md) | Required format for benchmark results and reports |

## Getting Started

1. Read **01_MISSION.md** — understand what "beat RabbitOrder" means.
2. Read **06_VARIANT_LANDSCAPE.md** — know the full variant space before changing anything.
3. Follow **03_WORKFLOW.md** — every change goes through: measure baseline → change one thing → measure → keep or revert.
4. Use **`scripts/graphbrew_experiment.py`** for ALL evaluation.
