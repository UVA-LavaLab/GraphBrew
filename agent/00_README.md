# GraphBrew Agent Pack (Claude)

This folder contains instructions for an agent to explore the GraphBrew codebase,
specifically AdaptiveOrder-ML (Algorithm 14), and produce actionable results:
- correctness + safety validation
- performance evaluation plans
- ablation study plan
- SOTA improvement proposals
- debugging playbooks
- concrete code changes via PR-ready patches

## Quick Navigation

| File | Purpose |
|------|---------|
| [01_MISSION.md](01_MISSION.md) | Goals, constraints, and scope |
| [02_REPO_MAP.md](02_REPO_MAP.md) | Key files, call chains, and infrastructure concepts |
| [03_WORKFLOW.md](03_WORKFLOW.md) | Phase-by-phase workflow + mandatory development protocol |
| [04_CHECKLISTS.md](04_CHECKLISTS.md) | Correctness, performance, model quality, and iteration checklists |
| [05_ABLATIONS_AND_EXPERIMENTS.md](05_ABLATIONS_AND_EXPERIMENTS.md) | Ablation toggles, evaluation tiers, graph categories, and result interpretation |
| [06_SOTA_IDEAS.md](06_SOTA_IDEAS.md) | Tiered improvement proposals (minimal → heavy) |
| [07_DEBUG_PLAYBOOK.md](07_DEBUG_PLAYBOOK.md) | Multi-file debugging procedure + common pitfalls |
| [08_OUTPUT_FORMATS.md](08_OUTPUT_FORMATS.md) | Required deliverable format for all agent outputs |
| [09_PR_GUIDELINES.md](09_PR_GUIDELINES.md) | Pull request rules + how to add new algorithms/variants |

## Getting Started

1. Read **01_MISSION.md** to understand goals and constraints.
2. Follow **03_WORKFLOW.md** phase by phase.
3. All outputs must follow **08_OUTPUT_FORMATS.md**.
4. Use **`scripts/graphbrew_experiment.py`** for all evaluation — never bypass it with raw binary calls (except smoke tests).
