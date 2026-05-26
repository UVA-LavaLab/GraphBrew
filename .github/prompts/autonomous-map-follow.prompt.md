---
description: "Use when: you want Copilot to follow a plan, map, checklist, or runbook autonomously without asking to continue after each step."
---

# Autonomous Map Follow

Follow the referenced plan/map/checklist autonomously.

Plan or map to follow:

`${input:plan:Path or description of the plan to follow}`

Operating rules:

- Read the plan and any local docs needed to understand it.
- Convert the next useful slice into a concise todo list.
- Execute the next unblocked tasks without asking for permission to continue.
- Ask only for secrets, destructive actions, missing information that cannot be inferred, or major mutually exclusive choices.
- Validate each completed slice with the repo's focused tests or dry-runs.
- Keep generated results out of commits unless explicitly requested.
- If a long-running simulation is launched, record the terminal ID, run directory, current job, and safe monitoring command.
- Continue until the slice is complete or genuinely blocked.

For GraphBrew ECG/gem5/Sniper work, respect simulator guardrails:

- Do not run multiple GraphBrew gem5 jobs on the same node unless sideband files are isolated.
- Prefer existing runners over ad hoc commands.
- Use `final_paper_run.py` for manifest jobs, `paper_pipeline.py --skip-run` for aggregation, and Slurm shard filters for distributed runs.

Start now by reading the plan and choosing the next unblocked task.
