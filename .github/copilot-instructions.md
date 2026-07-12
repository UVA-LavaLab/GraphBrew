# ECG Artifact Agent Instructions

## Autonomous Map-Following Mode

When the user asks to follow a plan, checklist, or the paper runbook under
`research/ecg-hpca/`, work autonomously through the next unblocked steps.

Treat the referenced plan as an executable backlog:

1. Read the relevant plan/runbook and nearby docs.
2. Convert the next useful slice into a short todo list.
3. Execute the next unblocked item.
4. Validate with focused tests, compile checks, dry-runs, or status checks.
5. Update the paper SSOT/runbook when behavior changes.
6. Continue to the next unblocked item until the current slice is complete, a real blocker appears, or a long-running external job must be left running.

Only stop or ask for input when one of these is true:

- The next action is destructive or would delete/overwrite user data.
- A command asks for a password, token, passphrase, or other secret.
- Required information is genuinely missing and cannot be inferred from the repo.
- The plan offers mutually exclusive technical choices with real consequences.
- A long-running simulation has been launched and the user needs the terminal/run ID to monitor it.

For ECG/cache_sim/gem5/Sniper work:

- Do not run multiple gem5 ECG jobs on the same node unless sideband files are isolated.
- Keep generated `results/` artifacts out of commits.
- Prefer `scripts/experiments/ecg/flows/paper_run.py`, `roi_matrix.py`,
  `flows/proof_matrix.py`, and `flows/paper_pipeline.py` over ad hoc commands.
- For long runs, launch the run, record the run directory and terminal ID, check early status/logs, then continue with non-conflicting work.
- Before commits, run focused validation: `git diff --check`, relevant `py_compile`, `bash -n` for shell templates, and focused pytest where applicable.
- If a plan item requires large graph data that is missing, document the exact expected path and proceed with available-local or synthetic validation.

Communication style:

- Provide short progress updates after batches of work.
- Report deltas and next actions instead of restating the full plan.
- Final answers should say what changed, where it is, what passed, and what remains blocked or running.
