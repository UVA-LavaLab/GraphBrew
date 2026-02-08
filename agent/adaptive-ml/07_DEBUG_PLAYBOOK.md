# Debug Playbook (multi-file)

When something looks wrong, follow this order:

---

## Step 1 — Reproduce

- Identify minimal graph input that triggers it.
- Capture:
  - Config flags
  - Seed
  - Selected type_id
  - Chosen algos per community

## Step 2 — Trace

- Build a trace of:
  ```
  input graph → communities → features → scores → chosen algo
  → permutation → stitched permutation → output graph
  ```
- For each stage, print sizes and hashes (e.g., checksum of perm vector).

## Step 3 — Validate Invariants

- Assert bijection and ranges.
- Check cross-community edges remain valid.
- Confirm that fallback triggers match expectation.

## Step 4 — Isolate

- Run toggles:
  - no-Leiden
  - no-types
  - force ORIGINAL
  - force a specific reorder
- See where bug originates.

## Step 5 — Fix

- Fix must include:
  - Root cause explanation
  - Unit test or regression script
  - Before/after trace snippet

---

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Using `--expand-variants` on CLI | Use `--all-variants` (internal dest is `expand_variants`) |
| Running `--phase benchmark` without prior reorder | Use `--full` which runs reorder first, or run `--phase reorder` first |
| `--size` defaults to nothing useful | Always specify `--size small\|medium\|large` or use `--graph-list` |
| Specifying `--csr-variants` auto-enables all variant expansion | By design — it sets `--all-variants`. Use `--algo-list` to further filter if needed |
| SSSP results are unstable with 2 trials | Use `--trials 5` minimum for SSSP/BFS |
| Missing graphs cause silent skips | Use `--full` to auto-download, or `--download-only --size X` to pre-download |
| Re-running re-does everything | It **doesn't** — existing `.lo` and `.time` files are reused. Use `--force-reorder` to redo |
| Raw binary calls for evaluation | Only for smoke tests. Use `graphbrew_experiment.py` for everything else |
