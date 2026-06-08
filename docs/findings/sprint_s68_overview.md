# Sprint S68 — Automated rubber-duck-gated gem5 fix pipeline

**Goal:** Close the gem5 ECG_PFX `pf_issued = 0` gap discovered in the
gem5 implementation audit (`docs/findings/gem5_implementation_audit_v1.md`)
and validate paper-faithful cycle-accurate prefetch behavior.

**Architecture:** A sequence of idempotent milestone scripts, each with
machine-checkable exit criteria. Between milestones, a rubber-duck
sub-agent reviews the evidence and issues a `GREEN | YELLOW | RED`
verdict. The driver halts the pipeline on `RED`, advances on
`GREEN | YELLOW`. Re-runs of any milestone are safe — patches detect
their marker, builds skip if binary mtime exceeds source mtime.

## Pipeline

```
M1 → M1b → M2 → M3 → M4 → M5 → M5b → M6 → M7
```

| ID | Milestone | Exit criteria | Wall |
|----|-----------|---------------|------|
| M1  | MMU plumbing patch (`graph_se.py`) | `registerMMU` calls = 8 (4 sites × 2 branches) | <1 sec |
| M1b | Queue-servicing fix (`queued.hh`) + X86 rebuild | marker in queued.hh + binary mtime > src mtime | ~2 min |
| M2  | Post-fix smoke (email-Eu-core/pr) | ECG_PFX `pf_identified > 0` AND `pf_issued > 0`; DROPLET control regression-free | ~3 min |
| M3  | Multi-graph smoke (email-Eu-core + delaunay_n19) | both cells `status=ok` AND `pf_useful > 0` on at least delaunay_n19 | ~30-60 min |
| M4  | Rebuild RISCV gem5.opt | RISCV `gem5.opt` mtime > latest overlay commit; symbol `GEM5_ECG_PFX_RECENT_FILTER_SIZE` present | ~30-60 min |
| M5  | `ecg.extract` opcode smoke on RISCV | `pf_issued > 0` via ISA path | ~10-15 min |
| M5b | Latency-readiness guard (`queued.cc`) + X86 rebuild | marker in queued.cc + binary mtime > src mtime | ~2 min |
| M6  | gem5 vs cache_sim DRAM-traffic parity | relative delta ≤ 25% on email-Eu-core/pr | ~5 min |
| M7  | Verdict doc + supersession marker | audit doc updated + gap doc marked SUPERSEDED | <1 sec |

## Why rubber-duck-gated?

This is a paper-critical implementation fix. We've already had ONE
incorrect root-cause diagnosis (the original
`docs/findings/gem5_ecg_pfx_simobject_gap.md` blamed the single-slot
mailbox; the actual cause turned out to be the page-cross filter +
queue-servicing). Each milestone is a hypothesis with falsifiable
evidence. The rubber-duck catches:

- Misinterpretation of evidence (e.g., "pf_identified > 0" without
  checking `pf_issued > 0` could falsely declare M1 a full success)
- Missed alternative root causes
- Subtle cycle-accurate semantic mistakes that would distort M6
- Regressions in non-target prefetchers (DROPLET control)

## Reproduce

```bash
# Run the next ready milestone (idempotent)
bash scripts/experiments/ecg/sprint_s68/run_sprint.sh --next

# Or run a specific milestone by ID
bash scripts/experiments/ecg/sprint_s68/run_sprint.sh M1
bash scripts/experiments/ecg/sprint_s68/run_sprint.sh M1b
bash scripts/experiments/ecg/sprint_s68/run_sprint.sh M3   # long-running

# See sprint state at a glance
bash scripts/experiments/ecg/sprint_s68/run_sprint.sh --status
```

After each milestone reports `status=ok`, a rubber-duck request file is
written to `results/sprint_s68/<m_id>/rubber_duck_request.md`. The
verdict is recorded at `results/sprint_s68/<m_id>/rubber_duck_verdict.md`
and the driver gates advancement on:

- `GREEN` first line → advance
- `YELLOW` first line → advance with caveats noted
- `RED` first line → halt sprint, surface to user
- missing or unrecognized → halt, await verdict

## Idempotency contract

Every milestone script is safe to re-run. They:

- Detect their own work via a marker (`S68-MMU-PATCH`,
  `S68-QUEUE-SERVICING-PATCH`, `S68-LATENCY-GUARD-PATCH`) before
  touching anything
- Compare binary vs source mtime before triggering a rebuild
- Use `mkdir -p` and `rm -rf` only on their own milestone output dir

## Durable patches

`bench/include/gem5_sim/gem5/` is gitignored (treated as vendor checkout
managed by `scripts/setup_gem5.py`). In-place edits there would be lost
on the next `--rebuild` or `--clean`. To survive:

1. M1's `graph_se.py` patch lives in
   `bench/include/gem5_sim/configs/graphbrew/`, which is NOT under the
   gitignored gem5/ tree — already durable.
2. M1b and M5b's queued.{hh,cc} patches are exported to
   `bench/include/gem5_sim/overlays/mem/cache/prefetch/queued_hh.patch`
   and `queued_cc_latency.patch`. Registered in `UNIFIED_DIFF_PATCHES`
   in `scripts/setup_gem5.py`. The new `apply_unified_diff_patches()`
   function applies them via `patch -p1` with `--dry-run` idempotency
   check.

## Current state

See `bash scripts/experiments/ecg/sprint_s68/run_sprint.sh --status`
for live status. Sprint todos are also tracked in the session SQL
store with IDs `s68-m1-mmu-patch` through `s68-m7-verdict-doc` and
dependency edges in `todo_deps`.

## Evidence archive

| Path | Content |
|------|---------|
| `results/sprint_s68/<m_id>/status.json` | Machine-readable exit evidence (counts, mtimes, etc.) |
| `results/sprint_s68/<m_id>/log.txt` | Human-readable execution trace |
| `results/sprint_s68/<m_id>/rubber_duck_request.md` | Auto-generated review packet for the duck |
| `results/sprint_s68/<m_id>/rubber_duck_verdict.md` | Duck's GREEN/YELLOW/RED + findings |
| `results/sprint_s68/<m_id>/<arm>/roi_matrix.csv` | (for smoke milestones) prefetch + cache stats |
| `results/sprint_s68/<m_id>/<arm>/gem5/.../stats.txt` | (for smoke milestones) full gem5 stats dump |

## Final deliverable

When M7 finishes GREEN:

- `docs/findings/gem5_implementation_audit_v1.md` updated with
  "Post-fix verdict (sprint S68)" section listing per-milestone
  evidence + paper-ready blurb
- `docs/findings/gem5_ecg_pfx_simobject_gap.md` marked SUPERSEDED
  with cross-reference to S68 evidence
- gem5 cycle-accurate ECG_PFX validation working on email-Eu-core +
  delaunay_n19 + (if budget) larger graphs
- All fixes durable via the overlay-patch mechanism

This is the closeout for the gem5 chapter of the paper. After M7, the
remaining work is paper-writeup (Tables 7/8, L3-scaling figure, §6.3
discussion update).
