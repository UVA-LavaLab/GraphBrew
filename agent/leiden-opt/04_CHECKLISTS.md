# Checklists — Leiden Optimization

---

## Correctness Checklist (before every commit)

- [ ] `make` compiles without warnings (`-Wall -Wextra`)
- [ ] Test tier passes (3 graphs × 2 benchmarks × 2 trials)
- [ ] BFS/SSSP produce correct distances (verified by reference run)
- [ ] PR converges to same values as ORIGINAL ordering (within 1e-6)
- [ ] CC produces same component count as ORIGINAL
- [ ] No memory errors: `valgrind --tool=memcheck ./bench/bin/pr -f <graph> -o 17`
- [ ] Reorder mapping is a valid permutation (no duplicate or missing vertex IDs)

## Performance Checklist (before claiming improvement)

- [ ] Baseline measured on same machine, same load, same day
- [ ] At least 5 trials per graph per benchmark
- [ ] Geo-mean computed across ≥10 graphs (not cherry-picked)
- [ ] Reorder time included in end-to-end comparison
- [ ] RabbitOrder included as comparison point (not just ORIGINAL)
- [ ] No outlier-driven claims (remove top/bottom 5% if >20 data points)

## Quality Gates

| Gate | Threshold | Fail Action |
|------|-----------|-------------|
| Correctness | 100% identical outputs | Stop. Fix before proceeding. |
| Reorder time | ≤3× RabbitOrder | Optimise or simplify the variant |
| Kernel speedup | ≥1.0× vs RabbitOrder geo-mean | Revert or refine hypothesis |
| Cache miss rate | ≤ RabbitOrder L3 misses | Investigate ordering strategy |
| No regressions | ≤5% slowdown on any single graph | Investigate or conditionalise |

## Pre-Commit Checklist

- [ ] Change is a single, focused improvement (not a kitchen-sink commit)
- [ ] Commit message includes measured improvement: `leiden: <change> — <X>% geo-mean speedup over rabbit on MEDIUM`
- [ ] If new variant added: registered in `scripts/lib/reorder.py`
- [ ] If new parameter added: documented in `VibeConfig` struct with default
