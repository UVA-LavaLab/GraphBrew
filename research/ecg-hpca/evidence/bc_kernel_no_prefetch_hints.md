# BC kernel emits zero prefetch hints — finding + path forward

**Status:** Documented limitation. No fix scheduled.
**Date:** 2026-06-04
**Affects:** 4 of 16 literature cells (cit-Patents/bc, soc-LiveJournal1/bc,
soc-pokec/bc, web-Google/bc) and corresponding prefetcher claim averages.

## Observation

In the sprint 6f literature corpus sweep, four cells show:

```
prefetch_requests = 0
prefetch_fills = 0
prefetch_useful = 0
```

for both ECG_PFX and DROPLET arms. All four cells are BC
(Betweenness Centrality) workloads.

## Root cause

`bench/src_sim/bc.cc` does not call `SIM_CACHE_PREFETCH_VERTEX` at any
demand site. Compare:

```
$ grep -cE "SIM_CACHE_PREFETCH" bench/src_sim/{pr,bfs,sssp,bc}.cc
bench/src_sim/pr.cc: 2 hint sites
bench/src_sim/bfs.cc: 2 hint sites
bench/src_sim/sssp.cc: 2 hint sites
bench/src_sim/bc.cc: 0 hint sites
```

BC's source therefore generates no hints, no prefetch lookahead, no
mask-based prefetches. The cache_sim simulator runs BC as a baseline
demand-only workload regardless of `ECG_PREFETCH_MODE` or `--prefetcher`
flags.

## Impact on paper claims

The 4 BC cells correctly show zero prefetcher delta in our tables
(the prefetcher physically does nothing on them), but they dilute
"corpus mean" prefetcher claims when included.

Sprint 6f-2 added active-cell-only summary fields to
`paper_table_prefetcher.py` that exclude BC cells:
- `n_pfx_active_cells` (= 12 of 16 = 75%)
- `mean_pfx_marginal_demand_active_pp` (active-cell mean)
- `mean_droplet_marginal_demand_active_pp` (active-cell mean)

The corpus-mean (16-cell) and active-cell-mean (12-cell) are both
reported in the markdown; honest paper claims should use the
active-cell mean and footnote the BC carve-out.

## Path forward (not scheduled)

Adding `SIM_CACHE_PREFETCH_VERTEX` calls to bc.cc would activate
prefetcher behavior on the 4 BC cells:

```cpp
// In bc.cc's PullStep_Sim or similar inner loop:
for (auto it = neighbours.begin(); it != neighbours.end(); ++it) {
    SIM_CACHE_READ_EDGE(cache, it);
    NodeID v = *it;
    if (pfx_lookahead > 0 && graph_ctx.mask_config.prefetch_mode > 0) {
        // ... lookahead-window scan, similar to pr.cc/bfs.cc patterns
    }
    SIM_CACHE_READ_MASKED(cache, parent.data(), v, graph_ctx, vertex_masks[v]);
}
```

Estimated work: ~1-2 hours per kernel function (BC has two: PullStep
and frontier processing). Plus a re-sweep of the BC cells to validate.

Not scheduled because:
1. The active-cell summary already gives honest prefetcher claims
2. The BC carve-out is documented and reproducible (this finding)
3. Adding hints to BC won't change the prefetcher-saturation conclusion
   (see prefetcher_saturation_under_eviction.md): under ECG_DBG eviction,
   prefetchers converge with each other regardless of source kernel

## Citation

For the paper, the carve-out reads as:

> "We measure prefetcher claims on the 12 of 16 literature corpus cells
> where the workload kernel emits ECG prefetch hints. BC workloads
> (4 cells) do not yet emit hints in our reference kernel implementation;
> they are reported as eviction-only baselines."
