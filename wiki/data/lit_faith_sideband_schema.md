# Gate 248 — gem5/Sniper/cache_sim sideband-schema registry

- status: **active**
- schema fields: 6
- emit sites: 3
- Tier-A regex round-trip ok: True
- violations: 0

## Canonical schema

```
[graphctx] register region source=%s name=%s base=0x%lx upper=0x%lx hot_pct=%u grasp_region=%d\n
```

## Per-site

| site | exists | emit calls | fmt ok |
|---|---|---|---|
| `bench/include/gem5_sim/overlays/mem/cache/replacement_policies/graph_cache_context_gem5.hh` | True | 1 | True |
| `bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/graph_cache_context_sniper.cc` | True | 1 | True |
| `bench/include/cache_sim/graph_cache_context.h` | True | 1 | True |
