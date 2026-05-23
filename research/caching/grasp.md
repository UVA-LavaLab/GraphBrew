# GRASP — Graph-aware Cache Replacement with Software Prefetching

## Citation

```bibtex
@inproceedings{grasp-hpca20,
  title     = {Domain-Specialized Cache Management for Graph Analytics},
  author    = {Faldu, Priyank and Diamond, Jeff and Grot, Boris},
  booktitle = {International Symposium on High-Performance Computer Architecture (HPCA)},
  year      = {2020},
  month     = feb
}
```

## Official Repository

- **GitHub**: [faldupriyank/grasp](https://github.com/faldupriyank/grasp) — Apache-2.0
- **Contains**: Trace-based cache simulator (LRU/Belady/PIN/GRASP) + Ligra benchmarks
- **Related**: [faldupriyank/dbg](https://github.com/faldupriyank/dbg) — DBG reordering (Apache-2.0)

## Why Faithful Implementation Matters

GRASP is a **baseline policy** in the ECG paper. If our implementation deviates from Faldu et al. HPCA 2020:

1. **ECG Section A1 validation breaks** — the three GRASP invariants (below) become meaningless
2. **ECG mode equivalence fails** — ECG(DBG_ONLY) must match pure GRASP within 3% (Section A3)
3. **Published comparisons are unsound** — any ECG improvement claim over GRASP requires GRASP at full strength
4. **DBG bucket boundaries** must exactly match the paper's exponential grouping — changing thresholds alters RRPV distribution

Critical implementation parameters that **must** match the paper:
- `hot_fraction = 0.1` (10% of LLC reserved for high-degree vertices, "f" parameter in paper)
- Cache sim topology uses `11 degree-based buckets` (`GraphTopology::NUM_BUCKETS = 11` in `graph_cache_context.h`) for RRPV classification (note: the DBG *reordering* in `reorder_hub.h` uses 8 buckets for vertex permutation — these are separate systems)
- RRPV mapping: `rrpv = max_rrpv × (1 - edge_fraction)` where edge_fraction = cumulative degree share of bucket and higher — bucket 0 (hubs) → low RRPV, bucket 10 (cold) → high RRPV
- 3-tier reuse classification: HIGH (hot region), MODERATE (warm), LOW (cold)

## Paper Performance (from official repo)

### Cache Simulator Results (web-Google, 16-way 1MB LLC, DBG-reordered)

| Benchmark | LRU Miss% | Belady Miss% | PIN Miss% | GRASP Miss% |
|-----------|-----------|-------------|-----------|-------------|
| BC        | 60.2      | 62.4        | 36.5      | **33.3**    |
| SSSP      | 92.1      | 91.3        | 85.7      | **61.3**    |
| PR        | 87.9      | 75.7        | 64.7      | **56.5**    |
| PRD       | 87.9      | 75.7        | 64.7      | **56.5**    |
| Radii     | 84.2      | 72.4        | 62.7      | **51.0**    |

### Runtime Improvements (Ligra benchmarks, Broadwell Xeon E5-2630, 40 threads)

| Graph             | PR   | PRD  | BellmanFord |
|-------------------|------|------|-------------|
| twitter_rv        | 0.95 | 0.69 | 0.69        |
| twitter_mpi       | 0.99 | 0.65 | 0.67        |
| soc-LiveJournal1  | 0.97 | 0.81 | 0.76        |
| friendster        | 1.02 | 0.88 | 0.66        |
| web-Google        | 0.92 | 0.78 | 0.79        |

*(Values are normalized runtime — lower is better; e.g., 0.69 = 31% speedup)*

### Paper's Benchmarks & Graphs
- **Benchmarks**: PageRank (PR), PageRankDelta (PRD), BellmanFord (SSSP), BC, Radii — all Ligra-based
- **Graphs**: twitter_rv, twitter_mpi, soc-LiveJournal1, friendster, web-Google, dbpedia-link, pld-arc, sd-arc
- **Simulator**: Sniper full-system simulator (paper) + trace-based cache sim (repo)
- **Hardware**: Dual-socket Broadwell Intel Xeon E5-2630 (40 threads)

## Key Contributions

1. **Graph-aware insertion policy**: Uses vertex degree (DBG) to set RRIP insertion priorities — high-degree hubs get lower RRPV so they stay in cache longer
2. **Three-tier SRRIP extension**: Extends SRRIP with graph-structural metadata for 3-tier insertion
3. **Software prefetching integration**: Prefetch hints based on graph traversal patterns
4. **Cache-graph co-design**: First work to incorporate graph topology into cache replacement decisions

## Algorithm Pseudo-Code

### GRASP Insertion (from GraphBrew cache_sim.h)

```
function GRASP_INSERT(cache_set, address, line_size):
    victim_idx = findVictimGRASP(cache_set)
    cache_set[victim_idx].tag = address
    cache_set[victim_idx].valid = true

    // Degree-proportional RRPV assignment
    bucket = classifyBucket(address)        // address → degree bucket (0=hub, 10=cold)
    if bucket is valid:
        rrpv = bucketToRRPV(bucket, max=7)  // bucket 0 → RRPV 0, bucket 10 → RRPV 7
        cache_set[victim_idx].rrpv = rrpv
    else:
        cache_set[victim_idx].rrpv = 7      // Unknown → cold (distant re-reference)
```

### GRASP Hit Promotion

```
function GRASP_HIT(cache_set, idx, address):
    bucket = classifyBucket(address)
    if bucket == 0:                         // Highest-degree (hub)
        cache_set[idx].rrpv = 0             // Aggressive: reset to near-immediate
    else if bucket is valid:
        if cache_set[idx].rrpv > 0:
            cache_set[idx].rrpv -= 1        // Gradual: decrement by 1
```

### GRASP Eviction (same as SRRIP aging, but with degree-aware RRPV values)

```
function findVictimGRASP(cache_set):
    M_RRIP = 7    // 3-bit RRPV, max = 2^3 - 1
    while true:
        for i in 0..associativity-1:
            if cache_set[i].rrpv >= M_RRIP:
                return i                    // Evict line with max RRPV
        // No line at max — age all
        for i in 0..associativity-1:
            if cache_set[i].rrpv < M_RRIP:
                cache_set[i].rrpv += 1
```

### bucketToRRPV — Degree-Proportional Mapping

```
function bucketToRRPV(bucket, max_rrpv=7):
    // bucket 0 = highest degree (front of array after DBG reorder)
    // bucket N-1 = lowest degree (end of array)
    dbg_bucket = NUM_BUCKETS - 1 - bucket   // Reverse: front → high degree
    edge_sum = sum of bucket_total_degrees[dbg_bucket..NUM_BUCKETS-1]
    edge_fraction = edge_sum / total_edges

    rrpv = max_rrpv * (1.0 - edge_fraction)
    // High edge_fraction → low RRPV (important, keep in cache)
    // Low edge_fraction → high RRPV (unimportant, evict sooner)
    clamp rrpv to [1, max_rrpv]             // Reserve 0 for hit promotion
    return rrpv
```

## GraphBrew Validation Invariants (ECG Section A1)

| # | Invariant | Tolerance | Formula |
|---|-----------|-----------|---------|
| 1 | With DBG reordering, GRASP always beats SRRIP | strict ≤ | `GRASP_L3_miss ≤ SRRIP_L3_miss` (both with `-o 5` DBG) |
| 2 | Without DBG (original ordering), GRASP ≈ SRRIP | < 5% | `|GRASP - SRRIP| / SRRIP < 0.05` |
| 3 | Hot hub vertices (bucket-0) get RRPV much lower than cold | structural | Bucket 0 RRPV → 0; Bucket 10 RRPV → max |

**Rationale**: Claim 1 proves GRASP's structural advantage. Claim 2 proves the advantage comes *from DBG*, not from GRASP's eviction logic alone. Claim 3 verifies RRPV is degree-proportional.

## GraphBrew Integration

- **Cache Policy**: `EvictionPolicy::GRASP` in `bench/include/cache_sim/cache_sim.h`
- **Context**: `bench/include/cache_sim/graph_cache_context.h` — DBG metadata, bucket classification, reuse tiers
- **Configuration**: `hot_fraction=0.1`, `num_buckets=11`, `rrpv_max=7`
- **Depends on**: DBG vertex grouping (Algorithm 5, `-o 5`)
- **Tested in**: ECG Section A1 (faithfulness), B1 (all-policy comparison)
- **Experiment config**: `scripts/experiments/ecg_config.py`
