# P-OPT — Practical Optimal Cache Replacement for Graph Analytics

## Citation

```bibtex
@inproceedings{popt-hpca21,
  title     = {P-OPT: Practical Optimal Cache Replacement for Graph Analytics},
  author    = {Balaji, Vignesh and Crago, Neal and Jaleel, Aamer and Lucia, Brandon},
  booktitle = {2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA)},
  pages     = {668--681},
  year      = {2021},
  organization = {IEEE}
}
```

## Official Repository

- **GitHub**: [CMUAbstract/POPT-CacheSim-HPCA21](https://github.com/CMUAbstract/POPT-CacheSim-HPCA21) — MIT
- **Contains**: Pin-based cache simulator with 4 LLC policies: LRU, DRRIP, P-OPT, T-OPT (ideal Belady)
- **Applications**: PageRank (SpM-DV), PR-Delta (push-pull SpM-SpV), Connected Components (push SpM-DV)
- **Authors**: Vignesh Balaji, Neal Crago, Aamer Jaleel, Brandon Lucia
- **Repo maintainer**: Vignesh Balaji ([@bvignesh](https://github.com/bvignesh))
- **Related**: [CMUAbstract/Graph-Reordering-IISWC18](https://github.com/CMUAbstract/Graph-Reordering-IISWC18) — packing factor + Hub Sorting/Clustering (MIT)
- **SRRIP/DRRIP reference**: [RRIP Paper (Jaleel et al., ISCA'10)](https://people.csail.mit.edu/emer/papers/2010.06.isca.rrip.pdf)

## Why Faithful Implementation Matters

P-OPT is the **oracle baseline** in the ECG paper. If our implementation deviates from Balaji et al. HPCA 2021:

1. **ECG Section A2 validation breaks** — P-OPT must beat LRU by >10%, must be reorder-agnostic within 10%
2. **ECG mode equivalence fails** — ECG(POPT_PRIMARY) must match pure P-OPT within 5% (Section A3)
3. **Oracle ceiling is wrong** — P-OPT is the upper bound for what graph-aware replacement can achieve; if it's weak, ECG looks artificially good
4. **Rereference matrix encoding** must exactly match the paper's Algorithm 2 (Section 4.1) — wrong encoding gives wrong predictions

Critical implementation parameters:
- Rereference matrix: **256 epochs**, 8-bit entries with MSB encoding (see pseudo-code below)
- Max rereference distance: **127** (7-bit field)
- Eviction priority: non-graph data first, then max rereference distance, then RRIP tiebreak
- Insertion RRPV: 6 (M_RRPV - 1, same as SRRIP default)

## P-OPT Repo Cache Hierarchy

From the official P-OPT repo's simulator:
- **3-level cache hierarchy**, 8 cores
- Each core: private L1 + L2 with **PLRU** (Pseudo-LRU)
- Shared **NUCA LLC** (8 banks) — LLC replacement policy varies (LRU/DRRIP/P-OPT/T-OPT)
- All levels: **64B cache line**, **non-inclusive**
- **No coherence modeled** (pull-style graph apps use read-only shared data)
- P-OPT reserves LLC capacity for Rereference Matrix Columns

## Paper Performance Claims

From the paper (Balaji et al. HPCA 2021):
- P-OPT achieves **near-Belady-optimal** miss rates (within 1-5% on small graphs)
- P-OPT beats LRU by **>10%** in LLC miss reduction across tested workloads
- P-OPT is **reorder-agnostic**: performance stable across different vertex orderings (within 10%)
- P-OPT beats all RRIP variants (SRRIP, DRRIP, BRRIP) in LLC miss rate

### Paper's Benchmarks & Graphs
- **Benchmarks**: PageRank (SpM-DV), PR-Delta (SpM-SpV), Connected Components (CC_SV)
- **Simulator**: Pin-2.14 based cache simulator
- **Comparison policies**: LRU, DRRIP, T-OPT (ideal Belady upper bound)

## Key Contributions

1. **Practical Belady approximation**: Uses pre-computed rereference distance from graph transpose to predict future accesses
2. **Transpose-based prediction**: Edge (u→v) in graph means v→u in transpose — predicts when vertex u will be re-accessed
3. **Rereference distance matrix**: Compressed 8-bit per cache-line-per-epoch encoding
4. **Reorder-agnostic**: Works with or without prior vertex reordering
5. **Oracle-quality decisions**: Achieves cache hit rates close to Belady's theoretical optimum

## Algorithm Pseudo-Code

### P-OPT Eviction — Algorithm 2 (Section 4.1) (from GraphBrew cache_sim.h)

```
function findVictimPOPT(cache_set):
    // Phase 1: Evict non-graph data first (streaming/CSR metadata)
    for i in 0..associativity-1:
        if NOT isPropertyData(cache_set[i].line_addr):
            return i                          // Not vertex data → evict immediately

    // Phase 2: All ways contain graph vertex data — find max rereference distance
    maxRerefDist = 0
    for i in 0..associativity-1:
        dist = findNextRef(cache_set[i].line_addr)
        wayRerefDists[i] = min(dist, 127)
        if wayRerefDists[i] > maxRerefDist:
            maxRerefDist = wayRerefDists[i]

    // Phase 3: RRIP tiebreaker among lines with max rereference distance
    M_RRPV = 7
    while true:
        for i in 0..associativity-1:
            if wayRerefDists[i] == maxRerefDist AND cache_set[i].rrpv >= M_RRPV:
                return i                      // Evict: farthest future + aged out
        // Age only the tied lines (not all lines)
        for i in 0..associativity-1:
            if wayRerefDists[i] == maxRerefDist AND cache_set[i].rrpv < M_RRPV:
                cache_set[i].rrpv += 1
```

### Rereference Matrix Encoding (Algorithm 2, Section 4.1)

```
// Matrix: num_epochs × num_cache_lines, each entry = 8 bits
// 256 epochs, each epoch = |V| / 256 vertices

Entry encoding:
  MSB=1 (bit 7 set):   cache line IS referenced in this epoch
                        bits [6:0] = sub-epoch position of LAST access
  MSB=0 (bit 7 clear): cache line is NOT referenced in this epoch
                        bits [6:0] = distance (in epochs) to next reference
                        127 = no future reference known

function findNextRef(cline_id, current_vertex):
    epoch_id = current_vertex / epoch_size
    entry = matrix[epoch_id * num_cache_lines + cline_id]

    if entry has MSB set:                     // Referenced in this epoch
        last_sub = entry & 0x7F
        curr_sub = (current_vertex % epoch_size) / sub_epoch_size
        if curr_sub <= last_sub:
            return 0                          // Still upcoming in this epoch
        // Past final access — check next epoch
        if epoch_id + 1 < num_epochs:
            next_entry = matrix[(epoch_id + 1) * num_cache_lines + cline_id]
            if next_entry has MSB set: return 1
            return (next_entry & 0x7F) + 1
        return 127                            // No future reference
    else:                                     // NOT referenced this epoch
        distance = entry & 0x7F
        return distance                       // Epochs until next reference
```

### P-OPT Insertion & Hit

```
function POPT_INSERT(cache_set, victim_idx):
    cache_set[victim_idx].rrpv = 6            // M_RRPV - 1 (same as SRRIP default)

function POPT_HIT(cache_set, idx):
    cache_set[idx].rrpv = 0                   // Same as SRRIP: reset to 0
```

## GraphBrew Validation Invariants (ECG Section A2)

| # | Invariant | Tolerance | Formula |
|---|-----------|-----------|---------|
| 1 | P-OPT approaches OPT miss rate | 1-5% | Within 1-5% of Belady on small graphs |
| 2 | P-OPT is reorder-agnostic | < 10% | `|Orig+POPT - DBG+POPT| / Orig+POPT < 0.10` |
| 3 | P-OPT beats all RRIP variants | strict ≤ | `POPT_miss ≤ {LRU, SRRIP, GRASP}_miss` |
| 4 | P-OPT improvement vs LRU > 10% | > 10% | `(LRU_miss - POPT_miss) / LRU_miss > 0.10` |

## GraphBrew Integration

- **Cache Policy**: `EvictionPolicy::POPT` in `bench/include/cache_sim/cache_sim.h`
- **Implementation**: `bench/include/graphbrew/partition/cagra/popt.h` (rereference matrix + BibTeX)
- **Context**: `bench/include/cache_sim/graph_cache_context.h` — rereference config, findNextRef()
- **Tested in**: ECG Section A2 (faithfulness), B1 (all-policy comparison)
- **Experiment config**: `scripts/experiments/ecg_config.py`
