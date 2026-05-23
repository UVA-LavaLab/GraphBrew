# Baseline Cache Replacement Policies

## Overview

GraphBrew's cache simulation includes 5 non-graph-aware baselines used as comparison points for GRASP, P-OPT, and ECG. Getting these right is essential — if baselines are weak, improvements are inflated; if baselines don't match published implementations, comparisons with other papers' results are invalid.

---

## LRU — Least Recently Used

**Eviction**: Remove the line accessed least recently.

```
On access:  move line to MRU position
On evict:   remove line from LRU position
```

- Stack property: increasing cache size never increases misses
- True LRU (not approximation) — matches P-OPT repo's LRU exactly

---

## FIFO — First In, First Out

**Eviction**: Remove the line inserted earliest.

```
On insert:  add to back of queue
On hit:     do NOT change position (unlike LRU)
On evict:   remove from front of queue
```

- No stack property (Belady's anomaly possible)
- Simpler than LRU — no movement on hits

---

## RANDOM — Random Eviction

**Eviction**: Choose a random cache line from the set.

```
On evict:   uniform random selection from valid lines
```

- Zero metadata overhead
- Lower bound for policy quality

---

## LFU — Least Frequently Used

**Eviction**: Remove the line with the fewest total accesses.

```
On access:  increment access counter
On evict:   remove line with smallest counter
```

- Similar concept to GRASP's hub preference but without structural awareness
- Suffers from frequency accumulation (stale high-frequency lines persist)

---

## SRRIP — Static Re-Reference Interval Prediction

### Citation

```bibtex
@inproceedings{rrip-isca10,
  title     = {High Performance Cache Replacement Using Re-Reference Interval Prediction (RRIP)},
  author    = {Jaleel, Aamer and Theobald, Kevin B. and Steely, Simon C. and Emer, Joel},
  booktitle = {Proceedings of the 37th Annual International Symposium on Computer Architecture (ISCA)},
  pages     = {90--101},
  year      = {2010},
  organization = {ACM}
}
```

**Paper PDF**: [people.csail.mit.edu/emer/papers/2010.06.isca.rrip.pdf](https://people.csail.mit.edu/emer/papers/2010.06.isca.rrip.pdf) (referenced in P-OPT repo)

### Why SRRIP Matters

SRRIP is the **foundation** for GRASP and ECG. Both extend SRRIP's RRPV mechanism with graph-structural metadata. If SRRIP is wrong, every graph-aware policy built on top is wrong.

### Algorithm Pseudo-Code (from GraphBrew cache_sim.h)

```
M = 3 (bits), max RRPV = 2^M - 1 = 7

SRRIP_INSERT(cache_set, victim_idx):
    cache_set[victim_idx].rrpv = 2            // Long re-reference interval
                                               // M-1 = 3-1 = 2 (scan-resistant default)

SRRIP_HIT(cache_set, idx):
    cache_set[idx].rrpv = 0                   // Near-immediate re-reference

findVictimSRRIP(cache_set):
    while true:
        for i in 0..associativity-1:
            if cache_set[i].rrpv == 3:        // Find line with distant re-reference
                return i
        // Increment all RRPVs (aging)
        for i in 0..associativity-1:
            if cache_set[i].rrpv < 3:
                cache_set[i].rrpv += 1
```

### DRRIP (Dynamic RRIP)

P-OPT repo also implements DRRIP — set-dueling between SRRIP & BRRIP:
- **SRRIP**: Insert at RRPV = M-1 (long re-reference)
- **BRRIP**: Insert at RRPV = M (distant, bimodal insertion)
- **DRRIP**: Dedicates some sets to SRRIP, some to BRRIP, uses miss counters to choose the winner for the remaining sets

### Relationship to Graph-Aware Policies

| Policy | Insertion RRPV | Hit RRPV | Eviction | Graph Awareness |
|--------|---------------|----------|----------|----------------|
| SRRIP | Static (M-1) | 0 | Max RRPV + aging | None |
| GRASP | DBG-dependent (0 to 7) | 0 (hub) or decrement | Same as SRRIP | Vertex degree |
| ECG | DBG from MASK + P-OPT at eviction | 0 (hub) or decrement | 3-level tiebreaking | Degree + oracle |

## GraphBrew Integration

- All baselines in: `bench/include/cache_sim/cache_sim.h` — `EvictionPolicy` enum
- Experiment config: `scripts/experiments/ecg_config.py` — `BASELINE_POLICIES` list
- Used as: Comparison baselines in ECG experiments (B1: all 9 policies)
