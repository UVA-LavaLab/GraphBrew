# ECG Successor HPCA Paper

## Working title

**Public title pending**

The public name should be graph-specific and should not reuse the workshop title.
The implementation remains under the `ECG_*` namespace.

## Prior publication gate

The preliminary paper is:

> A. T. Mughrabi, M. Baradaran, A. Samara, and K. Skadron, “ECG:
> Expressing Locality and Prefetching for Optimal Caching in Graph Structures,”
> IEEE IPDPSW 2024, pp. 520–525, DOI 10.1109/IPDPSW59749.2024.00094.

This is an archival IEEE workshop publication. HPCA 2027 does not permit a
substantially similar submission. Before registration:

1. send the PC chairs the workshop paper and a contribution-delta summary;
2. cite the workshop paper in the submission in the third person;
3. ensure the new title and abstract describe the new architecture, not the
   workshop mask/prefetch prototype;
4. retain written chair guidance with the artifact records.

Renaming alone does not make the submission eligible; the technical contribution
must be materially different.

## Required contribution delta

| IPDPSW 2024 ECG | HPCA successor |
|---|---|
| single metadata mask concept | K2 two-future-reference records |
| preliminary replacement/prefetch study | adaptive replacement plus StreamShield placement |
| basic trace-driven evaluation | cache_sim + gem5 + Sniper implementation and equivalence |
| conceptual graph instruction | executable `ecg.load2` / `ecg.stream.load2` request-bound ISA |
| no complete overhead attribution | K2-vs-bypass factorial, traffic, capacity, timing, and instruction accounting |
| PageRank-focused | PR plus traversal-aware BFS/SSSP policy design |

## Thesis

Graph analytics already stream an edge record before accessing irregular vertex
properties. The ECG successor carries compact future-reuse information in that record,
allowing the cache hierarchy to make graph-aware replacement and placement
decisions without P-OPT's reserved LLC ways or a separate metadata lookup.

## Mechanism

- **K2:** one 8-byte record carries `dest32 | epoch1_16 | epoch2_16`; replacement
  uses the nearer valid rereference.
- **Adaptive eviction:** PR uses `epoch_first`; BFS/SSSP use `degree_first`;
  BC/CC use `rrip_first`.
- **StreamShield:** one-touch packed records fill private caches, retain LLC-hit
  behavior, and do not allocate after an LLC miss.
- **ISA:** `ecg.load2` loads a cached K2 record; `ecg.stream.load2` adds the
  request-bound LLC no-allocation bit.

## Contributions

1. Edge-carried two-epoch reuse guidance with zero reserved LLC ways.
2. Request-bound cache-placement control that separates private-cache streaming
   from shared-LLC residency.
3. A shared eviction decision and exact delivery/decision gates across cache_sim,
   gem5, and Sniper.
4. Full accounting of record bytes, P-OPT reserved capacity, demand misses,
   total traffic, simulated time, and instruction count.

## Evaluation structure

1. Mechanism and factorial attribution in cache_sim on real graphs.
2. ISA and request-bound mechanism validation in gem5.
3. Scale and timing confirmation in Sniper using the full policy set.
4. Hardware/storage accounting and sensitivity analysis.

The real-graph Sniper matrix is the remaining gate before claiming overall
detailed-simulator superiority over P-OPT.
