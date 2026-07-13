# Methodology

Architecture definitions and diagrams are centralized in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Simulator roles

| Simulator | Role |
|---|---|
| cache_sim | Fast functional authority, real-graph factorials, bug finding |
| gem5 | Cycle-accurate record-load ISA and request-bound StreamShield confirmation |
| Sniper | Real-graph scale and paper timing matrix |

Absolute gem5 and Sniper miss rates are not compared because their inclusion,
frontend, and accounting models differ. Cross-simulator evidence is interpreted
as mechanism agreement and direction relative to each simulator's LRU.

## Required policy set

Every reported comparison includes:

1. LRU
2. SRRIP
3. GRASP
4. charged P-OPT
5. K2
6. K2+StreamShield

The canonical runner labels are `ECG:K2` and `ECG:K2_STREAMSHIELD`.

## Headline real-graph cell

- Graph/kernel: web-Google PageRank, one iteration, DBG order
- Caches: 32kB L1D, 256kB L2, 2MB/16-way LLC, 64B lines
- Structure prefetch: STRIDE8 for every policy
- Sniper: one core, virtual sideband domain, bounded 100M-instruction ROI
- Metrics: simulated time, instruction count, L3 accesses/misses, bypass
  reads/writes, and total traffic

## Hardware accounting

- K2 record: 8 bytes per streamed edge record.
- The ECG successor reserves no LLC way.
- P-OPT is charged its rereference-matrix capacity.
- StreamShield is one request flag propagated through derived prefetches.
- No hidden matrix, per-access LLC metadata broadcast, or zero-latency bypass is
  permitted in a headline row.
