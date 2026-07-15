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
PR/BFS/SSSP/BC/CC use gem5 `ecg.load2` and the equivalent Sniper fused record
sideband, eliminating explicit per-edge `extract2` from canonical Schedule-2
runs. Timing remains in-order-only in gem5; O3 is disabled until the epoch pair
is attached to its exact request. Historical preliminary rows generated before
the fused all-kernel port remain cache-metric evidence only.
Even PR fused timing is accepted only when live fused receipts validate against
the exported K2 sideband. Without receipts, the row remains cache-metric-only;
its packed-record software path can execute a different instruction stream than
the baseline and must not be described as a pure replacement-policy speedup.
gem5 analysis keeps only the benchmark-emitted ROI statistics block. Its later
automatic exit dump contains post-ROI teardown activity and is not a second
measurement.

## Required policy set

Every reported comparison includes:

1. LRU
2. SRRIP
3. GRASP
4. charged P-OPT
5. K2
6. online K2
7. K2+StreamShield
8. online K2+StreamShield

The canonical runner labels are `ECG:K2`, `ECG:K2_ONLINE`,
`ECG:K2_STREAMSHIELD`, and `ECG:K2_ONLINE_STREAMSHIELD`.

The replacement-quality profile additionally includes uncharged P-OPT,
`ECG:K1`, every static K2 arm, and `ECG:K2_ONLINE`. These are diagnostic
columns, not a reduced headline baseline set.

## Separated experiment questions

| Profile | Question | Prefetch/traffic treatment |
|---|---|---|
| `ecg_preliminary_5alg_3sim` | Does K2 move each of PR/BFS/SSSP/BC/CC in the right direction versus LRU, SRRIP, GRASP, and charged P-OPT within each simulator? | no prefetch; charged K2; common bounded `kron_s15_k4` cell |
| `ecg_preliminary_5alg_stride` | Does predictable structure prefetching hide K2 record latency without hiding its bandwidth cost? | matched STRIDE8 for every policy; report demand misses and total traffic separately |
| `ecg_replacement_baseline` | Which static K2 arm is best across PR/BFS/SSSP/BC/CC, and how much regret does online dueling incur? | no prefetch; ECG delivery uncharged |
| `ecg_cache_sim_factorial` | What do K2 and StreamShield contribute under hardware-faithful traffic? | STRIDE8 for all; ECG record traffic charged |
| `streamshield_sniper_realgraph` | Does the complete mechanism improve detailed-simulator time and traffic? | bounded, full six-policy matrix |

`ecg_charged=1` preserves each backend's executable transport rather than
forcing identical micro-operations. cache_sim widens the fused edge record and
therefore emphasizes replacement quality, while gem5 and Sniper execute
explicit record-delivery accesses that appear in their LLC accounting. The
preliminary ranks are consequently interpreted within each simulator; the
cache_sim column is a fused replacement diagnostic and the detailed-simulator
columns are transport-inclusive.

For matched STRIDE sensitivity, cache_sim and gem5 expose demand LLC misses
separately from prefetch traffic. Sniper's current NUCA statistics combine
demand and prefetch read misses, so its STRIDE rows report total LLC read-miss
traffic only; no Sniper demand-miss reduction is inferred from that aggregate.

K1 retains its original 16-bit/65,535-epoch range. K2 independently clamps to
32,768 epochs because its two 15-bit fields share the tiered 64-bit record.
Result rows retain the requested value as `ecg_epochs_requested`, report the
executed value as `ecg_epochs_effective`, and keep the compatibility column
`ecg_epochs` equal to the effective value. Thus a request for 65,535 epochs is
reported as 65,535 for K1 and 32,768 for K2.
Charged K1 uses an 8-byte edge record whenever destination, tier, and epoch no
longer fit in the original 32-bit edge word; uncharged replacement studies keep
the original 4-byte edge stream and deliver metadata out of band.
For unweighted PR/BFS/BC/CC, the 8-byte K2 record replaces the 4-byte vertex-ID
edge word. Weighted SSSP must also retain its weight, so the current faithful
path reads the existing 8-byte weighted edge plus the 8-byte K2 metadata record
(16 bytes total per relaxed edge). BC's runtime successor-DAG backward phase has
no static K2 record; CC remains scoped to symmetric/undirected graphs.

The static adaptive mapping is PR=`epoch_first`, BFS/SSSP=`degree_first`, and
BC/CC=`rrip_first`. `ECG:K2_ONLINE` remains kernel-name agnostic.

All governed property arrays are 4KB-aligned in cache_sim, gem5, and Sniper so
logical 16-vertex metadata lines match physical 64-byte cache lines. cache_sim
charges wide records against fixed synthetic IN/OUT stream bases indexed by the
global CSR edge position, avoiding allocator fragmentation and ASLR-dependent
set placement. Canonical cache_sim runs additionally use `setarch -R`.

All three simulators retain policy-aware pre-ROI cache warming and reset only
statistics at the ROI boundary; Sniper therefore runs with cache warming
enabled rather than the cold `--no-cache-warming` fast-forward mode. The
synthetic Sniper mechanism profile is the exception: it disables warming and
requires live fused-K2 receipts so the transport proof cannot become vacuous.
Sniper graph-policy rows also require the context-ready acknowledgement; P-OPT
rows additionally require `reref=1` before results are accepted.

## Headline real-graph cell

- Graph/kernel: web-Google PageRank, one iteration, DBG order
- Caches: 32kB L1D, 256kB L2, 2MB/16-way LLC, 64B lines
- Structure prefetch: pending a bounded Sniper configuration; the current
  generic STRIDE8 setting is diagnostic-only because it overprefetches
- Sniper: one core, virtual sideband domain, one complete PageRank iteration
- Metrics: simulated time, instruction count, L3 accesses/misses, bypass
  reads/writes, and total traffic

## Hardware accounting

- K2 record: 8 bytes (`dest32 | tier2 | epoch1_15 | epoch2_15`).
- The ECG successor reserves no LLC way.
- P-OPT is charged its rereference-matrix capacity.
- StreamShield is one request flag propagated through derived prefetches.
- No hidden matrix, per-access LLC metadata broadcast, or zero-latency bypass is
  permitted in a headline row.

## Reproducibility and matched pairs

Every matrix completion marker carries a strict full configuration hash over
the command, material environment, input fingerprints, binaries, and git state.
Matched no-prefetch/STRIDE analysis additionally uses a comparison hash that
normalizes only the output directory and intentional prefetch knobs. It omits
orchestration-script fingerprints and redundant git-state identity while
retaining benchmark/simulator binaries and configuration inputs. Legacy
comparison hashes may be reconstructed only from a resolved job whose strict
full hash exactly matches the completion marker.

Filtered reruns use distinct run directories. The runner refuses to replace a
broader resolved manifest with an `--only`/filtered subset; the pipeline
aggregates the resulting shard directories after validating each completion
marker.
