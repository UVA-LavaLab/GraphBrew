# Methodology

Architecture definitions and diagrams are centralized in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Simulator roles

| Simulator | Role |
|---|---|
| cache_sim | Fast functional authority, real-graph factorials, bug finding |
| gem5 | Cycle-accurate masked property-load ISA and request-bound StreamShield confirmation |
| Sniper | Real-graph scale and paper timing matrix |

Absolute gem5 and Sniper miss rates are not compared because their inclusion,
frontend, and accounting models differ. Cross-simulator evidence is interpreted
as mechanism agreement and direction relative to each simulator's LRU.
PR/BFS/SSSP/BC/CC retain the fused indexed K2-I path as the default gem5
variant, and all five also implement the canonical K2-M path selected with
`GEM5_ECG_ISA_VARIANT=mask`. K2-M receives an already-computed address and
replaces a normal property load one-for-one. It passes the five-kernel mechanism
gate; matched performance timing remains pending.
gem5 additionally implements the architectural current-epoch/context CSRs and
snapshots them onto request-bound K2 loads. The harness allocates monotonic
nonzero context IDs, charges begin/end and current-epoch writes in the ROI, and
fails closed rather than reusing an ID. Integrated OoO stress and any future
drain/invalidation protocol for intentional ID reuse remain pending, so no new
performance claim follows from this implementation alone.
The serialized X86/Timing hint is cleared at the ordered context end; any late
prefetch fill fails closed rather than inheriting a later context.
Sniper's completed packed-record timing matrix is an idealized K2-I-like model,
not measured K2-I or mask-only timing. Tiny PR and weighted SSSP O3 runs prove request-local pair
delivery; scale runs remain on TimingSimpleCPU. Historical
gem5 rows labeled `ecg.load2`/`ecg.wload2` predate this correction and are not
reinterpreted without rerunning.

The K2-M Sniper model uses transport-matched guest execution: every policy
loads the same 8-byte records for unweighted kernels, while SSSP uses the same
native 8-byte weighted edge or general 12-byte fallback. A five-kernel gate
requires exact semantic results and at most 0.25% instruction divergence.
Current mechanism cells achieve exactly 1.000x instruction ratio. Timing remains
diagnostic pending fresh post-binding, equal-semantic-work rows. The current Sniper
implementation binds sideband K2 metadata to an explicit identical marker
around the exact edge-governed destination load for every policy; source
property loads and BC/CC non-edge phases remain unmarked. The marker snapshots
the per-core quantized current epoch and a monotonic ROI context, so victim
selection no longer rereads a mutable outer-vertex clock. It remains a Sniper
model rather than execution of the RISC-V CSR ISA.
For bounded transport-matched K2-M comparisons,
`--sniper-semantic-edge-limit` counts the same static graph-edge visit before
every policy branch and stops only before the next edge.
It requires `sg_kernel`, one core, and `ecg_isa_variant=mask`. The mandatory
`[SEMANTIC-ROI ...]` marker records the requested limit, actual
visits, and whether execution was truncated. This cap is mutually exclusive with
the committed-instruction cap. Semantic-capped rows may be paired only when all
three marker fields match; instruction-capped rows remain cache-direction evidence
only because policies can execute different amounts of graph work.
When truncation occurs, the semantic checksum certifies equality of the same
deterministic execution prefix, not completion of the full graph algorithm.
Full-result correctness remains a separate uncapped gate.
Even K2-I fused timing is accepted only when live fused receipts validate against
the exported K2 sideband. Without receipts, the row remains cache-metric-only;
its packed-record software path can execute a different instruction stream than
the baseline and must not be described as a pure replacement-policy speedup.
gem5 analysis keeps only the benchmark-emitted ROI statistics block. Its later
automatic exit dump contains post-ROI teardown activity and is not a second
measurement.

## Future K2-M headline policy set

The next headline comparison, after K2-M implementation, must include:

1. LRU
2. SRRIP
3. GRASP
4. charged P-OPT
5. Hawkeye
6. K2-M
7. online K2-M
8. K2-M+StreamShield
9. online K2-M+StreamShield

New runner labels must distinguish `K2-M` from `K2-I`. Existing labels
`ECG:K2`, `ECG:K2_ONLINE`, `ECG:K2_STREAMSHIELD`, and
`ECG:K2_ONLINE_STREAMSHIELD` denote historical/prototype K2-I timing until the
runner split lands.
`ECG:K2_ONLINE_ADAPTIVE_STREAMSHIELD` is retained only as a placement ablation.

The replacement-quality profile additionally includes uncharged P-OPT,
`ECG:K1`, every static K2 arm, and `ECG:K2_ONLINE`. These are diagnostic
columns, not a reduced headline baseline set.

`cache_sim` has no instruction-PC input. Its `HAWKEYE_PROXY` arm therefore uses
a compile-time static graph-access-site ID, retains faithful OPTgen/sampler/
predictor/RRIP mechanics, and is labeled `proxy_not_real_instruction_pc` in
every row. It is a fast diagnostic, not the headline Hawkeye result. gem5 must
supply the faithful real-PC comparison; that adapter is implemented, but no
result is frozen yet.
The gem5 Hawkeye adapter is scoped to the artifact's conventional uncompressed
set-associative LLC; compressed-block move semantics are not claimed.

## Separated experiment questions

| Profile | Question | Prefetch/traffic treatment |
|---|---|---|
| `ecg_preliminary_5alg_3sim` | Does K2 move each of PR/BFS/SSSP/BC/CC in the right direction versus LRU, SRRIP, GRASP, and charged P-OPT within each simulator? | no prefetch; charged K2; common bounded `kron_s15_k4` cell |
| `ecg_preliminary_5alg_stride` | Does predictable structure prefetching hide K2 record latency without hiding its bandwidth cost? | matched STRIDE8 for every policy; report demand misses and total traffic separately |
| `ecg_replacement_baseline` | Which static K2 arm is best across PR/BFS/SSSP/BC/CC, and how much regret does online dueling incur? | no prefetch; ECG delivery uncharged |
| `ecg_cache_sim_factorial` | What do K2 and StreamShield contribute under hardware-faithful traffic? | STRIDE8 for all; ECG record traffic charged |
| `ecg_3sim_sampled_allalg` | Do all three backends show coherent bounded-workload behavior across all five kernels? | no prefetch; deterministic compact samples; full semantic completion |
| `ecg_sniper_sampled_pr_streamengine` | Can fused K2 delivery tolerate its record bandwidth on equal-work sampled PageRank? | no prefetch; GRASP/charged P-OPT/K2-online+SS; record misses remain in Sniper LLC accounting |
| `ecg_sniper_realgraph_warm_probe` | Do full web-Google warm LRU and K2 reach a detailed ROI? | explicit SIFT; normal cache warming; 100K detailed instructions |
| `ecg_sniper_realgraph_600m` | What cache direction appears on full graphs under a prior-paper-style detailed ROI? | explicit SIFT; normal cache warming; detailed ROI capped at 600M instructions; timing invalid |
| `ecg_sniper_semantic_gate` | Can all five kernels compare policies after exactly the same number of static edge visits? | no prefetch; exact semantic marker required; no instruction cap |
| `streamshield_sniper_realgraph` | Does the complete mechanism improve detailed-simulator time and traffic? | bounded, full six-policy matrix |

`ecg_charged=1` preserves each backend's executable transport rather than
forcing identical micro-operations. cache_sim widens the fused edge record and
therefore emphasizes replacement quality, while gem5 and Sniper execute
explicit record-delivery accesses that appear in their LLC accounting. The
preliminary ranks are consequently interpreted within each simulator; the
cache_sim column is a fused replacement diagnostic and the detailed-simulator
columns are transport-inclusive.

The Sniper fused model registers the packed K2 record range and treats the
64-bit record load as the delivery event. Non-tracing runs execute no
software-only `extract2`/trace delivery call. Environment-controlled hint paths
are resolved once before the ROI so disabled instrumentation does not distort
cross-policy instruction counts.
Canonical fused PR/BFS/BC/CC loops are explicitly split into no-trace and
legacy/traced forms so the no-trace path executes no per-edge delivery or clear
condition. Weighted SSSP already uses an outer fused-path split.

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
edge word. Eligible weighted SSSP (`N <= 2^24`, weights in `[1,255]`) replaces
its original 8-byte edge with one compact 8-byte K2 record. Other weighted
graphs retain the 8-byte `dest32|weight32` edge plus a parallel 4-byte sidecar
(12 bytes total). BC's runtime successor-DAG backward phase has no static K2
record; CC remains scoped to symmetric/undirected graphs.

The static adaptive mapping is PR=`epoch_first`, BFS/SSSP=`degree_first`, and
BC/CC=`rrip_first`. `ECG:K2_ONLINE` remains kernel-name agnostic.

All governed property arrays are 4KB-aligned in cache_sim, gem5, and Sniper so
logical 16-vertex metadata lines match physical 64-byte cache lines. cache_sim
charges wide records against fixed synthetic IN/OUT stream bases indexed by the
global CSR edge position, avoiding allocator fragmentation and ASLR-dependent
set placement. Canonical cache_sim runs additionally use `setarch -R`.

### Property-array scope

GRASP and P-OPT classify every registered vertex-property array in all three
simulators. K2 retains that GRASP insertion fallback, but request-bound epochs
apply only to loads with a valid static edge record:

| Kernel | GRASP/P-OPT registered arrays | K2 epoch-governed loads |
|---|---|---|
| PR | `scores`, `contrib` | `contrib` |
| BFS | `parent` | `parent` |
| SSSP | `dist` | `dist` |
| BC | `scores`, `depth`, `path_counts`, `deltas` | forward `depth`, `path_counts` |
| CC | `comp` | edge-governed `comp[dest]` |

BC's backward successor-DAG accesses and CC's union-find pointer chasing deliver
no new K2 epoch because a static edge record would be invalid there. Epochs are
line-resident metadata, so a plain access does not eagerly erase a stamp from an
earlier governed load. The runner records both `property_regions` and
`ecg_epoch_regions` in every row.

All three simulators retain policy-aware pre-ROI cache warming and reset only
statistics at the ROI boundary; Sniper therefore runs with cache warming
enabled rather than the cold `--no-cache-warming` fast-forward mode. The
synthetic Sniper mechanism profile is the exception: it disables warming and
requires live fused-K2 receipts so the transport proof cannot become vacuous.
Sniper graph-policy rows also require the context-ready acknowledgement; P-OPT
rows additionally require `reref=1` before results are accepted.

The sampled matrix is a backend-corroboration diagnostic, not scale authority.
Its edge-prefix/induced samples, high-degree root remapping, smaller
graph sizes, altered LLC-to-working-set ratios, and symmetrized citation graph can alter policy
ranking. Full-graph cache_sim results remain authoritative for replacement
quality. The 600M Sniper profile follows DROPLET's bounded-ROI precedent, while
GRASP and P-OPT similarly used representative-iteration sampling. Because K2
changes the executed instruction stream, capped rows support cache/direction
claims only and are marked `timing_valid_for_speedup=0`.

The original 360-row sampled matrix remains valid for cache metrics, but its
cross-policy Sniper timing predates removal of disabled hot-path instrumentation
and is not timing evidence. The historical
`ecg_sniper_sampled_pr_streamengine` profile remains an attribution diagnostic.
Its P-OPT row charges reserved LLC capacity, while matrix-stream latency remains
outside Sniper target time and is reported separately.

The 120-row post-scope Sniper rerun at
`ecg_sniper_sampled_allalg_compact_scope_final_20260721` is the sampled
all-kernel **idealized packed-record K2-I-like model** authority, not measured
K2-I ISA timing and not the core K2-M timing result.
It combines the surgical no-trace loops, BC
`depth,path_counts` scope, SSSP source-load isolation, CC phase clears, and the
compact weighted SSSP record. Every BC row records the corrected governed
regions, and every eligible SSSP K2 row reports one 8-byte replacement record.
The Sniper frontend still executes x86 extraction/indexed loads and infers
metadata from source plus property line, so neither total speedup nor TPI
isolates K2-M. The 12-byte general weighted fallback remains validated. The
earlier surgical matrix is historical attribution evidence.

All aggregate ratios use the geometric mean across the applicable graph/kernel
cells. P-OPT and K2 overheads appear in different columns by construction:
P-OPT matrix streaming is added to effective LLC misses/traffic, while K2 record
delivery primarily changes executed instructions and explicit record accesses.
K2 reserves no LLC data way, but its line metadata, current-epoch channel,
8-byte records, and weighted sidecars must all be charged. P-OPT instead pays
reserved capacity plus modeled matrix traffic. Equal-area results are pending.

Sniper CACHE_ONLY warming updates cache contents but intentionally does not
model time. The installed GraphBrew patch therefore leaves exact queue state
untouched and suppresses shared-memory elapsed-time accumulation until DETAILED
mode. Full web-Google warm LRU and K2 100K probes both reach ROI with this
configuration.

Sniper's P-OPT host emulator computes each candidate's dynamic rereference
distance once per eviction and applies an equivalent closed-form RRIP aging
step. `SNIPER_POPT_FAST=0` retains the legacy repeated-consultation path for
equivalence checks. This changes host simulation cost only; victim decisions,
target timing, reserved capacity, and matrix-stream accounting remain unchanged.
gem5 already memoizes one distance per candidate before its RRIP tie loop, so it
does not require the Sniper-specific fast-path repair.

RISC-V guest kernels use `-funswitch-loops` in addition to the common `-O1`
build. Every policy runs in the same binary; the flag hoists environment-selected
delivery modes out of edge loops instead of charging K2 repeated mode checks.
TimingSimpleCPU uses the serialized mailbox fallback; when the O3 request-bound
producer is enabled, the LLC accepts only per-Request K2 metadata and disables
mailbox fallback for plain loads.

## Headline real-graph cell

- Graph/kernel: web-Google PageRank, one iteration, DBG order
- Caches: 32kB L1D, 256kB L2, 2MB/16-way LLC, 64B lines
- Structure prefetch: pending a bounded Sniper configuration; the current
  generic STRIDE8 setting is diagnostic-only because it overprefetches
- Sniper: one core, virtual sideband domain, one complete PageRank iteration
- Metrics: simulated time, instruction count, L3 accesses/misses, bypass
  reads/writes, and total traffic

## Hardware accounting

- Unweighted K2 record: 8 bytes
  (`dest32 | tier2 | epoch1_15 | epoch2_15`).
- Weighted SSSP: one replacing 8-byte compact record when eligible; otherwise a
  4-byte sidecar plus the existing 8-byte weighted edge.
- The ECG successor reserves no LLC data way; metadata area remains charged.
- Primary equal-data-capacity rows retain the full 16-way LLC and disclose the
  added metadata SRAM/logic cost.
- Separate 15-way and 14-way K2 rows are equal-silicon sensitivities, not the
  physical mechanism used to store K2 metadata.
- The runner applies `--k2-l3-ways` only to Schedule-2 K2 policies and records
  both baseline and effective geometry. Conventional baselines remain 16-way;
  charged P-OPT independently retains its matrix-capacity charge.
- P-OPT is charged its rereference-matrix capacity.
- Non-P-OPT K2-M rows must neither construct nor load the P-OPT rereference
  matrix. Their future-reuse state comes only from the streamed K2 records and
  resident line metadata.
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
