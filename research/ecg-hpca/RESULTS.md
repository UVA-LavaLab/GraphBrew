# Frozen Results

## Legacy real-graph cache_sim factorial (superseded)

These rows used the earlier cache_sim prototype that bypassed LLC lookup as well
as allocation. Current StreamShield preserves LLC hits and suppresses only miss
allocation, matching gem5 and Sniper. The profile
`ecg_cache_sim_factorial` must be rerun before these values are used in the paper.

PR `-i1 -o5`, 32kB L1D, 256kB L2, 16-way LLC, STRIDE8. Lower demand
memory accesses is better.

| Graph | LRU | SRRIP | GRASP | P-OPT | K1 | K1+SS | K2 | K2+SS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| web-Google / 2MB | 1,758,103 | 1,390,247 | 1,330,034 | 1,036,428 | 1,080,415 | 997,671 | 815,073 | **764,123** |
| soc-pokec / 2MB | 13,400,665 | 11,489,464 | 9,249,576 | 8,143,075 | 7,551,255 | 7,323,620 | 6,433,736 | **6,228,099** |
| cit-Patents / 8MB | 9,389,641 | 7,624,847 | 6,251,112 | 4,769,337 | 4,288,176 | 4,063,879 | 3,943,972 | **3,747,240** |

Historical weighted attribution under full lookup bypass: **K2 77.3%**,
**StreamShield 22.7%**. This is retained for provenance, not as a current claim.

Canonical corrected factorial profile: `ecg_cache_sim_factorial`.

## Corrected tag-hit-preserving StreamShield factorial

The current factorial contains 36 rows: three real graphs x 12 policies, PR
`-i1 -o5`, matched STRIDE8, charged K1/K2 records, size-correct P-OPT, and
StreamShield that preserves LLC lookup/hits and suppresses only miss allocation.
“Full” is online K2+StreamShield.

| Graph | K2 vs K1 demand reduction | Online vs K2 demand reduction | StreamShield on online demand | StreamShield on online traffic | Full demand reduction vs charged P-OPT | Full traffic delta vs charged P-OPT |
|---|---:|---:|---:|---:|---:|---:|
| web-Google / 2MB | 28.05% | 5.38% | 7.30% | 3.09% | 35.90% | **+7.58%** |
| soc-pokec / 2MB | 18.14% | 10.80% | 4.81% | 2.70% | 26.37% | **+2.91%** |
| cit-Patents / 8MB | 9.18% | 7.10% | 8.05% | 3.90% | 31.85% | **+5.42%** |

Across graphs, StreamShield adds a 6.73% geomean demand-miss reduction and a
3.23% total-traffic reduction beyond online K2. Full ECG reduces demand misses
31.48% versus charged P-OPT and 7.72% versus uncharged practical P-OPT, but
total traffic remains 5.28% and 29.17% higher, respectively. It also reduces
demand misses 34.42% versus GRASP while using 2.09% more traffic.

Weighted avoided-demand-miss attribution relative to K1, split at online K2,
is **K2+online 83.94% / StreamShield 16.06%**. The static K2 split from the same
K1 baseline is 81.62% / 18.38%. These replace the legacy 77.3% / 22.7%
lookup-bypass attribution. The corrected factorial therefore validates
StreamShield as a useful incremental placement mechanism, but not as a
total-bandwidth win over P-OPT.

Adaptive allocate-vs-shield placement lands between the two static choices:

| Graph | Adaptive miss delta vs static SS | Adaptive traffic delta vs static SS | Adaptive demand reduction vs online K2 | Adaptive traffic reduction vs online K2 |
|---|---:|---:|---:|---:|
| web-Google / 2MB | +4.54% | +1.84% | 3.09% | 1.31% |
| soc-pokec / 2MB | +1.50% | +0.83% | 3.38% | 1.90% |
| cit-Patents / 8MB | +1.84% | +0.85% | 6.35% | 3.08% |

Geomean adaptive placement reduces demand misses 4.28% and traffic 2.10%
versus always allocating online K2, but trails static StreamShield by 2.62%
misses and 1.17% traffic on PR. This is the expected learning/leader overhead
when shielding is uniformly favorable; adaptive placement is intended to avoid
harm on kernels or phases with record reuse.

Aggregate: `results/ecg_experiments/paper_pipeline/`
`ecg_cache_sim_factorial_adaptive_final_20260715/aggregate/roi_matrix_all.csv`.

## StreamShield all-kernel generality

The no-prefetch generality matrix contains 120 rows: three real graphs x
PR/BFS/SSSP/BC/CC x eight policies, with charged K2 records and the full
LRU/SRRIP/GRASP/P-OPT baseline set. This isolates LLC placement rather than
prefetch latency hiding.

| Kernel | Static SS demand-miss reduction vs allocate | Adaptive demand-miss reduction vs allocate | Adaptive regret vs static SS |
|---|---:|---:|---:|
| PR | 3.17% | 2.02% | 1.18% |
| BFS | 6.81% | 5.34% | 1.58% |
| SSSP | 1.31% | 0.62% | 0.69% |
| BC | 0.65% | 0.31% | 0.35% |
| CC | 2.11% | 1.18% | 0.95% |

Static StreamShield beats always allocating K2 records on all 15 graph/kernel
cells. Adaptive placement beats static StreamShield on only 2/15 cells and is
0.95% worse in geomean misses/traffic, with 3.36% maximum positive regret.
There is therefore no evidence that LLC record reuse justifies the extra
placement selector in this corpus. The final design uses generic static
StreamShield; adaptive placement remains a validated, default-off ablation.

Because this matrix disables prefetching while charging K2 records, it does not
claim overall policy superiority. Its purpose is the allocate-vs-shield
decision.

Aggregate: `results/ecg_experiments/paper_pipeline/`
`ecg_streamshield_generality_final_20260715/aggregate/roi_matrix_all.csv`.

## Tiered K2 and online selection (cache_sim functional authority)

The complete `ecg_replacement_baseline` contains 180 rows: three real graphs x
PR/BFS/SSSP/BC/CC x 12 policies. It compares uncharged and charged P-OPT, K1,
all five static tiered-K2 arms, and `ECG:K2_ONLINE`, with ECG delivery uncharged
to isolate replacement quality.

Every static arm is best on at least one cell: GRASP 5, degree 4, epoch 3, RRIP
2, and LRU 1. This variation validates online selection rather than one
benchmark-name mapping.

| Reference | Online K2 geomean LLC-miss delta | Cells with fewer online misses |
|---|---:|---:|
| per-cell best static K2 arm | +0.26% | 8/15 |
| LRU | -19.92% | 15/15 |
| SRRIP | -13.99% | 14/15 |
| GRASP | -4.49% | 10/15 |
| charged P-OPT | -17.59% | 15/15 |
| uncharged practical P-OPT | -4.49% | 11/15 |
| K1 | -20.52% | 15/15 |

Online K2's worst positive regret versus the best static arm is 6.72%; it also
beats every static arm on 8/15 cells because followers can combine the
leader-selected behavior over time. These are cache_sim replacement-authority
results, not final detailed-simulator performance numbers.

Aggregate: `results/ecg_experiments/paper_pipeline/`
`ecg_replacement_baseline_final_20260714/aggregate/`
`online_dueling_regret.csv`.

The post-`8ef03d28` 15-cell Schedule-2 gate passes for
cache_sim/gem5/Sniper x PR/BFS/SSSP/BC/CC. Every detailed-simulator kernel
matches 32/32 expected K2 deliveries with zero distance mismatches and obeys the
shared victim specification. gem5 executes the real RISC-V masked property load;
Sniper validates the equivalent fused sideband immediately before the governed
access. Tiny O3 PR and weighted SSSP runs each deliver 8/8 traced K2 request
extensions to the correct property line. This is mechanism/spec evidence, not a
frozen real-graph performance ranking.

### Masked-load target-instruction correction

Replacement-only authority already showed the forced K2 `grasp_only` arm
beating standalone GRASP on 15/15 real graph/kernel cells, with a 0.9899
geomean miss ratio. The initial masked-load gem5 timing path nevertheless
charged software scaffolding inside every edge iteration.

On web-Google-n16 PR, successive corrections give:

| Path | ROI instruction ratio vs LRU | Speedup vs LRU |
|---|---:|---:|
| separate clear + guest EXPECT trace | 1.888x | 0.796x |
| clear folded into masked load | 1.470x | 0.877x |
| EXPECT trace moved into gem5 | 1.098x | 1.049x |
| common binary with loop-invariant modes unswitched | **0.977x** | **1.080x** |

The final matched cell is 1.005x faster than GRASP while retaining a lower L3
miss rate (0.432 versus 0.506). P-OPT remains 1.078x faster on this PR cell
because its live matrix reaches a 0.369 miss rate. This is one bounded
diagnostic, not the final all-kernel result; the clean masked-load gem5 matrix
must be rerun.

The no-prefetch `kron_s15_k4` preliminary matrix is complete:

| Kernel | cache_sim fused K2 rank | gem5 explicit K2 rank | Sniper instrumented K2 rank | Robust read |
|---|---:|---:|---:|---|
| PR | 5/6 | 6/6 | 5/6 | K2 trails conventional policies |
| BFS | 5/6 | 5/6 | 5/6 | K2 trails; cache_sim online K2 improves |
| SSSP | 5/6 | 5/6 | 6/6 | K2 traffic is unfavorable |
| BC | **2/6** | **2/6** | **2/6** | consistent second behind GRASP |
| CC | **2/6** | 4/6 | 4/6 | beats charged P-OPT, not GRASP |

Aggregate: `results/ecg_experiments/paper_pipeline/`
`ecg_preliminary_5alg_final_20260714/aggregate/`
`preliminary_5alg_policy_ranks.csv`.
This synthetic result says BC is the strongest portable K2 candidate; it does
not establish a real-graph win.

This preliminary table predates the all-kernel fused port. Its cache_sim column
uses the fused widened-record model, while the historical gem5/Sniper rows
include explicit instrumented delivery. Those detailed-simulator ranks remain
cache diagnostics; future reruns use fused delivery and require live receipts.

The bounded preliminary performance profile is
`ecg_preliminary_5alg_3sim`. It uses the same `kron_s15_k4` workload and cache
geometry for all three simulators with LRU/SRRIP/GRASP/charged-P-OPT/K2/
K2-online and produced the no-prefetch table above. Conclusions use only
within-simulator ranks and deltas.
The paired `ecg_preliminary_5alg_stride` profile repeats the matrix with
matched STRIDE8 and keeps demand-miss and total-traffic conclusions separate.
Sniper K2-vs-LRU speedup is suppressed when fused receipts are absent. Those
rows contribute only transport-inclusive cache diagnostics; their
packed-record instruction count is not an apples-to-apples replacement-only
comparison.

## Sampled three-simulator full-work matrix

The completed `ecg_3sim_sampled_allalg` matrix contains 360 valid rows:
cache_sim/gem5/Sniper x web-Google-n16/soc-pokec-n16/
cit-Patents-n18-sym x PR/BFS/SSSP/BC/CC x eight policies. Every row runs to
semantic completion; the strict graph/backend/metric/transport gate passes.

Geomean effective LLC-miss reduction versus each simulator's own LRU baseline
(positive is better):

| Policy | cache_sim | gem5 | Sniper |
|---|---:|---:|---:|
| SRRIP | 5.85% | 3.65% | 3.68% |
| GRASP | 12.32% | 12.55% | 13.81% |
| charged P-OPT | -22.20% | -20.90% | -27.65% |
| K2 | -5.58% | -9.37% | -4.27% |
| K2-online | -5.40% | -4.68% | -4.02% |
| K2+StreamShield | -4.46% | -5.73% | -2.98% |
| K2-online+StreamShield | -2.97% | -3.18% | -1.17% |

“Effective” includes charged matrix-stream overhead for P-OPT; it is not the
raw demand-miss column.
Before that analytical stream charge, sampled P-OPT reduces raw L3 misses by
18.51% in cache_sim, 15.31% in gem5, and 8.40% in Sniper.

The all-kernel miss aggregate therefore does **not** validate a generic K2
replacement win. Its small working sets make fixed record delivery a dominant
cost. The rows remain useful for backend corroboration of mature policies:
SRRIP has the same direction in 14/15 cells, GRASP in 11/15, and the mean
per-cell eight-policy Spearman rank correlation is 0.84 for gem5/Sniper.
Cache_sim-to-detailed correlation is lower (0.50-0.54), so absolute ranks remain
substrate-sensitive.

Against full-graph cache_sim `ecg_streamshield_generality`, sampled
K2-online+StreamShield agrees in direction on 14/15 charged cells and sampled
K2-online on 12/15. Against the uncharged replacement authority
`ecg_replacement_baseline`, K2-online agrees on only 6/15 cells. Thus the
samples reproduce the **charged-overhead regime**, not the underlying
uncharged K2 replacement benefit. Use them only as bounded cross-simulator
diagnostics. Full-graph cache_sim remains the scale/replacement authority, and
the live 600M Sniper profile provides paper-precedent full-graph bounded-ROI
evidence.

Aggregate: `results/ecg_experiments/paper_pipeline/`
`ecg_3sim_sampled_allalg_final_20260717_v2/aggregate/`.
The rank statistic is frozen in
`sampled_crosssim_rank_correlation.csv` using mean per-cell Spearman over all
eight policies and effective LLC misses.

### Corrected sampled Sniper all-kernel timing

The corrected Sniper-only rerun contains 120 valid rows: three deterministic
samples x PR/BFS/SSSP/BC/CC x eight policies. The strict coverage gate passes
and every graph/kernel group is hash-consistent.

Geomean across all 15 graph/kernel cells versus LRU:

| Policy | Speedup | TPI speedup | Instruction ratio | Effective L3-miss reduction |
|---|---:|---:|---:|---:|
| SRRIP | 1.025x | 1.025x | 1.000x | 3.44% |
| GRASP | 1.099x | 1.099x | 1.000x | **15.19%** |
| charged P-OPT | 1.075x | 1.075x | 1.000x | -19.26% |
| K2 | **1.154x** | **1.258x** | 1.090x | 1.31% |
| K2-online | **1.154x** | **1.258x** | 1.090x | 0.62% |
| K2+StreamShield | 1.137x | 1.239x | 1.090x | 1.81% |
| K2-online+StreamShield | 1.143x | 1.246x | 1.090x | 1.59% |

K2-online+StreamShield kernel geomean:

| Kernel | Speedup vs LRU | vs GRASP | vs charged P-OPT | Effective miss reduction vs LRU |
|---|---:|---:|---:|---:|
| PR | **1.357x** | 1.209x | 1.155x | 16.54% |
| BFS | **1.334x** | 1.210x | 1.239x | 4.43% |
| SSSP | **0.973x** | 0.881x | 0.873x | -19.25% |
| BC | 1.007x | 0.992x | 1.013x | 5.34% |
| CC | 1.101x | 0.953x | 1.073x | -2.53% |

GRASP wins 7/15 timing cells, the four K2 variants collectively win 7/15,
and P-OPT wins 1/15. The 4-byte weighted sidecar raises final K2-online+
StreamShield from 1.052x to 1.143x overall and reduces its instruction ratio
from 1.182x to 1.090x. It now beats GRASP and charged P-OPT in overall sampled
geomean, but still does not dominate every kernel.

Weighted SSSP improves from 0.642x to 0.973x versus LRU. Its 12-byte path
(8-byte weighted edge + 4-byte sidecar) still carries a 1.314x SSSP instruction
ratio; a 1.278x per-instruction TPI improvement nearly repays that cost. It
remains slower than GRASP and P-OPT, so a zero-record GRASP transport arm is
still required for per-kernel no-regret behavior.

The BFS geomean is sensitive to the tiny web-Google-n16 cell: K2-online+
StreamShield is 1.877x there, while cit-Patents and soc-pokec are 0.968x and
1.306x; excluding web-Google gives 1.124x for BFS. Excluding that single
200K-instruction graph/kernel cell from the overall aggregate leaves
K2-online+StreamShield at 1.103x and GRASP at 1.106x, so their overall ordering
is not robust to the shortest workload. Sniper assigns the full CPI stack to
`unknown`, so the TPI gains cannot be decomposed into LLC, DRAM, or pipeline
components.

The effective-miss column charges P-OPT's matrix stream as additional cache-line
fills, which dominates small working sets. K2's main overhead instead appears
as the 1.090x instruction ratio. StreamShield also trades timing for placement
in BC: adding it to online K2 reduces BC speedup by about 6% even though the
cache_sim placement study favors static shielding.

The supported scope is therefore strong PR/BFS and positive sampled overall
geomean, with near-neutral BC/SSSP and modest CC benefit. Per-kernel no-regret
still requires a true zero-record baseline arm.

Aggregate: `results/ecg_experiments/paper_pipeline/`
`ecg_sniper_sampled_allalg_weighted_sidecar_final_20260719/aggregate/`.

### Fused sampled PageRank timing

The dedicated `ecg_sniper_sampled_pr_streamengine` profile contains nine
equal-work rows: three deterministic graph samples x GRASP/charged P-OPT/
K2-online+StreamShield. The Sniper fused path treats the packed 64-bit record
load as the delivery event. Disabled hint checks are hoisted before the ROI and
non-tracing runs execute no software-only delivery call.

Geomean K2-online+StreamShield ratios:

| Baseline | Simulated time | Total speedup | Ticks/instruction speedup | Instructions | L3 accesses | Total L3 misses | Non-record miss reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRASP | 0.828x | 1.207x | 1.149x | 0.952x | 1.159x | 1.085x | 35.92% |
| charged P-OPT | 0.870x | 1.150x | 1.094x | 0.952x | 1.155x | 1.333x | 21.27% |

`sniper_stream_bypass_reads` counts bypassed record LLC misses, so
`l3_misses - sniper_stream_bypass_reads` isolates the non-record miss stream.
The packed-record loop executes 4.8% fewer instructions than the baseline CSR
iterator, so total speedup is reported alongside ticks-per-instruction. Even
after this decomposition, K2 improves TPI by 1.149x versus GRASP and 1.094x
versus P-OPT.

The result confirms the intended tradeoff on sampled PageRank. LLC lookup
pressure rises about 16%, while total LLC misses rise 8.50% versus GRASP.
Against P-OPT, K2 has 33.29% more raw LLC misses; after adding P-OPT's modeled
matrix-stream lines, the overhead is 14.03% and K2's non-record miss reduction
is 32.65%. P-OPT's simulated time remains favorable because matrix-stream
latency is not added to Sniper target time. Sniper assigns the full CPI stack to
`unknown`, so the TPI improvement cannot be decomposed further. This is bounded
sampled-PR evidence, not a full-graph or all-kernel timing claim.

Aggregate: `results/ecg_experiments/paper_pipeline/`
`ecg_sniper_sampled_pr_streamengine_final_v2_20260717/aggregate/`.

### Full-graph warm Sniper gate

The queue blocker was caused by CACHE_ONLY warmup accumulating timing in
`history_list` and `ShmemPerfModel` even though the interval core clock was not
advancing. The reproducible setup patch leaves those timing structures untouched
until DETAILED mode while preserving cache-state updates.

Full web-Google, 2MB/16-way LLC, normal warming, and a 100K detailed ROI:

| Policy | Status | Context loaded | L3 accesses | L3 misses |
|---|---|---:|---:|---:|
| LRU | pass | n/a | 39,102 | 26,146 |
| K2 | pass | 1 | 37,881 | 27,486 |

These capped rows prove warm full-graph ROI entry only; they are not speedup or
equal-instruction evidence.

The post-fix `ecg_sniper_realgraph_600m` web-Google PR K2 cell also passes with
normal warming and valid context. It reaches semantic completion before the
cap at 179,432,203 reported instructions, with 7,892,046 L3 accesses,
2,153,699 misses, and miss rate 0.2729. The row remains
`timing_valid_for_speedup=0`; do not compare its time directly with a baseline
that reaches the 600M cap.

### Sniper P-OPT host-emulation acceleration

Sniper now computes each P-OPT candidate distance once per eviction and replaces
repeated RRIP aging scans with an equivalent delta update. The optimized path is
the default; `SNIPER_POPT_FAST=0` retains the legacy implementation.

Fresh same-binary web-Google-n16 PR A/B rows are bit-identical in instructions,
simulated ticks, L3 accesses/misses/rate, reserved ways, and matrix-stream
charge. Profiling reduces `findNextRef` calls by 7.96x and P-OPT victim-selection
host time by 2.48x. On the previously impractical cit-Patents cells, optimized
BC and SSSP complete in 191 seconds and 130 seconds, respectively; the legacy
versions remained unfinished after more than 11.5 hours.

This is a simulator-throughput optimization only. It does not alter P-OPT
victims, target timing, LLC capacity, or traffic accounting.

gem5 does not contain the legacy Sniper pathology. Its P-OPT adapter computes
`findNextRef` once per candidate, stores the result in `wayDists`, and performs
RRIP tie aging without further matrix consultations. Across the 15 sampled
gem5 graph/kernel cells, P-OPT host wall time is 1.008x LRU in geomean and
1.029x at worst. gem5's long wall time is therefore the detailed CPU/memory
simulation cost, not repeated P-OPT consultation.

### Preliminary STRIDE8 sensitivity (synthetic diagnostic)

K2 change from no prefetch to matched STRIDE8 on `kron_s15_k4`:

| Kernel | cache_sim demand | cache_sim traffic | gem5 demand | gem5 DRAM | Sniper demand | Sniper LLC-read traffic |
|---|---:|---:|---:|---:|---:|---:|
| PR | -89.8% | +0.0% | -82.2% | +0.2% | n/a | +9,195.1% |
| BFS | -79.2% | +81.8% | -29.2% | +41.1% | n/a | +59,482.8% |
| SSSP | -66.7% | +255.1% | -31.0% | +37.8% | n/a | +18,797.5% |
| BC | -27.0% | +94.9% | -16.5% | +15.6% | n/a | +17,512.9% |
| CC | -78.5% | +122.0% | -52.9% | +1.6% | n/a | +12,512.4% |

cache_sim and gem5 confirm that predictable record access can reduce demand
misses, although only cache_sim PR is traffic-neutral and BFS/SSSP remain
bandwidth-heavy. Sniper does not expose a demand/prefetch NUCA miss split; its
K2 total LLC read misses increase by 93x--596x, and every policy overprefetches
under the current generic simple prefetcher. Therefore the matched STRIDE8
profile rejects that Sniper prefetch configuration as a cross-simulator paper
path. No Sniper demand-miss reduction or speedup is inferred from these rows.

Aggregate: `results/ecg_experiments/paper_pipeline/`
`ecg_preliminary_5alg_final_20260714/aggregate/`
`preliminary_5alg_stride_sensitivity.csv`.

## Detailed-simulator mechanism cells

| Simulator | Cell | K2 time | K2+SS time | Speedup | K2 misses | K2+SS misses |
|---|---|---:|---:|---:|---:|---:|
| gem5 | kron_s16_k4 | 30.476B ticks | 26.962B ticks | **13.03%** | 39,333 | 16,425 |
| Sniper | kron_s16_k16 | 46.952T ticks | 46.647T ticks | **0.65%** | 9,889,214 | 9,859,131 |

The Sniper pair executes the same 118,517,996 instructions. These are mechanism
cells, not policy-ranking evidence.

Canonical reproduction profiles:

- `gem5_streamshield_mechanism`
- `sniper_streamshield_mechanism`

## Pending headline row

The manifest profile `streamshield_sniper_realgraph` produces the complete
web-Google Sniper matrix. Results are added here only after every required
policy finishes the same complete PageRank iteration.
