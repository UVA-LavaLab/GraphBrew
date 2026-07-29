# Results

> **HOW TO READ THIS FILE.** It is an append-only experimental record, so it
> contains withdrawn and superseded results on purpose. Nothing is deleted;
> mistakes are marked and kept. Before quoting anything:
>
> | marker in heading | status |
> |---|---|
> | `WITHDRAWN` / `RETRACTION` | not a result. Retained for audit. Do not cite. |
> | `CORRECTED` / `CORRECTION` | supersedes an earlier section; read this one |
> | `superseded` | replaced by a later section |
> | `FROZEN` | evidence-archived, tied to a commit |
> | anything else | live, but read the Scope line at the end of the section |
>
> **Current live position (2026-07-28), in reading order:**
> 1. `ATTRIBUTION` below is a **BFS-specific attribution cell**, not a universal
>    explanation. It proves that cell's 62% loss was 8-byte record width rather
>    than victim selection. Later gem5 PageRank results include a 4-byte
>    cit-Patents transport loss, so “every K2 loss is width” is withdrawn.
> 2. `SCALING: the record fits 4 bytes at Twitter scale, at 2-bit epochs`
> 3. `Fast signal, corrected` -- K2 wins PageRank, GRASP wins the aggregate.
> 4. `CORRECTED: two earlier fast-signal sections were wrong, and how`
> 5. `FROZEN: public P-OPT artifact direction` -- an external baseline
>    validation only; it does not rank K2.
>
> **No K2 headline performance claim in this file is admissible yet.**
> `claim_gate.json` is authoritative. The only allowed performance-direction
> claim is the narrowly scoped external P-OPT-over-DRRIP artifact result below;
> it does not validate GraphBrew's P-OPT model or compare K2.
>
> **Standing caution.** This result set moved substantially five times under
> configuration changes alone (metadata accounting, mechanism availability,
> prefetcher model, record width, private cache sizing) with no change to any
> policy. Configuration sensitivity has exceeded the effects being measured, so
> any number quoted without its full configuration receipt is meaningless.

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
shared victim specification. gem5 executes the real RISC-V fused indexed K2-I
load; Sniper validates its idealized packed-record sideband model immediately
before the governed access. Tiny O3 PR and weighted SSSP runs each deliver 8/8 traced K2 request
extensions to the correct property line. This is mechanism/spec evidence, not a
frozen real-graph performance ranking.

The v2 computed-address K2-M load is now separately implemented. Real decoder
tests pass U32/S32/U64/F32 and compact-weighted forms. Tiny O3 PR and compact
SSSP each deliver 8/8 traced masks to post-filter consumer accepts on the exact
property Requests. A clean five-kernel cache_sim+gem5 gate passes with zero K2
distance mismatches; BFS/SSSP/BC have decisive epoch victims and BC proves both
4-byte depth and 8-byte path-count delivery. This is ISA/mechanism evidence,
not a K2-M timing result. The epoch/context CSRs were implemented afterward;
these frozen rows have not been relabeled.

The transport-matched Sniper model also passes its five-kernel mechanism gate.
LRU and K2-M execute identical dynamic instruction counts in every cell:

| Kernel | K2-M speedup vs LRU | L3 miss ratio | Instruction ratio |
|---|---:|---:|---:|
| PR | 1.018x | 0.959x | 1.000x |
| BFS | 1.034x | 0.968x | 1.000x |
| SSSP | 1.009x | 0.972x | 1.000x |
| BC | 0.991x | 0.895x | 1.000x |
| CC | 0.977x | 0.990x | 1.000x |
| Geomean | **1.006x** | **0.957x** | **1.000x** |

The gate uses the same 8-byte record loops, exact semantic-result equality,
uncapped ROIs, hashed binaries/output markers, and a 0.25% instruction
tolerance. PR and compact SSSP cold proofs each report one validated receipt and
zero bad records. This establishes instruction parity and mechanism direction,
not real-graph K2-M performance. Those frozen rows predate both gem5's epoch
CSR and Sniper's explicit governed-load marker, so both mechanisms were modeled
in that dataset.

The fresh post-binding semantic gate executes 4,096 static edge visits for
every policy and passes 25/25 rows:

| Kernel | K2-M diagnostic speedup vs LRU | K2-M L3 miss ratio | Instruction ratio |
|---|---:|---:|---:|
| PR | 1.041x | 0.767x | 1.000x |
| BFS | 0.987x | 1.160x | 1.000x |
| SSSP | 0.998x | 1.070x | 1.000x |
| BC | 0.929x | 0.862x | 1.000x |
| CC | 0.926x | 0.984x | 1.000x |
| Geomean | **0.975x** | **0.958x** | **1.000x** |

All rows report exact governed-load binding, epoch/context association, matched
transport, identical semantic output, and no instruction cap. K2-M improves
misses on PR/BC/CC and regresses BFS/SSSP. Only PR improves diagnostic time;
BC/CC remain slower. These are truncated synthetic-prefix rows with
`timing_valid_for_speedup=0`, not headline or full-graph timing.

The committed fresh three-simulator computed-address K2-M conformance gate
passes all 15 PR/BFS/SSSP/BC/CC x cache_sim/gem5/Sniper cells. Every eviction
obeys the shared specification, every distance mismatch count is zero, gem5
executes K2-M in all kernels, Sniper exact binding passes, and BC dual-load
coverage passes. This demonstrates one mechanism integrated into three
dissimilar substrates; it does not claim equal cache statistics or victim
sequences across simulators.

### Sniper host-cost audit
The abandoned full cit-Patents K2-I-like run was progressing, but its 14-hour
wall time did not come from the indexed K2 metadata lookup or victim selector.
Internal timers on an SSSP mechanism cell measure:

- fused K2 lookup: 7,865 calls, 1.42 ms total, 181 ns/call;
- all ECG replacement calls: 136 ms total;
- all ECG hit updates: 68 ms total;
- all ECG insertion preparation: 33 ms total;
- complete host run: 129 s.

A 256-entry per-core lookup memo produced only 41 hits in 698,800 calls
(0.0059%), preserved target statistics exactly, and regressed host wall time
from 1,724 s to 1,852 s. It was rejected. The dominant cost is SIFT
trace/detailed simulation; the old K2-I-like guest executed more instructions
and memory references and ran concurrently with other CPU-intensive jobs.
Controlled live and explicit-SIFT frontend cells took 236.469 s and 237.510 s
with bit-identical target statistics. An isolated `-O3` Sniper build improved a
70.17 s cell to only 69.17 s, within run noise, and was not adopted. Persistent
trace replay also does not preserve the warmed-ROI contract: replaying only the
ROI starts cold, while replaying the full cache-warm prefix costs about as much
as the direct run.

The mask-mode runner audit then found a separate pre-ROI confound:
`SNIPER_REQUIRE_POPT_MATRIX=1` was being forced for every transport-matched
policy. K2-M and its matched LRU rows therefore constructed and exported the
P-OPT matrix even though neither policy consulted it. The runner now restricts
that structure to P-OPT and fails closed if a non-P-OPT mask row reports a
loaded rereference matrix. A focused compact-SSSP smoke reports
`required=0,reref=0` for K2-M with no matrix export and `required=1,reref=1`
for P-OPT. The reboot-interrupted 100M calibration predates this correction and
is discarded.

Future full-graph runs therefore use the exact-instruction/memory-reference
K2-M model rather than a speculative host memoization.

### Architectural epoch/context channel
gem5 now exposes user-level `ecg.cur_epoch` (`0x800`) and `ecg.context`
(`0x801`) CSRs. K2-M/K2-I and the request-bound single-epoch load snapshot the
current epoch, context ID, and O3 program-order sequence onto the governed
Request. Resident K2 metadata stores its context ID, and victim selection uses
the allocating request's current epoch rather than the global vertex-magic
clock.

Classic-cache MSHRs keep the greatest same-hart/same-context sequence and set a
sticky conflict for cross-hart, cross-context, invalid-context, or mixed
ordinary/K2 targets; conflicted fills remain unstamped. A standalone mutation
test covers those transitions and replay idempotence. The current benchmark
harness allocates monotonic nonzero IDs, clears the CSRs at context end, and
fails closed instead of reusing IDs. Integrated OoO stress, any optional
drain/invalidation protocol for intentional reuse, and exact Sniper request
binding remain open, so this milestone adds no performance result.

### Equal-area runner gate
The runner and paper manifest now provide separate `ecg_equal_area_15` and
`ecg_equal_area_14` full-graph cache_sim profiles. Conventional baselines retain
the 16-way geometry; only Schedule-2 K2 policies receive the requested override.
Each row records baseline and effective size/associativity, area mode, requested
K2 ways, and the 49-bit line-metadata premise.

### Full-graph capacity sweep (16/15/14 ways) — FROZEN
Three complete cache_sim runs at paper scale: web-Google, soc-pokec and
cit-Patents x PR/BFS/SSSP/BC/CC x nine policies, 8 MiB/16-way LLC, 135 ok rows
each. Metric is `l3_misses_with_overhead` (K2 record traffic and P-OPT matrix
streaming are charged), reported as a geomean ratio to LRU over all 15 cells.
Lower is better.

| Policy | 16-way equal-capacity | 15-way equal-area | 14-way equal-area |
|---|---:|---:|---:|
| GRASP | **0.836** | 0.836 | 0.836 |
| HAWKEYE_PROXY | 0.898 | 0.898 | 0.898 |
| SRRIP | 0.929 | 0.929 | 0.929 |
| K2-online+StreamShield | 0.926 | 0.955 | 0.986 |
| K2+StreamShield | 0.935 | 0.964 | 0.995 |
| K2-online | 0.956 | 0.984 | 1.016 |
| K2 | 0.960 | 0.989 | 1.020 |
| charged P-OPT | 0.969 | 0.969 | 0.969 |

Conclusions, stated against interest:

- **K2 does not beat GRASP on this metric.** Even in its most favourable
  16-way equal-capacity configuration, the best K2 variant (0.926) trails GRASP
  (0.836), and a K2 variant is the best policy in only 3 of 15 cells. Best-K2
  beats GRASP in 4/15 cells. Any claim that K2 outperforms all baselines,
  including GRASP, is not supported by this evidence.
- **K2 does beat charged P-OPT**, the closest prior art, in 9/15 cells and on
  the geomean (0.960 versus 0.969 at 16 ways).
- **SSSP is K2's genuine win**: K2+StreamShield is the best policy on all three
  real graphs (cit-Patents 0.700 versus GRASP 0.745; soc-pokec 0.719 versus
  0.734; web-Google 0.778 versus 0.848).
- **BFS is K2's clear weakness**, landing above LRU on every graph
  (1.27/1.40/1.26).
- StreamShield is consistently worth ~2-3 points (0.960 to 0.935 at 16 ways).
- The sweep quantifies the equal-area cost at roughly **3 points per way**
  (0.960 to 0.989 to 1.020). This is why 14/15-way rows are sensitivities and
  never the headline configuration.

Scope: one metric (overhead-aware LLC misses) in cache_sim. It does not settle
timing, where K2's sequential record stream and StreamShield's traffic
reduction may behave differently; that remains the pending Sniper matrix.

### RETRACTION: three K2 conclusions withdrawn after review
An adversarial review of the 2026-07-25 analysis chain invalidated three
conclusions recorded above. They are withdrawn here rather than edited away.

**1. The gem5 "instruction-bound, not memory-bound" result is not admissible.**
It was computed from historical rows labelled `ecg.load2`, which
`METHODOLOGY.md` already states "predate this correction and are not
reinterpreted without rerunning". Using them violated our own rule. The result
also used *sampled* graphs at reduced cache sizes while the surrounding
functional study used *full* graphs at 8 MiB, so it could not have validated
full-graph bandwidth behaviour even if the rows were current.

**2. "IPC is higher" was not independent evidence.** IPC ratio is
algebraically instructions/time: 1.422 / 1.228 = 1.158, versus the 1.157
reported. "IPC rises" and "instruction-normalised time is 0.864" are one
observation stated twice, not two confirmations. Dividing time by the
instruction ratio also assumes the removed instructions consume average CPI and
that deleting them preserves prefetch distance, MLP and dependences -- while
those same instructions construct the records that create K2's advantage. It is
a counterfactual, not a projection.

**3. The STRIDE8 result rests on an idealised prefetcher.** cache_sim's stream
prefetcher classifies with `graph_ctx_->findRegion(address)`, i.e. it knows
exactly which addresses are structural and never mispredicts property; it
issues `degree` fills unconditionally with no MSHR, queue, lateness or
bandwidth backpressure. Its own comment states the intent is that wider 8B
records are not "unfairly penalised". A result produced by a mechanism built to
protect the hypothesis cannot be used to confirm it.

**What the STRIDE8 miss metric concealed.** Demand misses fall, but the traffic
does not go away:

| policy | median demand misses | median total traffic | median prefetch fills |
|---|---:|---:|---:|
| GRASP | 0.528 | **0.879** | 1.042 |
| K2 | 0.513 | **1.368** | 1.683 |
| K2+StreamShield | 0.469 | **1.360** | 1.671 |

K2 wins on demand misses only by converting them into ~1.36x total memory
traffic and ~1.67x prefetch fills, which the miss metric does not price.

**Process failure.** The metric changed five times -- total misses, then
property-only, then a weighted cost, then instruction-normalised time, then
prefetched misses -- and each change followed a result unfavourable to K2. Even
where each step was individually defensible, the sequence is metric selection.
Primary metrics must be frozen before the next run.

**Surviving claim.** K2 substantially reduces governed irregular property
misses by consuming an additional sequential per-edge stream, and StreamShield
reduces that stream's LLC pollution. Total memory traffic rises materially. No
end-to-end performance lead has been demonstrated.

### KNOWN FLAW: the P-OPT matrix stream is charged but not prefetched
Under STRIDE8 the charged comparison reports charged P-OPT at a median 0.926
and, on web-Google PageRank, at **2.684** -- nearly three times worse than LRU.
That is not credible for a near-oracle policy, and it is our accounting at
fault, not P-OPT.

On web-Google PageRank, P-OPT takes **1** actual LLC miss against LRU's 85,351:
the oracle is essentially perfect. We then add
`popt_matrix_stream_cache_lines = 229,108` as a flat penalty, producing 229,109
"misses". The asymmetry is:

- K2's edge records are **simulated memory accesses**, so the stride prefetcher
  covers them and their misses collapse.
- P-OPT's rereference-matrix stream is an **analytic post-hoc charge**, so no
  prefetcher can ever cover it.

Both are sequential streams. Letting the prefetcher hide one but not the other
systematically favours K2. Effect on P-OPT under STRIDE8:

| | median | web-Google PR |
|---|---:|---:|
| raw (oracle quality) | 0.491 | ~0.000 |
| charged (flat matrix penalty) | 0.926 | 2.684 |

Read raw, P-OPT is competitive with the best K2 variant (0.491 versus 0.469),
and on PageRank -- the kernel the P-OPT paper actually claims -- it behaves as
published. Any K2-versus-P-OPT claim taken from the charged STRIDE8 column is
therefore overstated and must not be used.

The fix is to route the matrix stream through the simulated hierarchy so the
prefetcher can cover it exactly as it covers K2 records, or failing that to
apply the same prefetch-coverage discount to the flat charge. Until then, quote
P-OPT raw alongside charged, and treat the no-prefetch charged comparison
(where neither stream is prefetched, so the accounting is symmetric) as the
defensible one.

**RESOLVED: the matrix stream is now simulated, and it quantifies the flaw.**
cache_sim issues the rereference-matrix column stream as real non-temporal
accesses, tracking the paper's two resident columns explicitly: an epoch whose
column is still resident costs nothing, and any other epoch streams a fresh
column and evicts the older one. The columns do not allocate in the modelled
cache because the reserved ways that hold them are already deducted from the
geometry; allocating again would charge P-OPT twice for the same capacity.
Capacity and bandwidth are orthogonal costs, and P-OPT pays both.

Measured, web-Google PageRank, `-i 2`, 2 MiB 16-way, charged P-OPT:

| configuration | demand misses | prefetch fills | total traffic |
|---|---:|---:|---:|
| no prefetch, stream not simulated | 2,724,144 | 0 | 2,724,144 |
| no prefetch, stream simulated | 3,182,212 | 0 | 3,182,212 |
| STRIDE8, stream not simulated | 1,645,819 | 1,089,041 | 2,734,860 |
| STRIDE8, stream simulated | 1,648,232 | 1,551,066 | 3,199,298 |

The simulated stream issues 512 columns / 458,240 lines, exactly 256 columns
per PageRank iteration. **The flat charge was wrong in two opposite directions
at once:**

1. **It overcharged demand misses under a prefetcher, by roughly 95x.** The
   whole 458,240-line stream costs only 2,413 additional demand misses
   (1,648,232 against 1,645,819) because the prefetcher covers it; it appears
   as +462,025 prefetch fills instead. The flat charge added 229,108 demand
   misses. That, and not P-OPT, is the entire explanation of the 2.684.
2. **It undercharged traffic for every multi-iteration kernel.** The analytic
   count is `num_epochs * column_bytes`, a single sweep, so it charged 229,108
   lines while PageRank at `-i 2` truly streams 458,240. A first version of the
   simulated model reproduced this bug by only charging forward epoch progress,
   which made the stream cost identical at `-i 1`, `-i 2` and `-i 4`; the
   residency model above fixes it and the counts now scale exactly.

Without a prefetcher the simulated stream costs 458,068 demand misses against
its 458,240 lines, i.e. 99.96% of the stream misses, as a cold sequential
stream should.

Note which metric survives. Traffic tracks the stream faithfully in the
simulated model, and a prefetcher relocates that traffic rather than removing
it (+464,438 traffic while demand misses barely move). The bytes were always
real; the demand-miss column was fictional. This is a direct vindication of the
frozen primary metrics: the metric that misled is the one now barred from
carrying a performance argument under a prefetcher.

The runner fails closed on the invalid combinations: a charged P-OPT policy
with an active prefetcher and the analytic charge is rejected, and requesting
the simulated stream on a backend that does not implement it is an error rather
than a silent fallback to the analytic charge.

**Scope, and why this does not yet license a K2-versus-P-OPT claim.** Two
limitations are explicit. First, the stream is modelled on the ordinary demand
path, so it is covered by cache_sim's stream prefetcher; published P-OPT uses a
dedicated streaming engine that writes into the reserved ways and is evaluated
with conventional prefetching disabled. The "prefetcher covers it" result is
therefore a statement about our accounting, not a claim about P-OPT hardware,
and cache_sim's prefetcher is in any case ineligible for performance claims
under the frozen metrics. Second, `total_memory_traffic` here is demand plus
prefetch fills and excludes LLC writebacks. The symmetric-accounting gate stays
open until the stream is modelled as an engine-side transfer.

### CORRECTION: StreamShield was a mechanism only K2 was allowed to use
K2's structural bypass (StreamShield) declines to allocate its one-touch
per-edge records in the LLC. The same argument applies to any policy's CSR edge
stream: it is sequential and read-once, so allocating it evicts reusable
property lines. The runner offered the option to K2 alone, because
`ecg_transport_for` returned a no-bypass transport for every policy that was
not K2. Every previous K2-versus-baseline comparison therefore mixed "K2
replaces better" with "K2 is the only policy allowed to bypass".

The bypass is now available to every policy (`--structural-bypass all`), and it
matters far more to the baselines than StreamShield does to K2.

web-Google PageRank, `-i 2`, 2 MiB 16-way, no prefetcher, every metadata stream
charged (P-OPT's matrix stream simulated, K2's 4-byte record stream charged).
Demand misses equal traffic here because no prefetcher is active:

| policy | no bypass | with bypass | bypass gain |
|---|---:|---:|---:|
| GRASP | 3,121,133 | **2,964,622** | -5.0% |
| charged P-OPT | 3,182,235 | 3,128,955 | -1.7% |
| LRU | 4,356,759 | 3,486,848 | **-20.0%** |
| K2 | 3,608,879 | -- | -- |
| K2+StreamShield | -- | 3,522,202 | -2.4% vs K2 |

K2 and K2+StreamShield are separate policies and are reported as separate rows;
StreamShield *is* K2's bypass, so the two are not two settings of one row.

Two conclusions, and the second is the consequential one.

1. **The bypass helps the weakest policy most.** LRU gains 20.0% from it,
   against GRASP's 5.0% and P-OPT's 1.7%. Granting it only to K2 therefore
   flattered K2 most where the baselines were weakest.
2. **With the mechanism equalised, K2 loses to plain LRU.** K2 with
   StreamShield (3,522,202) is worse than LRU with the same bypass
   (3,486,848), and both are well behind GRASP with bypass (2,964,622). K2
   ranks third of four without any bypass and **last of four** once the
   baselines are allowed the bypass too.

Note what the two bypasses act on. StreamShield bypasses K2's per-edge *record*
stream; `--structural-bypass all` bypasses the *CSR edge* stream that the
baselines read. They are the same idea applied to each policy's own one-touch
structural stream, which is the point, but they are not the same bytes, and
`STRUCTURAL_BYPASS` deliberately does not alter charged K2 because StreamShield
already is K2's bypass. K2 is not given two bypasses.

Configuration, so the numbers can be reproduced exactly: this is a direct
environment probe of `bench/bin_sim/pr`, not the pinned `ECG:K2` policy. It
uses the packed 4-byte record (`[ECG RECORD] N=916428 epoch_bits=10 ->
record_bytes=4`, i.e. 20 id + 10 epoch + 2 tier bits) with
`ECG_VARIANT=epoch_only`. The canonical pinned `ECG:K2` policy instead uses the
8-byte Schedule-2 record and the adaptive variant, so it will not reproduce
these exact figures.

The bypass is not a free win either. On a small synthetic cell it *increases*
P-OPT traffic, because declining to allocate a stream that still has reuse
costs more than it saves. Nor is the CSR stream one-touch in every kernel:
PageRank rereads every edge each iteration, BFS bottom-up can rescan adjacency,
and BC repeats traversal per source. Bypass is therefore a sensitivity to be
reported, not an entitlement that automatically makes a comparison fair.

Every row now records `structural_bypass`, so a matrix that granted the option
unevenly is visible rather than implicit.

Scope: one graph, one kernel, one cache size, no prefetcher. Under the frozen
metrics this is a traffic result on a single cell, not a headline; the frozen
cell set and geomean still govern any claim. It is reported because it changes
the direction of the K2-versus-baseline comparison, which is exactly the kind
of finding the previous accounting would have hidden.

### DECOMPOSITION: K2's replacement is good; the bypass is worth more
"K2+StreamShield loses to LRU+bypass" is an end-to-end traffic result, but on
its own it does not say whether K2's *replacement* is any good, because K2 also
carries a per-edge record stream that LRU does not. Holding transport fixed and
varying only the victim rule separates the two. `ECG_VARIANT=lru_only` runs the
identical K2 transport with recency selection, so it is K2 with its replacement
intelligence switched off.

Same cell as above (web-Google PageRank, `-i 2`, 2 MiB 16-way, no prefetcher,
packed 4-byte record). Total traffic:

| | configuration | traffic |
|---|---|---:|
| A | LRU baseline | 4,356,828 |
| B | K2, charged record, `epoch_only` | 3,607,455 |
| C | K2, charged record, `lru_only` | 4,357,274 |
| D | K2, free metadata, `epoch_only` | 3,606,221 |
| E | K2, free metadata, `lru_only` | 4,356,828 |

E equals A to the line, which is the harness sanity check: K2 transport with
recency selection and free metadata is exactly plain LRU.

| effect | difference | |
|---|---:|---|
| transport cost of K2's record stream | C - A = **+446** | 0.01% |
| K2 replacement gain, charged | B - C = **-749,819** | -17.2% |
| K2 replacement gain, free metadata | D - E = -750,607 | -17.2% |
| net K2 versus LRU | B - A = -749,373 | -17.2% |

Three things follow, and they revise the previous section rather than
contradict it.

1. **K2's packed transport is essentially free**, +446 lines out of 4.36M. The
   4-byte packed record *replaces* the CSR edge read rather than adding to it,
   so at this graph size K2 streams the same bytes per edge as any baseline.
   The charged-versus-free columns differ by under 0.04%, so the record cost is
   not what holds K2 back.
2. **K2's replacement is genuinely good**: -17.2% traffic against identical
   transport with recency selection. That is a real algorithmic result and it
   is not an artifact of accounting.
3. **The structural bypass is worth more than K2's entire replacement
   advantage.** The bypass gives LRU -20.0%; K2's epoch replacement gives
   -17.2%. Both are removing the *same* pollution -- the structural stream
   displacing reusable property lines -- one mechanically and one
   algorithmically. That is why they barely stack: adding StreamShield on top
   of K2's replacement is worth only a further -2.4%, because the replacement
   has already captured most of what the bypass captures.

So the honest statement is not "K2's replacement is bad". It is that on this
cell K2 solves, slightly less well, a problem that a one-line bypass also
solves, while GRASP solves it better still (-28.4% versus LRU, -31.9% with the
bypass). A policy whose advantage is substitutable by a bypass has to argue on
cost, generality or the cases where the bypass is unavailable or harmful -- and
the bypass is indeed harmful on at least one cell measured above.

Scope: one graph, one kernel, one cache size, no prefetcher; single-cell
traffic under the frozen metrics, not a headline.

### CORRECTION: the stream prefetcher no longer classifies by oracle
The prefetcher that produced the withdrawn STRIDE8 lead decided what to
prefetch by asking `graph_ctx_->findRegion(address)` whether an address was
property data, and refusing if so. It therefore never mispredicted the one
distinction the experiment turned on, and it issued unconditionally with no
MSHR, queue, lateness or bandwidth backpressure.

cache_sim now defaults to an address-only stream detector
(`CACHE_STREAM_PREFETCH_MODEL=stride`). It sees addresses and nothing else: a
stream must be confirmed by two consecutive ascending line accesses within a
4 KiB region, any non-sequential step breaks confirmation, issue stops at the
region boundary, and it is bounded by a finite in-flight budget. It trains on
regular property accesses and wastes fills on irregular ones, which is the
mistake the oracle could not make. The oracle survives as a labelled upper
bound.

web-Google PageRank, `-i 2`, 2 MiB 16-way, degree 8, charged P-OPT with its
matrix stream simulated:

| policy / model | prefetches issued | demand misses | prefetch fills | total traffic |
|---|---:|---:|---:|---:|
| P-OPT, oracle | 141,971,552 | 1,646,658 | 1,550,635 | 3,197,293 |
| P-OPT, stride | 7,652,021 | 1,730,286 | 1,462,349 | 3,192,635 |
| LRU, oracle | 138,305,632 | 3,276,993 | 1,080,528 | 4,357,521 |
| LRU, stride | 4,434,985 | 3,342,925 | 1,015,250 | 4,358,175 |

The honest result is narrower than a first version of this section claimed, and
the difference is instructive.

1. **The oracle's cost-free issue rate is the real defect.** It issues 19-31x
   more prefetch requests than the honest detector (141.9M against 7.7M for
   P-OPT) with no modelled bandwidth or queue cost. A component that can
   request 30x the traffic for free makes any metadata-streaming policy look
   cheap.
2. **Its coverage advantage is modest.** Demand misses rise only 5.1% for
   P-OPT and 2.0% for LRU when the prefetcher has to detect the stream rather
   than be told where it is. Both metadata streams really are sequential, so an
   honest detector does find them.
3. **Traffic is again the stable metric**, differing by 0.1-0.2% across models
   while the demand column moves several percent. Note this is a weak
   confirmation rather than an independent one: a prefetcher converts demand
   misses into fills, so the two partly cancel by construction.

**A first version of this section reported +32.8% and was wrong.** That figure
came from a bug: `accessNonTemporal()`, the path carrying K2's per-edge records
*and* P-OPT's simulated matrix columns, still issued prefetches unconditionally
regardless of the selected model. The "honest prefetcher" therefore never
reached the metadata streams the comparison turns on, and the 32.8% measured
only the loss of oracle coverage on *ordinary* accesses. Both paths now use one
detector.

Rows record `stream_prefetch_model`, `stream_prefetch_issued`,
`stream_prefetch_throttled` and `stream_prefetch_untrained`, reset at the ROI
boundary with every other counter.

**This does not make cache_sim prefetch results admissible for performance
claims.** Fills are still synchronous and free: a prefetch completes before the
demand that triggered it, and there is no lateness, no bandwidth contention and
no real MSHR occupancy model. What the detector establishes is that the
semantic address oracle is gone from every path and that coverage no longer
depends on knowing which addresses are property. Timing, lateness and bandwidth
must still come from gem5.

### CORRECTED: two earlier fast-signal sections were wrong, and how
Two sections previously stood here claiming that the pinned configuration
forced an 8-byte record, that K2's second future epoch does not pay for itself,
and that the single-epoch K1 record should be promoted over K2. Adversarial
review and follow-up measurement invalidated all three. They are replaced
rather than edited, and the errors are recorded because each was a different
kind of mistake.

**Error 1: the 8-byte record was our shortcut, not K2's cost.**
`GraphSimEcgRecordBytes` contained `if (schedule_k == 2) return 8;`, which
skipped the bit budget entirely. On a 65,536-vertex graph K2's two-epoch record
needs 16 id + 2x5 epoch + 2 tier = 28 bits and fits in 4 bytes. Charging it 8
doubled K2's modelled transport, and every K2-versus-K1 comparison was
therefore a comparison of record widths rather than of policies. With
`ECG_RECORD_VARIABLE_WIDTH=1`, web-Google-n16 PageRank moves from 1.171 to
**0.660**, and K2 overtakes K1. The claim that the second epoch does not pay
for itself was an artifact of our own accounting.

**Error 2: "the specs hardcode 65535 epochs" was false.** 65535 is the
runner's global `--ecg-epochs` default, not a value pinned in `ECG:K1` or
`ECG:K2`.

**Error 3: the harness ran a toy private hierarchy.** The signal took
`roi_matrix`'s bare defaults, which are **1 kB L1 and 2 kB L2**, sized for
smoke tests. With a hierarchy that small almost every access reaches the LLC:
web-Google-n16 PageRank LRU traffic was 297,710 against 128,970 with a
realistic 32 kB / 256 kB hierarchy, a 2.3x inflation that changes what the LLC
policy is being asked to do. Every number in the withdrawn sections came from
that configuration, including the headline "K1+StreamShield beats GRASP at
0.613 and 0.688", which additionally used a plain-LRU baseline while the matrix
used LRU with the structural bypass. The private cache sizes are now explicit
in the harness.

### Fast signal, corrected: K2 wins PageRank, GRASP wins overall
Sampled graphs, cache_sim traffic versus LRU, no prefetcher, 32 kB / 256 kB
private caches, 32 epochs, variable-width record (K2 packs to 4 bytes),
structural bypass available to every policy, P-OPT matrix stream simulated:

| cell | GRASP | P-OPT | K2+SS | K1+SS |
|---|---:|---:|---:|---:|
| web-Google-n16 / pr | 1.116 | 1.505 | **0.878** | 0.896 |
| soc-pokec-n16 / pr | 1.130 | 1.304 | **0.920** | 0.935 |
| web-Google-n16 / bfs | 0.904 | 0.948 | 1.442 | 0.973 |
| soc-pokec-n16 / bfs | 0.917 | 0.967 | 1.393 | 0.993 |
| web-Google-n16 / cc | 0.873 | 0.929 | 1.217 | 0.889 |
| soc-pokec-n16 / cc | 0.865 | 0.914 | 1.089 | 0.897 |
| web-Google-n16 / sssp | 0.857 | 0.862 | 0.865 | 1.367 |
| soc-pokec-n16 / sssp | 0.872 | 0.860 | 0.838 | 1.324 |
| web-Google-n16 / bc | 0.905 | 0.911 | 1.000 | 0.955 |
| soc-pokec-n16 / bc | 0.887 | 0.915 | 1.015 | 0.948 |
| cit-Patents-n18 / cc | 0.822 | 0.797 | 0.956 | 0.934 |
| cit-Patents-n18 / pr | 0.989 | 1.573 | 2.013 | 2.022 |

Geomean over all 12 cells: GRASP 0.923, P-OPT 1.014, K1+SS 1.059, K2+SS 1.097.
Excluding the degree-1 cit-Patents sample: GRASP 0.928, P-OPT 0.994, K1+SS
1.006, K2+SS 1.047.

**The specific result worth pursuing is PageRank.** On both real sampled
graphs K2+StreamShield is the best policy *and GRASP is worse than LRU*
(1.116 and 1.130). A degree-based static hint mispredicts on these PageRank
cells while a per-edge next-reference epoch does not. That is a narrow,
falsifiable, mechanism-level claim, and it is the one place where K2's design
premise is visibly doing work.

**Aggregate leadership belongs to GRASP**, which is consistent across the
traversal kernels where K2 is weak. K2 is poor on BFS (1.39-1.44) and mixed on
CC, and K1 is the better of the two variants on traversal while K2 is better on
PageRank and SSSP. No general "K2 wins" claim is supported.

Scope: sampled graphs, one metric, no prefetcher, no timing. Three cells were
excluded as carrying no policy signal. Direction check only.

**A standing caution.** This result set has now moved substantially three
times, under (a) metadata-stream accounting, (b) record width, and (c) private
cache sizing, without any change to the policies themselves. Configuration
sensitivity is larger than the differences being measured, so the frozen cell
set must pin the private hierarchy, the record width rule and the epoch count
explicitly before any of this is quoted.

### ATTRIBUTION (scoped): the measured BFS loss is record width, not replacement
Five result reversals with no policy change prompted an attribution experiment
instead of another ranking. Holding the workload and the record fixed and
varying only the victim rule, then varying only the record, separates the two.

BFS, web-Google-n16, 128 kB LLC, 32 kB/256 kB private, StreamShield on:

| configuration | traffic | vs LRU | vs native GRASP |
|---|---:|---:|---:|
| native LRU | 147,425 | 1.000 | |
| native GRASP | 123,852 | 0.840 | 1.000 |
| K2, arm `grasp_only`, record charged | 200,613 | 1.361 | 1.620 |
| K2, arm `grasp_only`, **record free** | 123,867 | 0.840 | **1.000** |
| K2, arm `grasp_only`, **forced 4 B** | 123,408 | 0.837 | **0.996** |
| K2, arm `grasp_only`, forced 8 B | 200,698 | 1.361 | 1.620 |
| K2 online (set dueling) | 201,408 | 1.366 | 1.626 |

K2 running GRASP's own victim rule with a free record reproduces native GRASP
to within 15 lines out of 123,852, and with a real 4-byte record it is
marginally *better* than native GRASP. The entire 62% BFS deficit is the
record stream widening from 4 bytes to 8. **No part of it is victim
selection.**

This also resolves the set-dueling anomaly. Dueling was suspected of failing
because it did not rescue BFS. It did not fail: online (201,408) tracks the
best static arm `grasp_only` (200,613) closely, so it selected correctly. It
could not help because the loss was never in the arm it selects among. A
selector over victim rules cannot recover a transport cost.

**Scoped consequence.** For this BFS cell, viability reduces to keeping the
record at 4 bytes. This does not generalize to every graph/kernel: later gem5
PageRank evidence shows cit-Patents incurs a 13.6% transport penalty even with a
4-byte record, so topology/layout remains independently load-bearing.

### SCALING: the record fits 4 bytes at Twitter scale, at 2-bit epochs
The budget is `id_bits + epoch_bits * stamps + tier_bits <= 32`. K2 carries two
stamps, so epoch resolution costs double. PageRank on web-Google-n16, record
forced to 4 bytes, versus LRU 128,892 and GRASP 107,582:

| epoch bits | epochs | traffic | vs LRU | vs GRASP |
|---:|---:|---:|---:|---:|
| 2 | 4 | 93,758 | 0.727 | **0.872** |
| 3 | 8 | 89,112 | 0.691 | 0.828 |
| 4 | 16 | 86,543 | 0.671 | 0.804 |
| 5 | 32 | 84,900 | 0.659 | **0.789** |

K2 beats GRASP at every resolution including the cheapest. So epoch bits can be
traded for id bits while keeping the win:

| graph | vertices | id bits | K2 budget at 2-bit epochs | 4 B? |
|---|---:|---:|---:|---|
| web-Google | 916,428 | 20 | 20 + 4 + 2 = 26 | yes, 5-bit epochs also fit |
| soc-pokec | 1,632,803 | 21 | 21 + 4 + 2 = 27 | yes |
| cit-Patents | 3,774,768 | 22 | 22 + 4 + 2 = 28 | yes |
| Twitter-2010 | 41,652,230 | 26 | 26 + 4 + 2 = **32** | **yes, exactly** |
| Friendster | 65,608,366 | 26 | 26 + 4 + 2 = **32** | **yes, exactly** |
| 100M synthetic | 100,000,000 | 27 | 27 + 4 + 2 = 33 | only without tier bits |

Twitter and Friendster fit in exactly 32 bits at 2-bit epochs, where K2 still
returns 0.872 against GRASP. Past roughly 67M vertices the tier field must go,
and past ~134M the two-stamp record cannot be one edge wide on a 32-bit graph.

Two honest caveats. The 2-bit result is measured on a 65K-vertex sample, and an
epoch is `floor(current_vertex * epochs / num_vertices)`, so four epochs spans a
very different vertex count on Twitter than on the sample; the resolution
sweep must be repeated at scale before this is claimed. And the tier field has
only been removed from the *width* calculation, not from the replacement
mechanism, so "drop the tier bits" is not yet a validated configuration.

### S2: a narrow sidecar makes metadata cost independent of graph size

The packed record (S1) carries destination id + tier + stamps in one word and
SUBSTITUTES for the CSR edge, so it is free while it fits in 4 bytes and costs
100% once id_bits force it to 8. That ties metadata cost to |V| and caps the
4-byte record near 67M vertices.

A sidecar (S2) does not need the destination id, because the unmodified CSR edge
still carries it. K2-I would need the id in the operand to compute the property
address, but K2-M receives an already-computed address, so for K2-M a
stamps-only sidecar is sufficient. Its payload is
`stamps * epoch_bits + tier_bits`, which is **independent of graph size**. The
payload is bit-packed, so a 64-byte line holds `512 / payload_bits` entries and
the stream costs `ceil(payload_bits * E / 8)` bytes, not one byte per edge.

web-Google-n16 PageRank, 128 kB LLC, 32 kB/256 kB private, off-chip traffic in
both directions, no prefetcher:

| structure | traffic | vs LRU | vs GRASP | size limit |
|---|---:|---:|---:|---|
| LRU | 130,227 | 1.000 | 1.196 | -- |
| GRASP | 108,949 | 0.837 | 1.000 | -- |
| S1 packed, 4 B | 85,738 | **0.658** | **0.787** | ~67M vertices |
| S1 packed, 8 B | 151,407 | 1.163 | 1.390 | unbounded |
| S2 sidecar, 6 bits | 99,315 | **0.763** | **0.912** | **unbounded** |
| S2 sidecar, 8 bits | 103,332 | 0.793 | 0.948 | unbounded |
| S2 sidecar, 12 bits | 111,865 | 0.859 | 1.027 | unbounded |

Three readings.

1. **S1 at 4 bytes remains the best structure where it fits**, at 0.658. Nothing
   here displaces it for graphs under about 67M vertices.
2. **At the spill boundary S2 is decisively better than S1.** Where S1 must go
   to 8 bytes it returns 1.163, worse than LRU; S2 at a 6-bit payload returns
   0.763. That is a **34% reduction** against S1-8B, against a pre-registered
   kill criterion of 2%, and it still beats GRASP by 8.8%.
3. **The payload budget is real.** A 12-bit sidecar returns 1.027 and loses to
   GRASP, so the structure is not free and the stamps must stay narrow. The
   useful configuration is two stamps at 2-bit epochs plus 2 tier bits = 6 bits.

**Accounting gate.** Measured sidecar cost against the exact cache-line formula
`ceil(payload_bits * E / 8) / 64` lines per sweep, two sweeps:

| payload | predicted lines | measured delta | ratio |
|---:|---:|---:|---:|
| 6 bits | 11,778 | 13,577 | 1.153 |
| 8 bits | 15,706 | 17,594 | 1.120 |
| 12 bits | 23,558 | 26,127 | 1.109 |

Measured tracks the formula within 11-15%, and the ratio converges toward 1.0 as
the payload grows, which is consistent with a fixed conflict/refetch overhead
rather than a counting error.

Scope: one graph, one kernel, one LLC size, no prefetcher, cache_sim traffic. The
correctness gate (S1 and S2 must deliver identical destination, tier, stamps,
property address and output checksum across all kernels) is NOT yet run, so this
is a transport-cost result only and carries no claim that S2 preserves K2's
semantics. The pressure and cache-colour sensitivity sweeps in the registered
kill criterion are also outstanding.

### BASELINE: cache_sim, all five algorithms on one metadata SSOT

All five cache_sim kernels now configure delivery from `ecg_metadata.h` and
deliver through one site, so structure, width and placement cannot differ
between kernels. Each emits a receipt, and all five agree:

    [ECG-METADATA kernel=... delivery=packed stamps=2 epoch_bits=5 tier_bits=2
     id_bits=16 record_bytes=8 payload_bits=12 bytes_per_edge=8.000 ...
     packed_fits=1]

Note what the receipt exposes: `packed_fits=1` with `record_bytes=8`. The
two-stamp record *does* fit in 4 bytes here and is nonetheless charged 8, by
the Schedule-2 default. That inconsistency was invisible before the receipt
existed, and it is the defect that drove several earlier reversals.

**Semantic gate, all five kernels.** With metadata uncharged, the packed record
and the sidecar must be indistinguishable, because transport is removed and only
the stamps remain. They are, exactly:

| kernel | uncharged packed | uncharged sidecar |
|---|---:|---:|
| pr | 85,932 | 85,932 |
| bfs | 156,461 | 156,461 |
| cc | 84,333 | 84,333 |
| bc | 1,466,547 | 1,466,547 |
| sssp | 101,159 | 101,159 |

Identical L3 misses and writebacks in every case. Since the 15-cell conformance
gate verifies the eviction DECISION rather than how epochs were transported, a
structure that provably delivers identical stamps preserves conformance by
construction. That is what makes S2 admissible across three simulators.

**The baseline.** web-Google-n16, 128 kB LLC, 32 kB/256 kB private, no
prefetcher, ASLR disabled, off-chip traffic in both directions, versus LRU:

| kernel | LRU (lines) | GRASP | S1 4 B | S1 8 B | S2 6 bits |
|---|---:|---:|---:|---:|---:|
| pr | 130,189 | 0.835 | **0.657** | 1.164 | 0.763 |
| bfs | 149,459 | **0.829** | 1.007 | 1.538 | 1.323 |
| cc | 102,832 | **0.790** | 0.812 | 1.112 | 0.979 |
| bc | 1,340,196 | **0.910** | 1.092 | 1.145 | 1.204 |
| sssp | 140,514 | 0.762 | **0.546** | 0.714 | 1.033 |
| **geomean** | | 0.824 | **0.796** | 1.102 | 1.042 |

Four readings.

1. **A 4-byte packed record is the best structure overall** (0.796 against
   GRASP's 0.824), and wins outright on PageRank (0.657) and SSSP (0.546).
2. **GRASP wins the three traversal kernels**, so no general K2 lead exists.
   K2's advantage is concentrated where property reuse is high.
3. **S2 is the better fallback exactly where the packed record must widen.**
   Where S1 spills to 8 bytes, S2 at a 6-bit payload is better on pr (0.763 vs
   1.164), bfs (1.323 vs 1.538) and cc (0.979 vs 1.112).
4. **SSSP inverts that, and the inversion is structural, not noise.** Its edge
   is an 8-byte weighted node, so an 8-byte packed record still SUBSTITUTES for
   the edge and stays free (0.714), while a sidecar adds a second stream on top
   of it (1.033). The SSOT encodes this as a feasibility rule: a packed record
   is only usable when it can genuinely replace the edge, and delivery
   downgrades to a sidecar when it cannot.

So the structure choice is not global but per edge format: substitute when the
metadata fits inside the edge the kernel already reads, sidecar when it does
not. That is a sharper design statement than "K2 wins" and it is what the three
simulator ports must now reproduce.

Scope: one graph, one LLC size, no prefetcher, cache_sim traffic only. This is
the baseline gem5 and Sniper will be measured against, not a headline.

### PREFETCHING: the metadata stream is prefetchable only in ID-order kernels

Every number in the cache_sim baseline is prefetcher-off. Since the per-edge
record is a sequential stream, the obvious objection is that a real prefetcher
would hide it and the transport penalty would evaporate. Measured with the
honest address-only detector (degree 8), web-Google-n16, 128 kB LLC:

**PageRank**, which traverses vertices in ID order:

| config | demand misses | prefetch fills | off-chip | vs LRU |
|---|---:|---:|---:|---:|
| LRU | 128,970 | 0 | 130,189 | 1.000 |
| LRU + prefetch | 66,354 | 62,931 | 129,470 | 1.000 |
| K2 4 B | 85,082 | 0 | 85,539 | 0.657 |
| K2 4 B + prefetch | 23,788 | 62,126 | 86,033 | 0.665 |
| K2 8 B | 150,832 | 0 | 151,485 | 1.164 |
| K2 8 B + prefetch | 29,428 | 122,560 | 152,117 | 1.175 |

**BFS**, which traverses in frontier order:

| config | demand misses | prefetch fills | off-chip | vs LRU |
|---|---:|---:|---:|---:|
| LRU | 147,411 | 0 | 149,459 | 1.000 |
| LRU + prefetch | 147,392 | 74 | 147,562 | 1.000 |
| K2 8 B | 227,892 | 0 | 229,940 | 1.538 |
| K2 8 B + prefetch | 227,706 | 35 | 227,837 | 1.544 |

Three findings.

1. **On PageRank the prefetcher is extremely effective at hiding the metadata
   stream.** K2's demand misses fall 72% at 4 bytes (85,082 to 23,788) and 80%
   at 8 bytes (150,832 to 29,428). If the workload is latency-bound, most of the
   transport penalty could disappear in *time*.
2. **Traffic is unchanged in every single case.** 130,189 against 129,470;
   85,539 against 86,033; 151,485 against 152,117. A prefetcher relocates work,
   it does not remove it, so if the workload is bandwidth-bound the penalty
   stands in full. Every ratio versus LRU is stable to within 1%.
3. **On BFS the prefetcher does essentially nothing**: 35 fills against 227,892
   demand misses. The record is indexed by CSR edge position, so it is
   contiguous only when vertices are visited in ID order. BFS visits them in
   frontier order, so consecutive accesses jump between adjacency runs that
   average 7.7 edges, i.e. about 31 bytes, less than a cache line. The detector
   never confirms a stream.

**So the prefetchability of K2's metadata is a property of the traversal order,
not of the metadata.** ID-order kernels get it; frontier-order kernels cannot.
That is a falsifiable mechanism claim, and it explains why BFS is K2's worst
case: the transport doubles *and* cannot be hidden.

It also means the baseline's ranking is prefetch-robust when read as traffic,
and prefetch-sensitive when read as demand misses -- which is exactly why the
frozen metrics bar demand-miss arguments under an active prefetcher.

Whether the hidden latency on PageRank translates into speedup is still
unmeasured, and cache_sim cannot answer it: its fills are synchronous and free,
with no MLP, lateness, bandwidth or MSHR model. That requires gem5.

### THE TRADE: K2 converts exposed-latency misses into prefetchable traffic

The traffic ratios treat every line as equal. They are not. Decomposing L3
misses into PROPERTY (irregular, vertex-indexed) and STRUCTURAL (sequential edge
and record stream) on web-Google-n16 PageRank shows what K2 actually does:

| config | L3 misses | property | structural | property share |
|---|---:|---:|---:|---:|
| LRU | 128,970 | 66,152 | 62,818 | 51.3% |
| K2 4 B | 85,082 | **22,049** | 63,033 | 25.9% |
| K2 8 B | 150,832 | 25,041 | 125,791 | 16.6% |

K2 at 4 bytes cuts property misses by **67%** while leaving the structural
stream essentially unchanged (62,818 to 63,033), because a 4-byte record
substitutes for the 4-byte edge. At 8 bytes the property win survives but the
structural stream doubles, which is the entire penalty.

**The two streams do not cost the same, because only one is prefetchable.**
With the honest detector active:

| config | exposed demand misses | property | structural | prefetch fills |
|---|---:|---:|---:|---:|
| LRU | 66,354 | 61,185 | 5,169 | 62,931 |
| K2 4 B | **23,788** | 18,089 | 5,699 | 62,126 |
| K2 8 B | 29,428 | 21,053 | 8,375 | 122,560 |

The structural stream is about **92% prefetchable** (62,818 down to 5,169 for
LRU); the property stream is about **8%** (66,152 down to 61,185). Latency
exposure therefore lives almost entirely in the property stream, which is
exactly the stream K2 shrinks.

Consequently **K2 at 4 bytes exposes 64% fewer demand misses to full DRAM
latency than LRU** (23,788 against 66,354). Even K2 at 8 bytes exposes **56%
fewer** (29,428) *while using 17% more total bandwidth*. The bandwidth trade and
the latency trade point in opposite directions.

**So the answer to "is the trade zero-sum?" is no, and which way it resolves
depends on whether the memory system is bandwidth-saturated.**

- If **bandwidth-saturated**, traffic is the binding constraint and K2 at 8
  bytes loses by its 17% traffic increase.
- If **latency-bound with spare bandwidth**, the binding constraint is exposed
  misses and K2 wins substantially, at either width, because the extra edge
  traffic lands in the stream a prefetcher already covers.

cache_sim cannot settle this: it has no time, no DRAM model and no bandwidth
ceiling, so it cannot report utilisation. An order-of-magnitude estimate
suggests headroom -- 130,189 lines is about 8.3 MB of traffic for the whole
2-iteration kernel, which against a single-core achievable bandwidth of order
10 GB/s implies utilisation of a few percent -- but that is an estimate from a
model that does not simulate time, not a measurement, and it must not be quoted
as one.

This is the specific question the gem5 matrix now exists to answer, and it is
pre-registered here: **measure achieved DRAM bandwidth utilisation alongside
execution time.** If utilisation is low, the exposed-miss reduction should
appear as speedup at both record widths. If it is high, the 8-byte width should
lose in proportion to its traffic. Either outcome is informative, and the
prediction is stated before the run.

### BANDWIDTH SATURATION: gem5 already reports it, and it is low

The trade decomposition above leaves one question open: K2 can use more total
traffic while exposing far fewer demand misses to full DRAM latency, so which
side binds depends on whether the memory system is saturated. cache_sim cannot
answer that at all. gem5 can, and already does -- the statistic was simply never
captured into result rows.

From an existing archived gem5 run (`ecg_3sim_smoke_gem5_probe2_20260715`,
K2+StreamShield, PageRank, kron_s12_k4, 32 kB L3):

    system.mem_ctrl.dram.busUtil     0.83   # data bus utilisation, percent
    system.mem_ctrl.dram.peakBW  19207.00   # MiB/s
    system.mem_ctrl.dram.avgRdBW   139.87   # MiB/s
    system.mem_ctrl.dram.avgWrBW    19.46   # MiB/s

**Data bus utilisation is 0.83% of peak.** On that configuration the memory
system is essentially idle, which is the regime in which K2's trade is
favourable: extra bandwidth is nearly free and exposed latency is what costs.

Three reasons this is indicative and not yet the answer:

1. It is a **smoke configuration** -- kron_s12_k4 is a 4,096-vertex synthetic
   graph with a 32 kB L3, chosen to exercise mechanisms quickly, not to
   represent a realistic working set.
2. It is **single-core**. Utilisation is what several cores contend for, and a
   multi-core configuration could saturate where one core does not.
3. It is **one cell**, on the kernel most favourable to K2.

`roi_matrix` now captures `busUtil`, `busUtilRead`, `busUtilWrite`, `peakBW`,
`avgRdBW`, `avgWrBW` and `bwTotal`, plus a derived `dram_offchip_bytes` equal to
DRAM bytes read plus written, so the frozen primary metric is directly available
from a timing backend rather than reconstructed.

The pre-registered prediction is unchanged and now measurable: if utilisation
stays low on realistic cells, the 64% and 56% reductions in exposed demand
misses should appear as speedup at BOTH record widths; if it is high, the 8-byte
width should lose in proportion to its 17% traffic increase.

### FROZEN: three-simulator conformance re-established under the metadata SSOT

Introducing a second metadata delivery structure put the paper's foundation at
risk, because the 15-cell gate is what licenses treating the three simulators as
one mechanism. The gate has been re-run and re-frozen with transport now driven
by `ecg_metadata.h`:

| kernel | cache_sim | gem5 | Sniper |
|---|---|---|---|
| pr | ok | ok | ok |
| bfs | ok | ok | ok |
| bc | ok | ok | ok |
| cc | ok | ok | ok |
| sssp | ok | ok | ok |

`RESULT: ALL kernel x simulator cells CONFORM`. 15 of 15 cells pass, 13 carry a
decisive real-epoch victim, and every eviction obeys the shared specification.

Archived at `research/ecg-hpca/evidence/ecg_3sim_metadata_ssot_20260726` with 27
hashed inputs, captured from a clean worktree at `0f8121bf`.

Two things make this stronger than a repeat of the earlier freeze.

1. **Transport is now provably common.** All eleven sources -- five cache_sim
   kernels, five gem5 kernels and the Sniper workload -- derive record width and
   delivery structure from one header, and a test asserts none of them computes
   a width locally. Identical configuration produces byte-identical receipts on
   all three backends, over both schedules:
   `record_bytes=4 payload_bits=7` at Schedule-1 and `record_bytes=8
   payload_bits=12` at Schedule-2.
2. **Conformance is transport-independent by construction.** The gate verifies
   the eviction DECISION, and with metadata uncharged the packed record and the
   sidecar are byte-identical on all five kernels. A structure that delivers the
   same stamps therefore cannot change a victim choice, which is why a second
   delivery structure is admissible at all.

### WITHDRAWN: "the carried tier bits are genuinely free"

> **This section is withdrawn.** The measurement was real but the conclusion
> does not follow, and the arithmetic attached to it was wrong. Retained for
> audit. Do not cite.

The claim was that the record's GRASP tier bits cost nothing on the headline
variant, from `epoch_first` measuring 85,539 with and without them. Adversarial
review showed the result is tautological: `epoch_first` never reads
`ways[i].dbg` at all (`ecg_victim_policy.h`, the EPOCH_FIRST branch selects
records by recency, then stamped property by farthest distance, then falls back
to recency). A variant that never consults the tier cannot be shown to be
indifferent to it by removing it.

Three further defects in the same experiment:

1. **One delivery site was still ungated.** A hit-path assignment reloaded
   `edge_grasp_tier` regardless of `ECG_RECORD_TIER_BITS`, so the ablation was
   incomplete even where it did apply. Now gated; the finding is not re-derived
   here because the variant choice invalidates it independently.
2. **It was not tier versus no tier.** The insertion paths fall back to
   `classifyBucket(address)`, so the comparison was *carried* global-degree tier
   against *address-derived* GRASP tier, which is a different question.
3. **The scaling arithmetic was wrong.** Returning two bits to the id field
   multiplies addressable vertices by four, not two. The compact packer also
   hardcodes two tier bits, so no reach is currently reclaimed at all.

What survives is only the control: `grasp_only` moved from 102,252 to 103,391
(+1.11%) when the carried tier was withheld, which shows the gate reaches at
least one tier-dependent path. That is a sanity check on the mechanism, not a
result about tier cost.

A valid version of this experiment needs a variant that actually uses the tier
(`degree_first`, or a deliberately tier-tied construction), counters proving
zero carried-tier writes and zero tier-based victim decisions under ablation,
and a packer that honours `ECG_RECORD_TIER_BITS` so the bits are genuinely
reclaimed.

### CROSS-SIM DIVERGENCE: gem5 cannot deliver a 4-byte two-stamp record

The first gem5 timing cell disagreed with cache_sim in DIRECTION, not merely in
magnitude. web-Google-n16 PageRank, 128 kB LLC, two-stamp record, no
StreamShield, off-chip traffic versus LRU:

| simulator | traffic vs LRU | time vs LRU |
|---|---:|---:|
| cache_sim, same 16 kB/64 kB/128 kB geometry | **0.557** | (no timing) |
| gem5 | **1.189** | 1.167 |

Reproduced at cache_sim with gem5's exact geometry and policy resolution
(`ECG:K2` resolves `adaptive` to `epoch_first`, StreamShield off), so it is not
a configuration mismatch.

**Cause: the two simulators stream different container widths.** gem5's
Schedule-2 path builds `pvector<uint64_t> in_edge_pair_flat`, so it moves 8
bytes per edge structurally, whatever the bit budget computes. cache_sim models
the two-stamp record as substituting for the 4-byte CSR edge. The measurement
confirms it: gem5's K2 reads 2,675,776 more DRAM bytes than LRU, which is 1.33
passes of a 4-byte-per-edge array, i.e. the extra 4 bytes per edge of an 8-byte
container. DRAM *writes* are unchanged at 1.004, so this is not the array being
built inside the ROI; it is the stream being twice as wide.

The metadata SSOT reported `record_bytes=4` for that run. It was computing the
budget, which is what the record *could* occupy, while gem5 materialised it in a
wider container. That is precisely the divergence the header exists to prevent,
and the receipt is what exposed it -- but only once cache_sim and gem5 were
compared directly.

`declareContainerBytes()` now lets a backend state the container it actually
streams, and gem5's Schedule-2 declares 8. Its receipt reports
`record_bytes=8 bytes_per_edge=8.000`, matching what the guest moves.

**Consequence for the timing matrix.** The pre-registered 4-byte versus 8-byte
contrast **cannot be run on gem5 as the code stands**, because gem5 has no
4-byte two-stamp path: both arms would stream 8 bytes and the contrast would be
vacuous. Two options, neither yet taken:

1. Implement a 4-byte two-stamp record in gem5. The bits fit -- 16 id + 2x5
   epoch + 2 tier = 28 -- so this is a `uint32_t` pair-record path in
   `ecg_epoch_builder.h` plus the guest loop, not a design change.
2. Run the width contrast on cache_sim only, where substitution is modelled,
   and use gem5 solely for 8-byte timing and bandwidth utilisation.

Until one is chosen, no width claim may cite gem5.

What survives from the aborted run is the baseline behaviour, which carries no
ECG record and is therefore unaffected: bus utilisation 1.14-1.80% across LRU,
GRASP and P-OPT on both sampled graphs, and P-OPT converting a 38% traffic
reduction into only a 13.5% time reduction. The low-utilisation regime and the
sub-proportional traffic-to-time relationship both hold.

### WITHDRAWN: Is K2 memory-bound or instruction-bound? (gem5 full-work timing)
> **This section is withdrawn.** It is retained verbatim for audit, not as a
> result. Every number below is inadmissible under the frozen metrics: the rows
> are `ecg.load2` rows that `METHODOLOGY.md` marks superseded, the argument
> rests on IPC as independent evidence when IPC ratio is algebraically
> instruction ratio over time ratio, and the 0.864 headline is a counterfactual
> normalisation. See "RETRACTION: three K2 conclusions withdrawn after review"
> above. Do not cite, quote or aggregate anything in this section.

The cache_sim metric counts LLC misses, which implicitly prices a sequential
edge-stream miss the same as an irregular property miss. That understates K2,
because K2 deliberately trades irregular misses for sequential ones. gem5 can
arbitrate: it models MLP, DRAM row locality and prefetching, and the sampled
full-work matrix carries `timing_valid_for_speedup=1` with no instruction cap.

gem5, 15 sampled full-work cells, `ecg.load2` delivery, charged, versus LRU:

| Policy | instructions | time | IPC | DRAM read bytes | LLC misses |
|---|---:|---:|---:|---:|---:|
| GRASP | 1.000 | 0.957 | 1.045 | 0.874 | 0.875 |
| charged P-OPT | 1.000 | 0.955 | 1.047 | 0.847 | 0.847 |
| K2 | 1.495 | 1.284 | **1.165** | 1.091 | 1.094 |
| K2-online+StreamShield | 1.422 | 1.228 | **1.157** | 1.029 | 1.032 |

The decisive number is IPC. K2 incurs *more* LLC misses yet sustains a **higher
IPC than LRU** (1.157-1.165), and its DRAM read traffic rises only 3-9% against
a 3-7% miss increase. A policy whose extra misses were stalling the pipeline
would show IPC falling, not rising. The added traffic is the sequential edge
record stream, and the memory system absorbs it.

K2's measured slowdown is therefore **instruction-bound, not memory-bound**: it
executes 1.42-1.50x the instructions of LRU because the guest still constructs
and delivers K2 records in software. Dividing time by the instruction ratio
isolates the memory-system effect:

| | instr | time | instr-normalised time |
|---|---:|---:|---:|
| K2-online+StreamShield | 1.422 | 1.228 | **0.864** |

**13 of 15 cells are faster than LRU once instruction counts are matched**
(geomean 0.864, i.e. 13.6% faster), and IPC improves in the same 13 cells.

Two honest boundaries:

- The normalised figure is a **projection, not a measurement**. What is measured
  is that IPC rises and DRAM traffic barely moves; the 0.864 assumes the
  software delivery overhead can be removed, which is precisely what the
  hardware K2-I instruction exists to do. It is not a claimed speedup.
- **SSSP is the exception** (1.09 and 1.09, still slower when normalised), and
  consistently so: SSSP already uses compact weighted records, so it carries the
  least software overhead to remove, and what remains is genuine memory cost.

This reframes the cache_sim ranking. K2's miss-count deficit does not translate
into a proportional time deficit, so a miss-count-only comparison against GRASP
understates it. Scope: sampled graphs at reduced cache sizes; a full-graph
timing matrix remains pending.

### Hawkeye baseline scaffold
A clean-room Hawkeye policy module and LLC-only cache_sim adapter are now
implemented. The module includes 128-quantum OPTgen, 64 sampled cache sets,
350x8 sampler history, separate 3-bit demand/prefetch predictors, negative
training on friendly-line eviction, and Hawkeye's 3-bit RRPV rules. cache_sim
uses compile-time static graph-access-site IDs because it has no instruction
PC; rows are labeled `HAWKEYE_PROXY` and
`proxy_not_real_instruction_pc`. They remain development diagnostics.

The gem5 real-PC port is now implemented as `GraphHawkeyeRP`. It uses the same
clean-room core, obtains signatures from `Request::getPC()`, keeps demand and
prefetch training separate, ignores incoming writeback training while retaining
victim-side learning, and defers all predictor/OPTgen mutation until the fill
actually commits. Both gem5 ISAs build and the SimObject instantiates. The
dedicated `ecg_gem5_hawkeye_gate` now passes 30/30 rows:

| Kernel | Hawkeye/LRU L3 misses | Hawkeye speedup vs LRU |
|---|---:|---:|
| PR | 1.098x | 0.985x |
| BFS | 1.216x | 0.971x |
| SSSP | 1.182x | 0.986x |
| BC | 1.162x | 0.960x |
| CC | 1.144x | 0.981x |
| Geomean | **1.160x** | **0.977x** |

Every Hawkeye row uses the real request instruction PC. Hawkeye is worse than
LRU in every synthetic cell, so this is an implementation/execution gate, not
the general learned-policy result required for a paper ranking.

### Physical-characterization harness
`analysis/k2_cacti_packet.py` now emits hashed configs from the vendored CACTI
6.5 template for the 8 MiB/16-way LLC, a 1RW metadata SRAM, and a 1R1W port
sensitivity. The metadata input rounds each way's 49 logical plus seven SECDED
bits to 64 bits, then exposes all 16 ways as an 8,192-row x 1,024-bit array.
It also parses isolated `out.csv` reports and hashes every config/report and the
executed CACTI binary.

`analysis/k2_rtl_packet.py` emits hashed synthesis inputs for the exact
seven-variant victim-ranking core and 16-way 49-bit SECDED codecs. Verilator
functional checks cover all variant dispatch, selector ordering, collapsed
RRIP aging, property/context/two-epoch qualification, online winner updates,
and single/double-error behavior; Yosys structural checks pass.
`k2_replacement_path` is the complete replacement top at the fixed
32,768-epoch physical point. Its two descriptors accept only prefiltered
epoch-governed regions and synthesize 32 per-way range checks. Non-baseline
recency-rank maintenance must be charged separately. These are RTL inputs, not
technology measurements.
Per-unit request-state inputs are now implemented and verified: exact sticky
MSHR merge, epoch/context CSR state, a 95-bit pipeline copy, an optional
eight-lane 32-bit sequence allocator, and optional registered 16-way recency
rank state. Final area must multiply them by disclosed target-machine counts;
baseline MSHR CAM/allocation and queue control are excluded as common logic.
The physical schema enforces those counts and per-access activations and keeps
per-set recency area in the way-scaled equal-area term.

`analysis/k2_physical.py` validates explicit baseline-cache, metadata-SRAM,
SECDED, replacement-logic, and request-path measurements plus mandatory source,
config, report, RTL, library, and synthesis hashes. Hit lookup, request path,
and eviction selection delays are reported separately, and CACTI/synthesis
technology nodes must match. The harness rejects
missing or placeholder values and contains no default physical estimates.
CACTI 6.5 cannot represent 14/15-way associativity, so the existing reduced-way
rows remain simulation sensitivities. No CACTI or synthesis result has been
supplied or frozen.

### Three-cost accounting table
`analysis/three_costs.py` now generates the reviewer-facing accounting table
for web-Google, soc-pokec, and cit-Patents at configurable LLC sizes. It
separates:

1. K2 extra bytes per edge and total bytes for one active traversal stream;
2. K2's 33-bit minimum and 49-bit contextual metadata per line, expressed as
   added SRAM way-equivalents;
3. P-OPT's size-correct matrix bytes and reserved LLC data ways.

The default 2/8 MiB table reproduces the runner's capacity charge: P-OPT needs
1/1 ways on web-Google, 2/1 on soc-pokec, and 4/1 on cit-Patents. These are
analytical accounting rows, not performance results.

### Executable HPCA claim gate
`claim_gate.json` now links contribution deltas and headline claims to explicit
evidence gates. The validator checks commit reachability/file existence and
rejects any `allowed` claim whose dependencies are pending. It machine-pins the
major prohibitions, including treating the 1.329x packed K2-I-like model or
1.171x TPI as K2-M evidence, comparing absolute gem5/Sniper rates, claiming
zero hardware overhead, and ranking policies from the bounded synthetic cell.

### Static K2-M versus K2-I decomposition
The canonical RV64 U32.D32 assembly, anchored to production funct7 encodings,
reports:

| Sequence | Body instructions | Destination extraction | Address generation | Property load |
|---|---:|---:|---:|---|
| Baseline | 6 | 2 | 2 | ordinary `lw` |
| K2-M | 6 | 2 | 2 | `ecg.k2.mload` |
| K2-I | 2 | 0 | 0 | indexed `ecg.k2.iload` |

Thus K2-M does not claim an instruction reduction; K2-I removes four canonical
instructions. This is static decomposition only and does not imply timing.

### Exact Sniper governed-load association
The transport-matched Sniper K2-M workload now executes identical bind/clear
magic markers for every policy immediately around each edge-governed
destination-property load. The marker carries the exact virtual address; the
LLC consumes it on the matching hit or miss and then obtains the line-min K2
payload from the existing sideband. PR binds `contrib[neighbor]`, BFS
`parent[dest]`, SSSP `dist[dest]`, BC forward `depth[dest]` and
`path_counts[dest]`, and CC edge-phase `comp[dest]`. Source loads, BC backward
work, and CC pointer chasing/compression remain deliberately unmarked.

This closes source-plus-line association in the implementation without adding
policy-specific instructions or memory references. Existing frozen mechanism
rows predate the marker and are not relabeled; a future focused certification
must regenerate them.

The bind latch now also snapshots the per-core quantized current epoch and a
monotonic ROI context. Miss victim selection consumes the allocating request's
snapshot; L3 hits refresh the resident line with the same context, and stale or
unmarked requests cannot use K2 epochs. This removes eviction-time rereading of
the mutable outer-vertex clock, but remains a Sniper model rather than execution
of the RISC-V CSR ISA, so this milestone alone adds no timing claim.

### K2-I target-instruction correction
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
diagnostic, not the final all-kernel result; the clean K2-I gem5 matrix
must be rerun.

Sniper did not contain gem5's architectural clear/trace instructions, but its
fused PR/BFS/BC/CC loops still tested disabled software-delivery and legacy-clear
conditions on every edge. A global compiler unswitch was rejected because it
changed LRU by 16.6% and reduced the soc-pokec BFS K2 speedup. Surgically
splitting only the fused K2 loop preserves LRU while changing that BFS cell from
1.127x to 0.806x LRU instructions and from 1.306x to 1.705x speedup; its L3 miss
rate remains approximately 0.532. Frozen Sniper timing must be rerun before
these diagnostics replace the current 120-row table.

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

### Final compact sampled Sniper packed-record extension model
The post-scope Sniper-only rerun contains 120 valid rows: three deterministic
samples x PR/BFS/SSSP/BC/CC x eight policies. Strict invariants pass: every
group is hash-consistent, every LLC is exercised, BC records
`depth,path_counts`, and compact weighted SSSP replaces the original edge with
one 8-byte K2 record. This matrix is an idealized K2-I-like packed-loop model:
Sniper executes x86 destination extraction/indexed loads and infers metadata
from source plus property line rather than executing a K2-I instruction. Its
speedup, instruction reduction, and TPI are not attributable to K2-M. It is not
measured K2-I ISA timing.

Geomean across all 15 graph/kernel cells versus LRU:

| Policy | Speedup | TPI speedup | Instruction ratio | Effective L3-miss reduction |
|---|---:|---:|---:|---:|
| SRRIP | 1.029x | 1.029x | 1.000x | 3.46% |
| GRASP | 1.100x | 1.100x | 1.000x | **15.05%** |
| charged P-OPT | 1.082x | 1.082x | 1.000x | -18.55% |
| packed K2-I-like | 1.326x | 1.169x | 0.881x | 5.30% |
| packed K2-I-like online | 1.320x | 1.163x | 0.881x | 3.74% |
| packed K2-I-like+StreamShield | 1.327x | 1.169x | 0.881x | **5.89%** |
| packed K2-I-like online+StreamShield | **1.329x** | **1.171x** | 0.881x | 4.72% |

Packed K2-I-like online+StreamShield kernel geomean:

| Kernel | Speedup vs LRU | vs GRASP | vs charged P-OPT | Effective miss reduction vs LRU |
|---|---:|---:|---:|---:|
| PR | **1.792x** | 1.600x | 1.526x | 16.51% |
| BFS | **1.675x** | 1.523x | 1.546x | 4.07% |
| SSSP | **1.145x** | 1.034x | 1.024x | 3.30% |
| BC | **1.082x** | 1.058x | 1.065x | 4.64% |
| CC | 1.115x | 0.968x | 1.089x | -6.31% |

The four packed extension-model variants collectively win 9/15 timing cells,
GRASP wins 4/15, and
SRRIP/P-OPT win one each. The K2 wins split across online K2+StreamShield (5),
static K2+StreamShield (3), and online K2 (1). Relative to the surgical
sidecar matrix, compact delivery raises K2-online+StreamShield from 1.282x to
1.329x overall, reduces its instruction ratio from 0.928x to 0.881x, raises
SSSP from 0.966x to 1.145x, and raises BC from 1.074x to 1.082x. These are
extension-model results. The 1.171x TPI uses a different instruction mix and
must not be interpreted as a K2-M estimate.

Compact weighted SSSP now has a 1.015x instruction ratio and wins in geomean:
1.145x versus LRU, 1.034x versus GRASP, and 1.024x versus charged P-OPT. The
result remains heterogeneous: web-Google wins strongly, soc-pokec beats LRU
but narrowly trails GRASP/P-OPT, and cit-Patents remains a substantial loss.

The shortest web-Google BFS cell remains unusually strong at 2.361x, versus
1.163x on cit-Patents and 1.713x on soc-pokec. However, excluding that cell
leaves K2-online+StreamShield at 1.276x overall versus GRASP at 1.107x, so the
new overall ordering survives this single-cell exclusion. Sniper assigns the full CPI stack to `unknown`,
so the TPI gains cannot be decomposed into LLC, DRAM, or pipeline components.

The effective-miss column charges P-OPT's matrix stream as additional cache-line
fills, which dominates small working sets. The packed extension model executes
fewer instructions than LRU overall; compact weighted SSSP is near baseline
instruction count.

The packed extension model is strong on PR/BFS and positive on sampled
SSSP/BC. CC beats LRU and P-OPT but remains slightly
behind GRASP; cit-Patents SSSP remains the principal negative cell. No K2-M
timing claim is frozen yet.

The cit-Patents loss does **not** extrapolate as a size-only failure. The
`n18-sym` sample retains 262,144 vertices but only 340,054 undirected edges; it
has reported average degree 2.96, average path length 11.69, diameter 14, and
near-zero neighbor overlap. The full graph has 3,774,768 vertices and
16,518,947 undirected edges, with reported average degree 11.47, average path
length 3.60, diameter 4, and 35x higher neighbor overlap. The induced sample is
therefore much thinner and less locally reusable than the full topology.

A focused full-graph cache_sim SSSP gate with the current compact record confirms
the distinction:

| Policy | L3 miss rate | Effective L3 misses |
|---|---:|---:|
| LRU | 0.5630 | 18.237M |
| GRASP | 0.4192 | 13.579M |
| charged P-OPT | 0.4330 | 14.967M |
| K2 | 0.4137 | 13.401M |
| K2-online+StreamShield | **0.3943** | **12.770M** |

Thus compact K2-online+StreamShield reduces full-graph effective misses 30.0%
versus LRU, 6.0% versus GRASP, and 14.7% versus charged P-OPT. This rejects the
hypothesis that larger cit-Patents necessarily breaks K2; a bounded full-graph
Sniper timing probe is still required before making the same timing claim.

Aggregate: `results/ecg_experiments/paper_pipeline/`
`ecg_sniper_sampled_allalg_compact_scope_final_20260721/aggregate/`.

Full citation risk gate:
`results/ecg_experiments/final_paper_runs/`
`ecg_cache_sim_citpatents_sssp_compact_full_20260721/roi_matrix.csv`.

### Fused sampled PageRank timing
The historical `ecg_sniper_sampled_pr_streamengine` profile contains nine
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
`unknown`, so the TPI improvement cannot be decomposed further. This
pre-surgical profile is retained for attribution only; its timing is superseded
by the current all-kernel matrix above.

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

## K2 on gem5, decomposed: the record is free, decoding it is not (2026-07-26)

STATUS: measured, single cell, no claim promoted. `claim_gate.json` still marks
every performance claim `prohibited`.

Cell: web-Google-n16, PageRank, gem5 RISC-V, 16kB L1D / 64kB L2 / 128kB LLC
16-way, no prefetcher, 32 epochs, Schedule-2, compact 4-byte record.
Receipts verified `bytes_per_edge=4.000` with `Schedule-2 COMPACT record ON`,
zero `ECG-METADATA-FATAL`. Instruction counts are
`system.cpu.commitStats0.numInsts`, which `m5_reset_stats` DOES clear; `simInsts`
does NOT and therefore includes graph loading and record construction.

Versus LRU:

| policy | time | traffic | ROI insts |
|---|---:|---:|---:|
| POPT | 0.870 | 0.616 | 1.000 |
| GRASP | 0.937 | 0.822 | 1.000 |
| LRU | 1.000 | 1.000 | 1.000 |
| ECG:K2 | 1.230 | 0.964 | 1.726 |

K2 moves less traffic than LRU and still takes 23% longer. The matched
decomposition, using `ECG:K2_LRU` (identical transport, `lru_only` victim rule),
separates the two causes. The arms execute **19,580,752 ROI instructions each**,
a ratio of 1.000000, so they are matched on work by measurement rather than by
assertion.

| contrast | ROI insts | time | traffic | LLC misses |
|---|---:|---:|---:|---:|
| replacement rule (K2 vs K2_LRU) | 1.0000 | 0.989 | 0.963 | 0.961 |
| transport (K2_LRU vs LRU) | 1.7263 | 1.243 | 1.0015 | 1.001 |

Two findings.

**The compact record is free in bytes.** Adding the record stream changes
off-chip traffic by +0.15% and LLC misses by +0.1%, because the record
substitutes for the CSR edge rather than adding to it. This is the first
independent confirmation on a timing simulator of what cache_sim measured
(+446 lines on 4.36M, +0.01%), and it settles the cross-simulator divergence
recorded earlier: gem5 previously streamed a 64-bit container while reporting
the 4-byte budget.

**The decode is not free in time.** The same arm executes 72.6% more ROI
instructions, about 16 extra per edge, and takes 24.3% longer.
`widenEpochPair32` rebuilds the canonical 64-bit layout in software with
runtime shift amounts. DRAM bus utilisation is 1.2--1.7%, so the memory system
is ~98% idle and the instructions cost far more than the bytes they save.

The replacement rule itself is worth 3.7% of traffic and 1.1% of time on this
cell, which is real but small, and is swamped by the transport tax.

Consequence for the evaluation: at low utilisation the binding resource is
work, not bandwidth. A metadata format that must be decoded in software cannot
win here however narrow it is, so the width contrast on its own understates the
compact record. This is an argument for delivering the record through the ISA,
but only measurement of the ISA path can settle it; nothing here promotes a
claim.

Evidence: `results/ecg_experiments/probes/`
`k2_transport_decomposition_20260726_143844/`.

### Asymmetry found while reading the above: gem5 P-OPT is partly idealised

The same matrix reports P-OPT at 0.616--0.702 of LRU traffic and 0.870--0.909 of
LRU time, ahead of everything else. That row is not a comparable baseline.

gem5 delivers the rereference matrix through a **sideband file** read directly by
the replacement policy, so streaming its columns costs no simulated traffic, no
simulated latency and no instructions. What IS charged is LLC capacity: the
`size_correct` reserve model reserves ways for the resident columns
(1 way of 16 on web-Google-n16), which is the paper-faithful capacity charge.

The runner already records the omitted stream as `popt_matrix_stream_bytes` with
`popt_matrix_stream_mode=analytic`. Folding it back in:

| graph | gem5 off-chip | + matrix stream | understated by |
|---|---:|---:|---:|
| web-Google-n16 | 8,756,416 | 9,804,992 | 12.0% |
| soc-pokec-n16 | 12,100,608 | 13,149,184 | 8.7% |
| cit-Patents-n18-sym | 16,262,080 | 20,456,384 | 25.8% |

Execution time cannot be corrected this way at all: those bytes never entered
the memory system, so none of their latency appears anywhere in the reported
time. cache_sim does model this stream as real non-temporal accesses
(see the P-OPT matrix-stream section above); gem5 does not.

Under the frozen evaluation metrics an idealised mechanism is INELIGIBLE for a
performance claim, so gem5 P-OPT rows are an upper bound on P-OPT and must not
be quoted as a baseline K2 was measured against.
`record_width_timing.py` now prints this before any ratio.

### The two record widths fail for opposite reasons (2026-07-26)

Matched arms, web-Google-n16 PageRank, gem5, identical in everything but the
record container. Versus LRU:

| arm | ROI insts | traffic | time |
|---|---:|---:|---:|
| 8-byte record | 0.933 | 1.189 | 1.065 |
| 4-byte record | 1.726 | 0.964 | 1.230 |

The 8-byte record executes FEWER instructions than the LRU baseline (0.933),
because the record loop replaces the CSR adjacency walk and hands back the
destination directly; but the 64-bit container doubles the structural stream,
so it moves 18.9% more bytes. The 4-byte record fixes the stream and moves 3.6%
FEWER bytes than LRU, but `widenEpochPair32` rebuilds the canonical layout in
software with runtime shift amounts, costing 72.6% more instructions.

Directly: widening the record multiplies traffic by 1.233 and ROI instructions
by 0.541. Those are not the same experiment, so this contrast is width PLUS
software decode, and neither arm isolates width.

The combination that has never been measured is a narrow record with the decode
in hardware.

#### PRE-REGISTERED before running (stated here first, deliberately)

`ecg_extract2c` takes the 32-bit record in rs1 and a loop-invariant format word
(`id_bits | epoch_bits << 8`) in rs2, widens in the decoder, and returns the
destination. It delivers metadata through the same
`setDecodedEcgExtractHint2` path as `ecg_extract2`, so the two cannot mean
different policies.

Predictions for the compact-ISA arm versus LRU on this cell:

1. ROI instructions land near the 8-byte arm's 0.933, and clearly below the
   software-decode arm's 1.726. If they do not, the decode was not the cost.
2. Off-chip traffic stays at the compact arm's ~0.964, unchanged, because the
   instruction changes decode only and not what is fetched.
3. LLC misses and off-chip bytes must match the software-decode 4-byte arm to
   within run-to-run noise. Identical metadata must produce identical victim
   decisions; a difference means the decoder disagrees with
   `widenEpochPair32` and the arm is invalid regardless of its timing.
4. Execution time falls below the software-decode arm's 1.230. Whether it falls
   below 1.000 is NOT predicted.

Failure of 3 kills the arm outright. Failure of 1 or 2 falsifies the decode
diagnosis above.

### Result: the compact record wins once the decode is in the ISA (2026-07-26)

The pre-registered predictions above were checked against a matched three-arm
probe on web-Google-n16 PageRank, gem5, 16kB/64kB/128kB, no prefetcher, 32
epochs. All three arms use the PLAIN `packed + ecg.extract2` delivery
(`GRAPHBREW_K2_FUSED_LOAD=0`), because every fused delivery carries the
canonical 64-bit record and has no 32-bit form, so leaving one active would
confound width with decode.

| arm | ROI insts | off-chip bytes | LLC misses | ticks |
|---|---:|---:|---:|---:|
| 8-byte record | 25,680,038 | 16,880,448 | 253,507 | 69,133,793,500 |
| 4-byte, software decode | 38,243,258 | 13,702,208 | 203,858 | 81,845,041,000 |
| 4-byte, ISA decode | 15,293,846 | 13,706,624 | 203,937 | 47,799,502,000 |

Against the pre-registered predictions:

1. **RETRACTED as stated; the comparison was not matched.** The numbers
   (0.400 of the software-decode arm, 0.596 of the 8-byte arm, "45.7
   instructions per edge removed") are real measurements of those two binaries,
   but they do not isolate decode. The software-widen and 8-byte arms call
   `gem5_ecg_extract2_instruction`, which invokes the trace helper
   unconditionally, and the `GEM5_ECG_EXTRACT2` / `GEM5_ECG_CLEAR_EXTRACT2_HINT`
   macros, which re-test `gem5_ecg_extract_enabled()` per edge. Each of those is
   a function-local static whose initialisation guard executes on every call.
   Disassembly of the built binaries puts the unintended cost at roughly 18
   instructions per edge for the disabled trace and about 11 more for the
   redundant enable check -- on the order of 14.6M instructions over 502,529
   edges, all charged to the arms the compact-ISA arm was compared against and
   none charged to it.

   This is the same defect that was found and fixed inside
   `gem5_ecg_extract2c_instruction`, left unfixed on the other side of the same
   comparison. The arms are now symmetric (direct untraced calls, trace
   selected outside the loop) and the contrast must be re-measured before any
   figure from it is used.
2. **Passed.** Off-chip traffic 1.000322 of the software-decode arm: decoding
   changed nothing about what is fetched.
3. **Passed only in the weak sense, and the original wording was wrong.**
   LLC misses 1.000388 (203,937 versus 203,858). The first write-up attributed
   the 79-miss gap to "cycle placement", which is not a real explanation: gem5
   here is bit-deterministic (the K2 cell reproduces bit-exactly across two
   independent runs), so a difference is a difference. The two arms run
   different code -- the compact-ISA loop is much shorter than the
   software-widen loop -- so their instruction footprint and stack access
   pattern differ, and that is what moves 0.04% of the misses.

   The prediction asked for a MEASUREMENT of metadata equality, and the first
   write-up supplied an argument from inspection instead (both paths call the
   identical `setDecodedEcgExtractHint2`). That gap has since been closed by
   measurement, below.
4. **Passed.** Time falls to 0.584 of the software-decode arm and to 0.691 of
   the 8-byte arm.

So the decode diagnosis holds for the component it names: removing the software
widen leaves an arm that is simultaneously the narrowest in traffic and the
cheapest in instructions of the three.

Two scoping caveats that the first write-up did not state.

**These three arms are NOT on the same delivery path as the matrix.** They run
with `GRAPHBREW_K2_FUSED_LOAD=0`, i.e. plain `packed + ecg.extract2(c)`, while
every K2 cell in the width matrix used the fused `ecg.k2.iload`. The difference
is large, not cosmetic: the matrix's 8-byte K2 cell executes 10,586,105 ROI
instructions (below LRU's 11,342,919, because fusion absorbs the CSR walk),
whereas the non-fused 8-byte arm here executes 25,680,038. The correct
comparison is strictly within this table; none of these three arms is a
replacement for the fused matrix row.

**"The entire penalty is software decode" is true of the fused arm only.** In
the fused decomposition the transport tax was x1.7263 instructions, about 16.4
per edge, which is the widen budget. Here, with the widen gone, the ISA arm is
still at x1.348 of LRU -- roughly 7.9 instructions per edge for the record
load, the extract, the hint clear and the scalar property load. Those are costs
that fusion absorbs and this delivery does not. Removing the decode removed the
decode; it did not make the transport free.

**It is still not a win against LRU.** Against the LRU cell at this geometry
(43,149,560,500 ticks, 14,204,608 bytes, 11,342,919 ROI instructions), the
ISA-decode arm is 1.108 in time, 0.965 in traffic, 1.348 in instructions. It
also remains slightly behind the FUSED 8-byte arm measured in the matrix
(1.065), because fusing the record load with the property load removes another
instruction per edge that this arm still pays.

The obvious next mechanism, and the one the numbers now point at directly, is a
fused compact property load: the fused family currently forces the 64-bit
record, which is why the best compact arm cannot yet use it. No claim is
promoted here; `claim_gate.json` is unchanged.

Evidence: `results/ecg_experiments/probes/isa2_sw4b_181201`,
`isa2_w8b_181211`, `isa2_hw4b_fixed_183145`.

### Provenance corrections found by adversarial review (2026-07-26)

Three scoping problems in the sections above, none of which changes a number,
all of which change what the numbers are entitled to say.

**The width guard was not running for the runs it is credited to.** The commit
that put `ECG_EXPECT_BYTES_PER_EDGE` into the gem5 guest allowlist landed at
18:34. The record-width matrix finished at 17:49 and the transport
decomposition at 15:06. Both therefore ran while the abort-on-mismatch guard
was inert on gem5. Their widths are correct -- every stage's guest receipt was
read by hand and reports `bytes_per_edge=4.000` or `8.000` as intended -- but
they are trusted because the receipt agrees, NOT because anything enforced it.
Only the compact-ISA probe ran with the guard live. Any rerun should be treated
as the first enforced measurement of these cells.

**The LRU denominator is imported from another run.** The transport
decomposition contains only `ECG:K2` and `ECG:K2_LRU`; the plain-LRU figures it
divides by (43,149,560,500 ticks, 14,204,608 bytes, 11,342,919 instructions)
come from the width matrix. That is defensible here because gem5 is
bit-deterministic under the fixed-length hashed sideband directory -- the K2
cell reproduces bit-exactly across the two independent runs, including
`simTicks`, `commitStats0.numInsts` and `dram.bytesRead::total` -- but the
reproduction was only ever exhibited for K2, and it is the LRU denominator that
carries the "the record is free" claim. The determinism gate should be shown
for LRU specifically.

**`ECG:K2_LRU` equals LRU only in `ECG_GRASP_POPT` mode.** The `lru_only`
variant selects the way with the lowest `lastTouchTick`, and the ECG
replacement policy updates that field on both touch and reset exactly as gem5's
LRU does -- but only inside the `ECG_GRASP_POPT` branch. In any other ECG mode
that branch is skipped, recency never advances on a hit, and `lru_only`
degenerates to FIFO. The decomposition used `ECG_GRASP_POPT`, so the arms are
sound; but `ECG:K2_LRU` is not a general "K2 transport with LRU replacement"
primitive and must not be reused as one.


### Metadata equality, measured rather than argued (2026-07-26)

The compact path is now wired into the existing K2 delivery trace on both
sides: the guest emits `[ECG-K2-EXPECT ...]` for the software widen, and
`ecg_extract2c` emits the same record from inside the gem5 decoder. The guest
widen is computed only when `ECG_K2_DELIVERY_TRACE` is set, so it does not
re-enter the measured arm.

Running both arms on web-Google-n16 with `ECG_K2_DELIVERY_TRACE=200` and
comparing `(seq, dest, tier, epoch1, epoch2)`:

    software-widen records : 200
    ISA-decoder records    : 200
    compared               : 200
    mismatches             : 0

First records agree exactly: `(0, 93, 1, 0, 0)`, `(1, 196, 1, 0, 0)`,
`(2, 290, 1, 0, 0)`. The decoder's widening therefore reproduces
`widenEpochPair32` field for field, which -- together with the per-record proof
that `widenEpochPair32` reproduces the 64-bit builder -- closes the chain across
all three transcriptions of the format. Prediction 3 is now measured; the
residual 0.04% LLC-miss delta is code-footprint drift and nothing else.

### A build trap that made the first attempt at the above silently fail

The first traced run produced no guest records at all. The cause was not the
trace: `make gem5-riscv-m5ops-pr` reported success while rebuilding nothing,
because the gem5, Sniper and cache_sim rules listed gapbs, graphbrew and
external headers as prerequisites but NOT the ECG headers. Editing
`ecg_metadata.h` or `gem5_harness.h` therefore left every kernel binary stale
with make reporting "Built ...". The guest binary under test was 90 minutes
older than the header change it was supposed to contain.

This is the same failure mode as the inert enforcement knob, one layer lower,
and it is worth stating because it silently weakens any measurement that
follows a header-only edit. The build rules now list `DEP_ECG`, and a test
compares binary mtimes against the ECG headers, since a stale binary is
otherwise perfectly valid and cannot be detected any other way.

### Cross-run denominators carry ~2% uncertainty; within-run ones do not (2026-07-26)

Adversarial review objected that the transport decomposition divides by an LRU
cell taken from a different run. Checking that objection produced a result worth
recording in its own right.

Three LRU cells, identical geometry, web-Google-n16 PageRank:

| run | ticks | off-chip | ROI insts |
|---|---:|---:|---:|
| probe A | 42,400,004,000 | 14,210,048 | 11,279,926 |
| probe B, identical command, different out-dir | 42,400,004,000 | 14,210,048 | 11,279,926 |
| width matrix | 43,149,560,500 | 14,204,608 | 11,342,919 |

**gem5 is deterministic.** A and B agree to the tick, so nothing here is random
and re-running a cell reproduces it exactly.

**Nominally equivalent runs are not identical.** The matrix cell differs from the
probe by 1.74% in time, 0.56% in instructions and 0.04% in traffic. Data
placement is part of it -- the property arrays land one page apart in that
particular pair (`scores:0x3c1000` in the probe versus `scores:0x3c0000` in the
matrix), and at a 16kB L1 and 128kB LLC one page is worth nearly 2% of time.

But placement is NOT a sufficient explanation, and the first write-up asserted
it as though it were. Two later LRU cells reported IDENTICAL addresses
(`scores:0x3c1000`, `contrib:0x402000`) and still differed by 1.4% in time
(42,400,004,000 versus 43,005,214,500 ticks). Something else varies with the
invocation as well; it has not been identified. What is established is the
magnitude, not the mechanism: nominally identical cells in different
invocations differ by up to ~1.7% in time, while an identical command
reproduces to the tick. The runner already fixes the sideband directory to a
constant LENGTH to reduce this (`roi_matrix.py:83-101`); that mitigation is
necessary and demonstrably not sufficient.

Consequences, applied to the results above:

- **Comparisons must be within-run where possible.** The
  replacement-versus-transport split (K2 versus K2_LRU) is a single invocation
  and is unaffected. The width and decode contrasts are NOT: each arm is its own
  invocation, so even with a per-stage LRU denominator they are ratios of
  ratios, and the ~1.7% uncertainty enters twice. They must be reported with
  that uncertainty rather than as exact figures.
- **Cross-run ratios inherit ~2% in time, ~0.6% in instructions, ~0.04% in
  traffic.** Everything the write-up leans on clears that by a wide margin:
  transport time 1.243, transport instructions 1.7263, ISA-arm time 1.108. The
  headline "the record is free in bytes" is a traffic ratio of 1.0015 against
  0.04% noise, so it survives comfortably.
- **One number does NOT clear it and is hereby demoted.** The replacement rule's
  time effect of 0.989 is a 1.1% gain, inside the frozen +/-2% tie band and below
  the 3% effect threshold. It must be reported as NO MEASURABLE TIME EFFECT, not
  as a small win. Its traffic effect (0.963, 3.7%) does clear the threshold.
- The frozen +/-2% tie band was chosen a priori. It is now empirically justified:
  the observed placement-driven spread on the primary metric is 1.74%.

### What the decode contrast can and cannot show (2026-07-27)

Adversarial review of the decode milestone found two comparisons that do not
isolate what their names claim. Both are recorded here before any rerun.

**The software-versus-ISA contrast was not matched on helper overhead.** See the
retraction above. Fixed by giving the software path a direct untraced call and
hoisting the trace selection out of the loop, so both arms now execute the same
per-edge scaffolding and differ only in how the record is widened.

**The fused 4-byte versus 8-byte contrast is width PLUS decode, not width.** The
fused property-load family (`ecg.k2.iload`, `ecg.load2`, `ecg.stream.load2`)
accepts only the canonical 64-bit record, so the fused compact arm still widens
in guest software before issuing the load. The 8-byte arm does not. Stages
`40_isa_fused_4b` and `41_isa_fused_8b` therefore compare a compact
implementation against a wide one end to end; they do not price the container in
isolation. A fused instruction that accepts the compact record would close this,
and does not exist yet.

**Execution time from the non-fused stages is not speedup evidence.** The runner
marks the `packed+ecg.extract2` delivery family
`timing_valid_for_speedup=0` with a standing caveat, and that flag is
deliberately NOT relaxed for the compact-ISA arm: `ecg_extract2c` removes the
software widen, but the property load remains a separate instruction rather than
a fused request-bound one, so the delivery is still a prototype. Stages 42, 43
and 44 are evidence about INSTRUCTION COUNTS and TRAFFIC. Their times are
reported for completeness and are not a speedup claim.

The delivery label is now derived from the guest receipt rather than hardcoded:
a cell streaming a 4-byte record and decoding it in the ISA previously recorded
itself as `packed8+k2+ecg.extract2`.

### Decode and width, both matched, after the symmetry fix (2026-07-27)

Re-measured with both arms executing the same per-edge scaffolding. Stages 42,
43 and 44 of `ecg_isa_decode_matrix_20260727_015625`, web-Google-n16 PageRank,
128kB LLC, non-fused `packed + ecg.extract2(c)` delivery.

All three stages are independent invocations and all three produced a
BIT-IDENTICAL LRU cell (40,871,713,000 ticks, 14,211,840 bytes, 11,280,665 ROI
instructions), so the three-way comparison shares one denominator exactly rather
than approximately.

| arm | ROI insts | traffic | time |
|---|---:|---:|---:|
| LRU | 1.0000 | 1.0000 | 1.0000 |
| 4-byte, software widen | 2.6157 | 0.9639 | 1.6140 |
| 4-byte, ISA decode | 1.0775 | 0.9640 | 1.0286 |
| 8-byte, wide record | 1.1730 | 1.1881 | 1.1811 |

**The decode contrast (software versus ISA, identical record).** Instructions
x2.428, traffic x0.9999, time x1.569. The software widen costs 34.5 instructions
per edge and changes nothing whatsoever about what is fetched -- traffic agrees
to four decimal places, which is what a pure decode difference should look like.

**The width contrast (compact versus wide, both decoded in one instruction).**
This is the comparison the earlier matrix could not make, because its compact
arm still widened in software. With `ecg_extract2c` the compact record is
delivered in one instruction, exactly as the 8-byte record is by `ecg_extract2`,
so the arms differ in container width and little else: instructions x0.919,
traffic x0.8114, time x0.871. The compact record wins on all three axes. It uses
FEWER instructions than the wide record because `ecg_extract2c` returns the
destination, so the guest needs no mask.

Against LRU, the compact ISA arm is 1.0775 in instructions, 0.9640 in traffic
and 1.0286 in time.

**These stages are not speedup evidence and the times above are not a speedup
claim.** The runner marks this delivery family `timing_valid_for_speedup=0`
because the property load is still a separate instruction rather than a fused,
request-bound one. The admissible evidence here is the instruction counts and
the traffic; the times are reported for completeness and move in the same
direction.

### Instruction counts are build-sensitive; only within-build comparisons hold (2026-07-27)

The fused 4-byte arm on web-Google-n16 moved from 19,580,752 to 26,297,486 ROI
instructions -- 13.4 more per edge -- between two builds, at the same geometry,
the same delivery (`ecg.k2.iload`, receipt `bytes_per_edge=4.000`), and with the
same LRU denominator to within 0.6%.

Its source path did not change. `git diff` over `bench/src_gem5/pr.cc` between
the two builds shows only additions OUTSIDE the fused loop: the compact-ISA
gating, the fatal-abort branch, the hoisted trace flags, and two new loops that
the fused configuration never enters (it `continue`s first). The remaining
explanation is code generation: the kernel is compiled at `-O1`, and adding
unrelated paths to the same function changes register allocation and inlining in
the hot loop.

The consequence is a rule, not a curiosity. Instruction counts from different
builds are not comparable even when the source path under test is identical, so
no figure from the earlier width matrix may be placed in a table beside a figure
from the decode matrix. This is why the decode profile re-runs every arm --
including the baselines it could have imported -- inside one build and one
profile.

It also means the earlier fused-versus-non-fused comparison (the observation
that the fused 8-byte arm executed FEWER instructions than LRU, 0.939) is a
statement about that build only, and must be re-measured within the current one
before it is used.

### Transport, corrected: the record substitutes everywhere; the DECODE does not (2026-07-27)

An earlier reading of this matrix was going to say the compact record
"substitutes on 2 of 3 graphs and costs 13.7% on cit-Patents". That conflates
two different things and is withdrawn before use.

**The record substitutes structurally on all three graphs.** The compact loop
reads `in_edge_pair32_flat` and returns before `g.in_neigh(u)` is ever touched
(`bench/src_gem5/pr.cc`), and the record count equals the adjacency entry count
on every graph (cit-Patents-n18-sym: 340,054 undirected edges, 680,108 in-edges,
680,108 records). No graph streams both the record and the coordinate.

**What differs is the end-to-end traffic**, measured within-run as ECG:K2_LRU
against each stage's own LRU cell:

| graph | compact 4B (stage 40) | wide 8B (stage 41) |
|---|---:|---:|
| web-Google-n16 | 1.0013 | 1.2357 |
| soc-pokec-n16 | 0.9982 | 1.3567 |
| cit-Patents-n18-sym | 1.1365 | 1.3074 |

The wide record never substitutes -- it is twice the payload, and costs 24--36%.
The compact record is traffic-neutral on two graphs and costs 13.7% on the
third.

**cit-Patents' cost is not the record.** Decomposing that cell, the extra
traffic is essentially all reads (+3,148,096 of +3,161,024 bytes) and equals the
extra LLC misses exactly (+49,189 x 64 B). Misses grow at every level but
disproportionately at the LLC (L1 +4.8%, L2 +6.4%, LLC +15.3%).

Crucially, the per-record cost of the software widen is the SAME on the graph
that pays and the graph that does not:

| graph | extra L1 accesses per record | extra instructions per record |
|---|---:|---:|
| web-Google-n16 | 4.86 | 29.9 |
| cit-Patents-n18-sym | 4.61 | 31.7 |

So the widen executes the same work per edge everywhere; on web-Google its
accesses stay in cache (extra LLC misses: 269) and on cit-Patents they do not
(49,189). Stage 40 widens the compact record in software before issuing the
fused load, because the fused family accepts only the 64-bit record.

#### PRE-REGISTERED, stated before the cell finishes

Stage `43_isa_plain_4b_hardware` on cit-Patents-n18-sym has not run yet. It
removes the software widen and changes nothing else about what is fetched.

- If the widen is responsible, its ECG:K2 traffic should fall from stage 42's
  1.1357 towards ~1.0, and its extra LLC misses should largely disappear.
- If traffic stays near 1.13, the widen is NOT the cause and the compact record
  genuinely fails to substitute at this geometry -- in which case the honest
  claim is that traffic neutrality is graph-dependent and cit-Patents is a
  standing counter-example.

Either outcome is reportable; the prediction is recorded so the answer cannot be
chosen after the fact.

### The replacement-rule figure, stated precisely (2026-07-27)

K2 versus K2_LRU traffic, each against its own stage LRU: web-Google 0.9630,
soc-pokec 0.9594, cit-Patents 0.9993.

Two corrections to how this was going to be reported.

**"Identical instruction counts prove the arms are matched" is circular.** The
two arms run the same guest binary with the same input; only gem5's victim
selection differs, and a replacement policy cannot change how many instructions
the guest retires. Exact equality is therefore guaranteed by construction, not
evidence of anything -- the same 1.0000 appears for GRASP and P-OPT. What the
equality does establish is the weaker and still useful fact that no arm quietly
took a different code path.

**Name the variants, not "the replacement rule".** The contrast is
`epoch_first` (K2's configured rule for PageRank) against `lru_only`, both
inside `ECG_GRASP_POPT` mode, where the ECG policy's touch path updates recency
so `lru_only` reproduces LRU. Outside that mode the touch path is skipped and
`lru_only` would degenerate to FIFO, so this is not a general
"transport with LRU" primitive.

Honest statement: configured epoch-first replacement reduces off-chip traffic
by 3.7% and 4.1% on web-Google and soc-pokec, and by 0.07% -- nothing -- on
cit-Patents, relative to the configured LRU-only rule over identical transport.

**Known gap: the victim variant is requested, not attested.** The runner records
the variant it asked for; gem5 emits the variant name only through a gated trace
that is off in these runs, so nothing in the archived artifacts proves which
rule executed. The archived `config.ini` does prove `GraphEcgRP` and
`ECG_GRASP_POPT`. Closing this needs an ungated one-line receipt from
`ecg_rp.cc`, which requires a gem5 rebuild and is deferred until the running
matrix completes.

### The pre-registered test FAILED: the decode does not explain cit-Patents (2026-07-27)

Stage `43_isa_plain_4b_hardware` on cit-Patents-n18-sym has now run. The
prediction recorded above was that removing the software widen would pull
traffic from 1.1357 towards 1.0 if the widen were responsible. It did not.

| cit-Patents-n18-sym | traffic | ROI insts | LLC misses |
|---|---:|---:|---:|
| stage 42, software widen | 1.1357 | 1.9069 | 1.1519 |
| stage 43, ISA decode | 1.1358 | 1.0287 | 1.1519 |

Removing the widen cut instructions from 1.91x to 1.03x of LRU and moved
traffic by 0.0001 and LLC misses by nothing at all. **The software decode is
excluded as the cause.**

This is also the cleanest available confirmation that decode changes work and
not bytes: a 46% reduction in executed instructions on the same graph left the
memory traffic identical to four decimal places.

So the honest claim is the one pre-registered as the alternative: the compact
record is traffic-neutral on web-Google-n16 and soc-pokec-n16 and costs 13.6% on
cit-Patents-n18-sym, and **cit-Patents is a standing counter-example to traffic
neutrality** rather than an artifact of the prototype delivery.

**Candidate mechanism, NOT established.** cit-Patents-n18-sym is symmetrised,
so the baseline can serve both `g.in_neigh(u)` and `g.out_degree(u)` from one
offsets array, while the record arm must stream its own `pair_off` in addition
to `out_index`. That predicts an extra `(|V|+1)*8/64 = 32,768` lines; the
measured transport cost is 49,189 extra LLC misses (stage 40, ECG:K2_LRU against
its own LRU, which isolates transport from the victim rule). The hypothesis
therefore accounts for about two thirds of the effect and is not sufficient.
On web-Google the same accounting predicts 8,192 extra lines and only 269 are
observed, consistent with that directed graph already needing two offsets arrays
in both arms.

Recording this as open. The mechanism matters for the paper only insofar as the
worst cell must be reported, and it is.

## Decode matrix, complete: 15/15 cells (2026-07-27)

`ecg_isa_decode_matrix_20260727_024739`. PageRank, gem5 RISC-V, three sampled
real graphs, one build, enforcement live (`ECG_EXPECT_BYTES_PER_EDGE` reaches
the guest and aborts pre-ROI), every stage carrying its own LRU cell.

**Decode: software widen versus `ecg.extract2c`, identical record.**

| graph | ROI insts | traffic | time |
|---|---:|---:|---:|
| cit-Patents-n18-sym | x1.854 | x1.0000 | x1.387 |
| soc-pokec-n16 | x2.803 | x1.0000 | x1.768 |
| web-Google-n16 | x2.428 | x0.9999 | x1.569 |
| **geomean (n=3)** | **x2.328** | **x1.0000** | x1.567 |

Software widening costs 2.33x the ROI instructions and moves no bytes at all --
traffic is 1.0000 on every graph. That is the sharpest result in this matrix:
decode is pure work.

**Width: compact versus wide, both delivered in ONE instruction.**

| graph | ROI insts | traffic | time |
|---|---:|---:|---:|
| cit-Patents-n18-sym | x0.908 | x0.8694 | x0.899 |
| soc-pokec-n16 | x0.924 | x0.7418 | x0.841 |
| web-Google-n16 | x0.919 | x0.8114 | x0.871 |
| **geomean (n=3)** | **x0.917** | **x0.8058** | x0.870 |

This is the contrast the earlier width matrix could not make, because its
compact arm still widened in software. With `ecg_extract2c` both records are
delivered by a single instruction, so the arms differ in container width and
little else. The compact record wins on every axis on every graph: 19.4% less
off-chip traffic, 8.3% fewer instructions, 13.0% less time.

**Fused compact versus wide (stages 40/41): x1.725 insts, x0.8060 traffic,
x1.201 time.** Reported separately and NOT as a width contrast, because the
fused property-load family accepts only the 64-bit record, so its compact arm
still widens in software.

**Admissibility.** Stages 42-44 carry `timing_valid_for_speedup=0`: the property
load is a separate instruction rather than a fused request-bound one, so the
times above are context. The admissible evidence is the instruction counts and
the traffic. `claim_gate.json` is unchanged; no claim is promoted here.

## Fused compact property load: correctness gate (2026-07-27)

The fused family previously accepted only the canonical 64-bit record. A
compact arm therefore had to widen in guest software before issuing
`ecg.k2.iload`, so the fused 4-byte/8-byte comparison was width plus decode and
could not price either mechanism cleanly.

`ecg_load_k2_compact` is now a fused indexed property load whose operands are:

- `rs1`: property-array base;
- `rs2`: compact 32-bit Schedule-2 record;
- CSR `0x802` (`ecg.record_format`): loop-invariant
  `id_bits | epoch_bits << 8`.

The format is architectural state rather than simulator-side inference: both
source registers are already consumed by the memory operation, and the field
widths are part of how the instruction decodes its operand. The guest writes the
CSR once before the ROI.

Fail-closed activation is mandatory. `GEM5_ECG_COMPACT_FUSED=1` aborts before
the ROI unless the guest built a compact record and selected indexed
request-bound K2 delivery; it may not silently widen or fall back to another
load.

Correctness probe:
`results/ecg_experiments/probes/fused_compact_trace_20260727_162101`.

- receipt: `record_bytes=4`, `bytes_per_edge=4.000`;
- runtime banner: `[ECG_K2_ILOAD_C] ... ACTIVE`;
- metadata fatals: zero;
- guest expected records: 200;
- decoder records: 200;
- `(seq,dest,tier,epoch1,epoch2)` mismatches: **zero**.

This is mechanism correctness only. Performance must be measured in a new build
with its own LRU and wide-record controls; figures from the completed decode
matrix cannot be imported because instruction counts are build-sensitive.

### Fused compact review corrections and O3 request gate (2026-07-27)

Adversarial review found that the first implementation was safe only in its
narrow PR/TimingSimpleCPU use:

- the runner could label BFS/SSSP/BC/CC compact from the requested environment
  even though only PR implemented the instruction;
- direct `GEM5_ECG_COMPACT_FUSED=1` was scrubbed unless hidden inside the
  explicit-cell channel;
- FUNCT7 values `0x2c`--`0x2f` all decoded as the same instruction because the
  decoder ignored the width subfield;
- the trace probe exercised mailbox delivery, not O3 request binding;
- the archived no-baseline probe row was machine-labelled speedup-valid;
- CSR setup occurred after the stats reset despite prose saying pre-ROI.

All are corrected before performance evidence:

- `--gem5-compact-fused` is an explicit CLI/manifest option and rejects every
  benchmark except PR;
- rows are labelled compact only after the guest emits
  `[ECG_K2_ILOAD_C]`; a missing activation receipt makes the row an error;
- the decoder accepts only FUNCT7 `0x2c` (`ECG_WIDTH=0`) and rejects
  `0x2d`--`0x2f`;
- configuration checks, context setup and CSR writes occur before
  `m5_reset_stats`;
- an invocation without its own LRU cell is fail-closed as
  `timing_valid_for_speedup=0`.

O3 evidence:
`results/ecg_experiments/probes/fused_compact_o3_20260727_165949`.

- 8 request records;
- 7 LLC accepts (one request was satisfied before LLC);
- all 7 accepts have `source=request`, width 4, and exact
  `request_seq`, request destination, fill destination, tier, both epochs,
  current epoch and context;
- request/fill/payload mismatches: **zero**.

O3 may execute and replay custom loads out of program order, so guest EXPECT
sequence numbers are not a stable key. The persistent verifier now keys this
gate by the Request's program-order sequence instead.

The original TimingSimpleCPU probe row retains its archived
`timing_valid_for_speedup=1` value because artifacts are immutable, but that
machine label is superseded by this correction and the no-baseline rule. It is
correctness evidence only.

### Fused compact matrix retraction before reuse (2026-07-28)

The first complete fused compact matrix
(`ecg_fused_compact_matrix_20260727_181928`) is **not valid for decode or width
attribution**.

Stage 50 used a dedicated branch-free compact loop. Stages 51 and 52 used a
shared generic loop that reloaded invariant flags and widening parameters from
the stack and evaluated four loop-invariant branches on every edge. Relative to
stage 50, the wide stage executed 1.32--1.58x as many L1 accesses and incurred
1.17--1.36x as many LLC misses. Those effects directly contaminate instructions,
traffic and time.

Therefore:

- `software widen / compact K2-I = x2.434 instructions, x1.515 time` measures
  software widening **plus generic-loop overhead**;
- `compact / wide = x0.681 instructions, x0.8056 traffic, x0.787 time` compares
  a dedicated compact implementation against a branch-heavy generic wide
  implementation, not record width.

Both interpretations are retracted. The controls now use dedicated loops with
the same skeleton as stage 50; only record load/decode/opcode differ, and the
matrix must be rerun in the new build.

Three qualifications also become binding:

1. Scale runs use single-core TimingSimpleCPU serialized mailbox equivalence.
   Exact request binding is proven separately by the O3 micro-probe; scale rows
   must not be called request-bound.
2. The compact instruction's dynamic CSR read, masks and shifts are modeled as
   one custom memory instruction. Timing is an idealized ISA implementation
   point, not a hardware critical-path proof.
3. The old matrix had no archived PageRank output checksum. New runs emit a
   post-ROI receipt with iterations, semantic edge count and FNV checksum, and
   the runner fails all rows if policies disagree.

The stage-50 whole-system observation from that build remains descriptive:
geomean time tied LRU, traffic was 1.6% higher, and cit-Patents was the
+3.2% time/+13.6% traffic worst cell. It cannot be mixed with the corrected
controls because the guest build changes.

## FROZEN: public P-OPT artifact direction on a Pin 4.2/GCC 13 compatibility port (2026-07-28)

The public P-OPT PageRank artifact was rebuilt as a provenance-bound Pin 4.2
compatibility port and run once from a fresh, non-resumed output directory.
The run uses one PageRank sweep, the artifact's 24 MiB/16-way LLC, no
prefetching, disabled ASLR, and raw demand LLC misses.

| graph | DRRIP misses | P-OPT misses | P-OPT reduction |
|---|---:|---:|---:|
| UK-02 | 62,496,506 | 56,011,879 | 10.38% |
| HBUBL | 54,500,087 | 35,956,700 | 34.02% |
| KRON25 | 27,829,418 | 25,206,516 | 9.42% |
| URAND25 | 115,631,617 | 85,352,434 | 26.19% |

P-OPT has fewer misses than DRRIP on all four graphs. The geometric mean of
P-OPT/DRRIP is **0.79294**, or **20.71% fewer demand LLC misses**. All 12
LRU/DRRIP/P-OPT rows exit normally, contain exactly one explicit ROI statistics
block, and report matching PageRank error across policies for each graph.

This confirms the **qualitative P-OPT-over-DRRIP miss direction for this
compatibility configuration**. It is not an exact Pin 2.14/GCC 6.3
reproduction: applications were rebuilt with GCC 13, which is recorded as a
compatibility deviation. The magnitude is therefore not compiler-independent.
It is also not a speedup result, P-OPT-vs-GRASP evidence, validation of
GraphBrew's internal P-OPT model, or a K2-vs-P-OPT ranking.

Any earlier **28.18%** figure from the manual early-exit port is withdrawn. That
run was not bound to the accepted build receipt and terminated before the
application correctness receipt; its differing binaries cannot support a
compiler-effect claim.

The machine-readable owner is
`research/ecg-hpca/evidence/popt_public_direction_20260728.json`. The preserved
content-addressed archives have SHA-256
`b89caca9d4d9e8c78e1baeb1d69e040b6cfa43d4aacc33d1d0128956ee9156d9`
(v5 build) and
`2ce20f47ecab244400448937501451cba1fcaae4168c2db40f4c1923c22b15f3`
(completed run).

## RETRACTION: first matched-loop fused matrix used a stale DBG guest (2026-07-29)

`ecg_fused_compact_matrix_final_20260728` completed 27/27 rows and passed its
cross-stage PageRank checksum gate, but it is **not final evidence**.

Every job requested DBG ordering (`-o 5`). The tracked source had already been
corrected to compute average degree from directed adjacency entries, while the
prebuilt RISC-V guest still divided the undirected half-edge count by the node
count. On `cit-Patents-n18-sym`, the guest therefore used average degree 1
instead of 2. That can change the vertex mapping and every cache result for the
cell that defines the aggregate worst case.

The hot-loop matching, semantic receipts, variant receipts, compact activation,
and arithmetic all passed for that stale binary. They do not repair the
source/binary mismatch. Do not cite its 0.923 compact-K2/LRU time ratio, 0.8056
compact/wide traffic ratio, or any three-graph aggregate.

The RISC-V guest now builds under the receipt tool itself. It scans and hashes
the project and system dependency set before compilation, compiles into
temporary outputs, checks that git/compiler/input state did not change, then
atomically publishes the binary, depfile, and receipt. The accepted toolchain is
the pinned RISC-V compiler and its assembler, linker, specs, CRT objects, and
static libraries. A pinned syscall trace inventories every file opened by two
identical compile/link passes, including indirect specs, plugins, response
files, linker scripts, and archives. The final compiler pass sees those captured
bytes only through an in-memory read-only FUSE snapshot inside a restricted
proot filesystem; transient changes to the original pathnames cannot alter the
build. PRoot itself, its loader/libc/talloc runtime, fusepy, and libfuse are
hash-pinned and loaded from verified bytes or sealed file descriptors under an
allowlisted environment. The receipt binds the exact target/source, build
config, compiler command, linked m5 library, headers (including
`reorder_hub.h`), virtual aliases, and guest hash.

Before execution, `roi_matrix.py` copies the validated guest to a
content-addressed read-only path. Each invocation serves the guest and graph
from read-only in-memory FUSE files under their original names; this preserves
the guest's resolvable `/proc/self/exe` identity and the graph's `.sg` suffix.
The gem5 executable itself is inherited as a sealed memfd, while its Python
config modules are served read-only. Missing receipts, changed dependencies,
copied kernel receipts, inconsistent ISA overrides, or staged binary changes
fail closed. All nine jobs must be rerun from one newly built guest.
