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
(`CACHE_STREAM_PREFETCH_MODEL=stride`). It sees addresses and nothing else, as
hardware does: a stream must be confirmed by two consecutive ascending line
accesses within a 4 KiB region before it issues, any non-sequential step breaks
confirmation, and issue is bounded by a finite in-flight budget (32 by
default). It trains on regular property accesses and wastes fills on irregular
ones, which is precisely the mistake the oracle could not make. The oracle
remains available as an explicitly labelled upper bound.

web-Google PageRank, `-i 2`, 2 MiB 16-way, degree 8, charged P-OPT with its
matrix stream simulated:

| model | prefetches issued | demand misses | prefetch fills | total traffic |
|---|---:|---:|---:|---:|
| oracle (upper bound) | 138,305,632 | 1,647,524 | 1,551,347 | 3,198,871 |
| stride (honest) | 3,283,159 | 2,187,006 | 1,001,928 | 3,188,934 |

Two results.

1. **The oracle's coverage was substantially an artifact.** Demand misses rise
   **32.8%** when the prefetcher has to detect the stream instead of being told
   where it is. So the Phase 1a observation that "the prefetcher covers the
   whole matrix stream" describes the oracle, not a plausible prefetcher.
2. **The oracle also issued 138.3 million prefetch requests**, 42x the honest
   model, at no modelled bandwidth or queue cost. A component that can request
   42x the traffic for free will make any policy that streams metadata look
   cheap.

For LRU on the same cell the two models nearly agree (demand 3,276,088 oracle
against 3,316,509 stride, +1.2%), which is the expected sanity check: the CSR
edge stream really is sequential, so an honest detector finds it. The gap opens
specifically where the oracle's semantic knowledge was doing work.

Note once more which metric is stable. Total traffic differs by 0.3% between
the two models (3,198,871 against 3,188,934) while demand misses differ by
32.8%. A prefetcher relocates work rather than removing it, so traffic barely
notices which prefetcher is used, and the demand-miss column swings wildly.
That is the third independent confirmation of the frozen primary metric.

Rows record `stream_prefetch_model`, `stream_prefetch_issued`,
`stream_prefetch_throttled` and `stream_prefetch_untrained`, so an oracle
result cannot be mistaken for an honest one.

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
