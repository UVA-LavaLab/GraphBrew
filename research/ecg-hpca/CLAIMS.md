# Claim Ledger

## Submission eligibility gate

- The IPDPSW 2024 ECG paper is archival prior work.
- The HPCA paper must be demonstrably non-substantially-similar.
- PC-chair guidance is required before abstract registration.
- A new name improves differentiation but does not replace disclosure.

## Proven

- Tiered K2 construction, delivery, line metadata, effective distance, and
  victim decisions agree across cache_sim, gem5, and Sniper for
  PR/BFS/SSSP/BC/CC.
- All five kernels execute the fused indexed K2-I instruction in gem5.
  TimingSimpleCPU provides serialized semantic delivery for the scale cells;
  exact per-Request O3 binding is proven only by tiny PR and weighted SSSP
  cells. Sniper uses the idealized packed sideband model, and cache_sim remains
  the functional reference. The post-correction 15-cell gate passes with 32/32
  detailed-simulator deliveries and zero K2 distance mismatches per kernel.
- Typed computed-address K2-M modes are implemented in gem5 for U32, S32, U64,
  compact weighted U32, and FP32. PR and compact SSSP pass exact O3
  producer/consumer request-binding proofs; PR/BFS/SSSP/BC/CC pass the complete
  cache_sim+gem5 K2-M delivery/victim gate. The current-epoch CSR remains
  pending, so these runs still use the prototype vertex channel.
- The transport-matched Sniper K2-M model passes exact instruction parity on
  PR/BFS/SSSP/BC/CC. On the email mechanism cells, K2/LRU instruction ratio is
  1.000x, geomean speedup is 1.006x, and geomean L3-miss reduction is 4.35%.
  PR/BFS/SSSP improve; BC/CC remain near neutral. PR and compact SSSP receipt
  proofs report zero bad records. These are mechanism cells, not headline timing.
- The algorithm mapping is PR=`epoch_first`, BFS/SSSP=`degree_first`, and
  BC/CC=`rrip_first`. BC covers its forward static-edge phase; CC is
  undirected/symmetric only.
- The 64-bit record carries an order-independent hottest-per-line GRASP tier and
  two 15-bit epochs; the real RISC-V decoder round-trips all fields.
- Five-arm online set dueling is live in cache_sim, gem5, and Sniper without a
  benchmark-name decision.
- In the complete cache_sim real-graph replacement profile, every static arm is
  best on at least one cell and online K2 is within 0.26% geomean LLC misses of
  the per-cell best static arm while beating it on 8/15 cells.
- StreamShield is request-bound in gem5 and preserves normal L1/L2 behavior and
  LLC hits; only LLC miss allocation is suppressed.
- The static StreamShield primitive passes the full PR/BFS/SSSP/BC/CC
  cache_sim/gem5/Sniper mechanism gate; adaptive eligibility remains separate.
- Allocate-vs-StreamShield placement dueling is live in all three simulators and
  passes the full five-kernel mechanism gate. In cache_sim on the three PR
  graphs it reduces misses 4.28% versus always-allocate online K2, while paying
  2.62% regret versus always-shield.
- The completed all-kernel placement matrix finds static StreamShield better
  than always allocate on 15/15 cells. Adaptive placement has 0.95% geomean
  regret versus static SS and remains a default-off ablation, not a headline
  mechanism.
- Sniper preserves NUCA lookup, hits, and latency while suppressing insertion of
  bypassed misses.
- On synthetic mechanism cells, StreamShield improves fused K2 in both gem5 and
  Sniper.
- The 360-row sampled cache_sim/gem5/Sniper matrix passes strict coverage and
  transport gates. It corroborates SRRIP/GRASP direction but places every K2
  variant in a net-negative small-graph overhead regime; it is not headline K2
  performance evidence.
- In the corrected cache_sim real-graph factorial, StreamShield adds 6.73%
  geomean demand-miss reduction and 3.23% traffic reduction beyond online K2;
  full ECG beats charged P-OPT demand misses by 31.48% but uses 5.28% more
  traffic.
- In the corrected nine-row sampled PageRank Sniper profile, fused
  K2-online+StreamShield reaches 1.207x geomean speedup over GRASP and 1.150x
  over capacity-charged P-OPT. The packed traversal executes 4.8% fewer
  instructions; ticks-per-instruction still improve 1.149x and 1.094x.
  Total LLC misses rise 8.50% versus GRASP and 14.03% versus P-OPT after its
  matrix-stream charge, while non-record misses fall 35.92% and 32.65%.
  This pre-surgical attribution profile is superseded for timing by the current
  all-kernel matrix.
- Full web-Google warm-SIFT LRU and K2 both reach and complete a 100K detailed
  ROI after CACHE_ONLY queue/shared-memory timing is suppressed. Cache warming
  remains enabled and K2 context delivery is active.
- The post-fix 600M-capped web-Google PR K2 cell completes successfully with
  normal warming and context delivery. It finishes the full iteration before
  the cap at 179.4M reported instructions; the row is cache evidence only.
- The corrected 120-row sampled Sniper idealized packed-record K2-I-like model
  matrix passes strict coverage.
  K2-online+StreamShield reaches 1.792x on PR, 1.675x on BFS, 1.145x on SSSP,
  1.082x on BC, and 1.115x on CC versus LRU. Final sampled geomean is 1.329x,
  ahead of GRASP's 1.100x and charged P-OPT's 1.082x. Its 0.881x instruction
  ratio includes indexed/packed-loop savings and is not measured K2-I ISA
  timing or a core K2-M result. Its TPI does not isolate K2-M.
- The current packed K2-I-like extension model is strong on PR/BFS and positive
  on sampled SSSP/BC. Its model result survives removal of the shortest BFS cell. That
  exclusion leaves 1.276x for K2-online+StreamShield versus 1.107x for GRASP.
  Cit-Patents SSSP remains the principal negative cell, and CC remains slightly
  behind GRASP. Sniper's CPI components remain unavailable beyond total ticks
  per instruction.
- Full-graph cache_sim evidence argues against a size-only cit-Patents
  replacement failure: compact K2-online+StreamShield reaches 0.3943 L3 miss
  rate versus 0.5630 LRU, 0.4192 GRASP, and 0.4330 charged P-OPT. The sparse
  n18 sample's longer paths and near-zero neighbor overlap are a plausible
  explanation; full-graph Sniper timing remains unmeasured.

## Pending

- A complete real-graph Sniper comparison of LRU, SRRIP, GRASP, charged P-OPT,
  static/online K2, and both StreamShield variants.
- A full sampled/real-graph matched Sniper K2-M timing matrix.
- An explicit current-epoch CSR/request channel replacing prototype magic.
- Equal-area K2 metadata, logic, energy, and replacement-latency accounting.
- K2-M versus K2-I disassembly and retired-instruction categorization.
- A bounded Sniper structure-prefetch configuration that does not reproduce the
  generic simple prefetcher's 9x--596x LLC-read traffic expansion.
- Final normalized performance, LLC, traffic, and hardware-overhead paper tables.
- Detailed-simulator confirmation of the real-graph online-selector result.
- Completion and aggregation of the now-runnable full-graph 600M SIFT matrix.
- An optional zero-record GRASP ablation to isolate mask-stream cost; it is not
  required for masked-load correctness. Cit-Patents SSSP still trails GRASP
  despite the compact replacement record.

## Prohibited until the pending gate passes

- “The ECG successor generally beats P-OPT on full graphs in gem5 and Sniper.”
- “The synthetic kron mechanism cell ranks the policies.”
- Comparing absolute gem5 and Sniper miss rates.
- Treating cache_sim timing as a paper performance result.
- Presenting aggressive per-access stored refresh as hardware-free.
- Presenting the 1.329x packed K2-I-like model result as measured K2-I or K2-M speedup.
- Presenting the 1.171x model TPI as a K2-M estimate.
- Claiming zero K2 hardware overhead from zero reserved data ways.
