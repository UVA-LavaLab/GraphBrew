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
- All five kernels pass the fused three-simulator gate: gem5 uses real RISC-V
  `ecg.load2`, Sniper uses the fused record sideband model, and cache_sim remains
  the functional reference.
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

## Pending

- A complete real-graph Sniper comparison of LRU, SRRIP, GRASP, charged P-OPT,
  static/online K2, and both StreamShield variants.
- A bounded Sniper structure-prefetch configuration that does not reproduce the
  generic simple prefetcher's 9x--596x LLC-read traffic expansion.
- Final normalized performance, LLC, traffic, and hardware-overhead paper tables.
- Detailed-simulator confirmation of the real-graph online-selector result.
- Port/build of Sniper's disabled original Pin frontend, or a fix for warm SIFT
  K2's `queue_model_history` abort, before the full-graph 600M matrix can run.
- Request-bound K2 pair delivery before gem5 O3 is enabled.

## Prohibited until the pending gate passes

- “The ECG successor beats P-OPT in gem5 and Sniper.”
- “The synthetic kron mechanism cell ranks the policies.”
- Comparing absolute gem5 and Sniper miss rates.
- Treating cache_sim timing as a paper performance result.
- Presenting aggressive per-access stored refresh as hardware-free.
