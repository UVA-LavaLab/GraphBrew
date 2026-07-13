# Claim Ledger

## Submission eligibility gate

- The IPDPSW 2024 ECG paper is archival prior work.
- The HPCA paper must be demonstrably non-substantially-similar.
- PC-chair guidance is required before abstract registration.
- A new name improves differentiation but does not replace disclosure.

## Proven

- K2 construction, delivery, line metadata, effective distance, and victim
  decisions agree across cache_sim, gem5, and Sniper for PR and BFS.
- StreamShield is request-bound in gem5 and preserves normal L1/L2 behavior and
  LLC hits; only LLC miss allocation is suppressed.
- Sniper preserves NUCA lookup, hits, and latency while suppressing insertion of
  bypassed misses.
- On synthetic mechanism cells, StreamShield improves fused K2 in both gem5 and
  Sniper.

## Pending

- A complete real-graph Sniper comparison of LRU, SRRIP, GRASP, charged P-OPT,
  K2, and K2+StreamShield.
- A fresh real-graph cache_sim factorial using the current tag-hit-preserving
  StreamShield semantics. The legacy 77.3%/22.7% attribution used full LLC
  lookup bypass and is not a current paper claim.
- Final normalized performance, LLC, traffic, and hardware-overhead paper tables.
- Request-bound K2 pair delivery before gem5 O3 is enabled.

## Prohibited until the pending gate passes

- “The ECG successor beats P-OPT in gem5 and Sniper.”
- “The synthetic kron mechanism cell ranks the policies.”
- Comparing absolute gem5 and Sniper miss rates.
- Treating cache_sim timing as a paper performance result.
- Presenting aggressive per-access stored refresh as hardware-free.
