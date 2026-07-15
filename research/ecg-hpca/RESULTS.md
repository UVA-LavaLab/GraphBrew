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

Canonical replacement profile: `ecg_cache_sim_factorial`. A fresh rerun uses
tag-hit-preserving StreamShield and the current size-correct charged P-OPT model.

## Tiered K2 and online-selection baseline (pending)

The new `ecg_replacement_baseline` profile compares uncharged and charged P-OPT,
K1, every static tiered-K2 arm, and `ECG:K2_ONLINE` on
PR/BFS/SSSP/BC/CC. The pipeline emits `online_dueling_regret.csv`. No numerical
online-selection claim is frozen until this complete real-graph profile
finishes.

The five-algorithm Schedule-2 mechanism matrix passes in cache_sim, gem5, and
Sniper with zero K2 distance mismatches. This is mechanism/spec evidence, not a
frozen real-graph performance ranking.

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

The cache_sim column uses the fused widened-record model and emphasizes
replacement behavior. gem5 and Sniper include explicit record-delivery
accesses. Sniper ranks are cache-mechanism diagnostics only: policy-specific
delivery changes the instruction stream, and rows without validated fused
receipts are excluded from speedup claims.

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
