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
web-Google Sniper matrix. Results are added here only after every required policy
finishes under the same bounded ROI.
