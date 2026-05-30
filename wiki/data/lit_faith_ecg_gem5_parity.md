# ECG substrate-parity audit (gem5)

Gate 239 — ECG-Gem5-Parity. Locks the POPT-arm faithfulness of
the ECG substrate under cycle-accurate gem5 timing. Sibling to
gate 238 (cache_sim ECG-Parity). DBG arm + PFX activation are
explicitly out of scope today (see generator docstring).

## Rules
- **G1** — every required policy present with status=ok per (benchmark, section, l3_size) cell
- **G2** — |miss_rate(ECG_POPT_PRIMARY) - miss_rate(POPT)| <= 0.002
- **G3** — every row has backend=gem5 AND simulator=gem5 (no silent cache_sim ingestion)
- **G4** — sim_ticks >= 1 AND ipc > 0.0 on every row
- **G5** — LRU baseline has strictly positive l3_accesses and l3_misses on every cell
- **G6** — l3_misses <= l3_accesses and l3_miss_rate in [0,1] on every row
- **G7** — distinct sections present >= 2 (cold + re-warmed)

## Constants
- ε(POPT parity): `0.002`
- section floor: `2`
- sim-tick floor: `1`
- IPC floor: `> 0.0`
- required policies: `ECG_POPT_PRIMARY, LRU, POPT`
- baseline policies: `LRU`

## Totals
- observations: **12**
- cells (benchmark × section × L3): **4**
- benchmarks: `pr`
- backends: `gem5`
- sections: `1, 2`
- policies present: `ECG_POPT_PRIMARY, LRU, POPT`

## POPT parity (gem5)

| benchmark | section | L3 | POPT | ECG_POPT_PRIMARY | |Δ| |
| --- | ---: | --- | ---: | ---: | ---: |
| pr | 1 | 16MB | 0.062198 | 0.061111 | 0.0010870000000000046 |
| pr | 1 | 32MB | 0.062198 | 0.061111 | 0.0010870000000000046 |
| pr | 2 | 16MB | 0.111066 | 0.110745 | 0.00032100000000000184 |
| pr | 2 | 32MB | 0.111066 | 0.110745 | 0.00032100000000000184 |

## Violations

_None._
