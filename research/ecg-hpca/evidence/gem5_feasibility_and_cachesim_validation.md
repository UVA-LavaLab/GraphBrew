# gem5 graph-size feasibility + the cache_sim↔gem5 validation that de-risks final runs

**Date:** 2026-07-03
**Prompted by:** "largest graph feasible on gem5 in a reasonable time?" + "weeks OK for
final runs, but I need valid results first — no surprises."

## 1. Measured gem5 runtime (PR, 1 iteration, fast policy LRU/GRASP/POPT)

gem5 = TimingSimpleCPU, **no fast-forward** → simulates the whole binary (load + build +
kernel). Timed points this session:

| graph | vertices | edges | gem5 wall-clock | note |
|---|---|---|---|---|
| email-Eu-core | 1 K | ~25 K | 76 s | fixed floor dominates |
| kron_s16_k4 | 65 K | 494 K | 237 s | |
| kron_s17 | 131 K | ~2 M | 1110 s | |
| **web-Google** | **916 K** | **~5.1 M** | **~38 min** (2270 s) | real graph, 2MB L3 |

**Model:** `wall ≈ 65 s + edges / ~1900` (≈1,700–2,300 edges/sec; web-Google ran a bit
faster per edge than the synthetic floor). ROI itself is small (web-Google ROI hostSeconds
= 264 s / 4.4 min); the rest is simulated graph build. `hostInstRate` drops from ~17M
(small) to ~11.9M insts/s (web-Google) — the host slows on bigger working sets, so treat
the model as slightly conservative-to-optimistic near the top.

### Feasibility tiers (fast policy, PR ‑i1)
| budget | max edges | example |
|---|---|---|
| ~10 min | ~1 M | kron_s17-class |
| ~40 min | ~5 M | **web-Google (largest real graph feasible for a quick cell)** |
| ~5 h | ~30 M | soc-pokec — one-off only |
| ~1 day | ~70 M | kron-s22 — a few fit in a weeks budget |

Multipliers: **ECG mode-6 ≈ 5× slower** (÷5 on max edges); BFS/SSSP single-pass ≈ one PR
iteration; cache studies use ‑i1 (P-OPT paper simulates 1 PR iteration).

**Conclusion:** with a weeks budget, gem5 large-graph *cycle-accurate* runs (incl. the
RISC-V `ecg.extract` path) are achievable up to ~kron-s22 scale — the constraint was
iteration speed, not a hard ceiling.

## 2. The de-risking result: cache_sim ≈ gem5 at sensible cache sizes

The core question for committing weeks of compute: **does the cheap simulator predict the
expensive one?** Measured on web-Google @ 2MB/16-way, PR ‑i1, LRU:

| simulator | L3 miss rate |
|---|---|
| gem5 (TimingSimpleCPU, ROI) | **0.3244** |
| cache_sim (functional authority) | **0.3241** |

**They agree to <0.1pp on a real graph at a realistic cache size.** Contrast the synthetic
128kB cell, where gem5 (0.6447) and cache_sim (0.6606) differed by ~1.6pp — that gap is the
**inclusion artifact** (gem5 L3=mostly_incl must duplicate the 112kB inner cache; at 128kB
that costs ~1.6pp, at 2MB it is negligible: 112kB ≪ 2MB → they converge).

**Implication (this is the de-risking lever):** at the sensible cache sizes final runs use,
the gem5↔cache_sim delta is ~0. So cache_sim — which runs *every* graph cheaply — can
**pre-register** the gem5 result to <0.1pp. The weeks-long gem5 run then *confirms* a
prediction rather than *discovering* a number. A surprise is only possible if the delta
becomes size-dependent, which is directly testable (it did NOT here — it shrank with size,
the safe direction).

## 3. Pre-flight gate before committing weeks (no-surprise protocol)
1. **Freeze + tag config** (mostly_incl, cpu.data demand-mr, setarch -R, size_correct
   reserved-way, section-1 selection).
2. **Mechanism-liveness gate:** short prefix confirms `curEpoch>0` (epoch live) + eviction
   trace shows ECG acting + direction agrees vs LRU. Never spend weeks on an inert cell.
3. **Pre-register:** cache_sim predicts each cell + (near-zero) gem5 delta → write the
   expected gem5 number before launch.
4. **Delta-stability:** confirm the gem5↔cache_sim offset stays ~0 across feasible sizes
   (email→kron_s17→web-Google: it did, and shrank at scale).
5. **Canary + early-abort:** launch one large cell first, dump per-epoch partial ROI stats,
   gate the full matrix on the canary matching prediction.
6. **Determinism:** setarch -R + fixed seed + single-thread → bit-identical reruns.

## 4. RISC-V `ecg.extract` sell (why it has almost no surprise surface)
- **Cost is per-invocation and constant** (latency/energy/area) — measure once on a small
  gem5-RISC-V kernel; size-independent by construction.
- **Benefit = CHARGED→UNCHARGED delivery gap × edges** — measured at scale in cache_sim /
  Sniper (the gap the instruction removes by riding the demand vs a software 8B-record read).
- **Large gem5-RISC-V finals confirm the composition** — a fixed-latency op can only surprise
  on the workload, which cache_sim already pins to <0.1pp (§2). Characterize op cost
  cycle-accurately (small), workload benefit functionally at scale, confirm with pre-registered
  large-graph cycle-accurate finals.
