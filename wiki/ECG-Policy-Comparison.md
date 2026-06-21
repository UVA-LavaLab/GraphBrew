# ECG Policy Comparison & Verification

This page is the reproducible artifact for the ECG L3 replacement study: how to
run the policy comparison, verify each policy is implemented correctly, and the
headline result. Everything here is regenerable from two scripts.

- Run the matrix: `scripts/experiments/ecg/ecg_variant_matrix.py`
- Verify correctness: `scripts/experiments/ecg/verify_ecg.py`

## The policy set

ECG is an **L3-only** replacement policy: **GRASP insertion** (degree-tiered RRIP)
+ **RRIP eviction** + a **per-edge next-reference epoch tiebreak**. The epoch is
meaningful only for *property* lines (scores/contrib); edge-stream/record lines
carry no usable epoch. We compare against the literature baselines it combines:

| Policy | One-line definition | Source |
|--------|--------------------|--------|
| LRU    | Least-recently-used | baseline |
| GRASP  | Degree-tiered RRIP insertion + RRIP eviction | Faldu et al., HPCA'20 |
| P-OPT  | Transpose-Belady oracle: non-property first → rereference distance → RRIP | Balaji et al., HPCA'21 |
| ECG    | GRASP insertion + RRIP eviction + property-epoch tiebreak | this work |

## The factorial framework (`ECG_VARIANT`)

ECG is one policy (`ECG_MODE=ECG_GRASP_POPT`) with one eviction lever switched by
the `ECG_VARIANT` environment variable, so each column isolates exactly one
mechanism. This is what lets a reviewer attribute the result to a specific lever
rather than to an opaque bundle.

| `ECG_VARIANT` | Eviction rule | Isolates |
|---------------|---------------|----------|
| `grasp_only`   | RRIP max-RRPV only (no epoch) | the GRASP arm alone (≈ GRASP) |
| `epoch_only`   | Farthest-epoch among stamped property, else recency | the P-OPT-style epoch arm alone (≈ P-OPT) |
| `rrip_first`   | Max-RRPV set; records first, then farthest-epoch property | recency vetoes epoch |
| `epoch_first`  | Records by recency, then farthest-epoch property (epoch vetoes RRPV) | epoch vetoes recency |
| `shortcircuit` | Evict any non-property line first, then farthest-epoch property | property-protection (legacy) |

## Methodology: run BOTH `-o 0` and `-o 5`

GRASP's degree-tiered insertion and ECG's degree arm assume a **degree-sorted
layout** (hot = low address), which is produced by `-o 5` (Degree-Based Grouping,
DBG). `-o 0` (original order) handicaps GRASP/ECG. The matrix therefore reports
every cell at both orders; the **fair comparison is at `-o 5`**.

## Headline result (cache_sim, PageRank)

L3 miss-rate at the literature geometry (16-way L3, 32 kB L1 / 256 kB L2, `pr -i 1`)
at the fair `-o 5` (DBG) baseline, lower is better. **On power-law graphs the best
ECG variant beats GRASP and beats or ties P-OPT.**

| graph (L3) | topology | LRU | GRASP | P-OPT | ECG (best variant) |
|------------|----------|----:|------:|------:|-------------------:|
| web-Google (512 kB) | power-law | 0.844 | 0.673 | 0.633 | **0.623** (shortcircuit) |
| cit-Patents (1 MB)  | power-law | 0.896 | 0.820 | 0.747 | **0.680** (shortcircuit) |
| soc-pokec (1 MB)    | power-law | 0.694 | 0.556 | 0.551 | **0.499** (shortcircuit) |
| com-orkut (2 MB)    | power-law | 0.552 | 0.470 | 0.446 | **0.447** (epoch_only, ties P-OPT) |
| roadNet-CA (512 kB) | road/mesh | 0.966 | 0.986 | **0.935** | 0.968 (out of domain) |

The `epoch_only` arm alone already matches or beats P-OPT (e.g. cit-Patents
0.686 vs 0.747), confirming the epoch carries the P-OPT-class signal; `grasp_only`
tracks GRASP. **Scope (honest):** the win is a power-law phenomenon. On **roadNet-CA**
reuse is spatial, not property-driven — P-OPT's oracle wins, ECG ≈ LRU, and GRASP's
degree bias actively *hurts*. ECG is for scale-free analytics, not mesh traversal.
The same `-o 0` rows (where GRASP/ECG insertion is handicapped) are in
`/tmp/ecg_matrix.md` after a run.

## The best variant is model-dependent (honest finding)

No single variant wins on every simulator, and the factorial framework is what
exposed this:

- **cache_sim** isolates the L3 on a fixed stream; its L2 absorbs the edge stream
  so the L3 set is property-dominated → `shortcircuit` (evict non-property first)
  is best.
- **gem5** is a faithful inclusive hierarchy where the edge stream reaches the L3 →
  `shortcircuit` lets no-reuse records displace reused property and *underperforms*;
  `rrip_first` (recency vetoes, records-first) is the correct default there.

The production ECG policy is therefore **GRASP-insert + RRIP-evict + epoch-tiebreak
(`rrip_first`)** — robust in both models — with `shortcircuit` documented as a
cache_sim-favorable, gem5-pathological variant.

## Three-simulator status

The same policy + `ECG_VARIANT` factorial lives in all three simulators, sharing
the spec but using each model's native machinery:

| Simulator | ECG_GRASP_POPT + ECG_VARIANT | Epoch source | Status |
|-----------|------------------------------|--------------|--------|
| **cache_sim** | yes | per-edge memory-resident mask (0 LLC ways) | **verified** (`verify_ecg.py`, 7×40/40) + matrix |
| **gem5** | yes | per-edge mask via ISA `ecg.extract` | **verified** (`verify_ecg.py --gem5`, 5×40/40) |
| **Sniper** | yes | native `findNextRef` (P-OPT next-ref) | **verified** (`verify_ecg.py --sniper`, 4×40/40, guarded) |

Sniper's variant uses its native `findNextRef` for the property next-reference
distance (so it isolates the same eviction *levers* rather than the per-edge
mask). Real-graph Sniper runs are gated behind `--allow-sniper-sg-kernel-workload`
because the Sniper/SDE frontend has a documented ~50 GiB memory runaway
(infrastructure, independent of the ECG policy); run it under
`--sniper-memory-limit-gb`.

## Reproduce

```bash
# 1. build cache_sim
make sim-pr

# 2. run the full factorial matrix (rows=graphs, cols=variants), both orders
python3 scripts/experiments/ecg/ecg_variant_matrix.py --suite cache-sim \
    --cells "web-Google:512kB cit-Patents:1MB soc-pokec:1MB com-orkut:2MB roadNet-CA:512kB" \
    --orders "0 5" --out /tmp/ecg_matrix.md

# 3. same framework on gem5 (RISC-V backend; rrip_first is the gem5 default)
python3 scripts/experiments/ecg/ecg_variant_matrix.py --suite gem5 \
    --cells "email-Eu-core:16kB" --orders "5"
```

## Verify (correctness, not aggregates)

`verify_ecg.py` runs each policy with `ECG_EVICT_TRACE` enabled, parses every
eviction (each way's rrpv/epoch/dist/property/recency + the chosen victim), and
**asserts the victim matches that policy's defining rule**. Exit 0 iff every
eviction of every policy obeys its spec — no trust in aggregate miss-rates needed.
The same `[EVICT L3 ...]` trace format is emitted by all three simulators, so one
checker verifies every backend.

```bash
make sim-pr
python3 scripts/experiments/ecg/verify_ecg.py            # cache_sim: 7 policies
python3 scripts/experiments/ecg/verify_ecg.py --gem5     # + gem5 ECG variants
python3 scripts/experiments/ecg/verify_ecg.py --sniper   # + Sniper ECG variants (guarded, prlimit)
# expected: each policy "N/N evictions obey spec [OK]" → "ALL POLICIES VERIFIED ✓"
```

Verified to date: **cache_sim** (7 policies × 40/40 evictions), **gem5** (5
ECG_GRASP_POPT variants × 40/40), and **Sniper** (4 ECG-specific variants × 40/40;
the guarded `sg_kernel` run is memory-capped via `--sniper-memory-limit-gb`). All
three emit the same `[EVICT L3 ...]` trace, so one checker verifies every backend.
Larger real-graph Sniper runs may still hit the documented SDE memory behavior;
the tiny email-Eu-core verification run completes cleanly under the cap.

## Related

- [[Cache-Simulation]] — simulator architecture, all `ECG_MODE` values, env vars
- [[ECG-Final-Runs]] — gem5/Sniper final-run profiles, charged P-OPT, topology caveats
- [[Baseline-Literature-Faithfulness]] — GRASP/P-OPT faithfulness audit
