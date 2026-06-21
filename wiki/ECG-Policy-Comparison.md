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

The same policy + `ECG_VARIANT` factorial lives in all three simulators. The
eviction **decision** is a single shared function, `ecg_policy::selectVictim`
(`bench/include/ecg_victim_policy.h`), that cache_sim, gem5 and Sniper all call —
so nothing is "ported" or "mirrored": the decision logic is literally identical
across backends (a SSOT test asserts the co-located copies are byte-identical),
and one unit test of that function verifies the eviction choice for all three.
Each simulator keeps only a thin adapter that builds the per-way state from its
native cache lines (epoch via memory-resident mask for cache_sim/gem5, via
`findNextRef` for Sniper); the adapter is covered by that backend's live trace.

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

`verify_ecg.py` checks correctness in **three complementary layers**:

1. **Synthetic exact-victim tests** (`bench/src_sim/test_ecg_victim.cc`) — construct
   controlled 8-way sets (property/record lines with chosen epochs/recency) and assert
   the **exact** victim each variant must pick, computed independently in the test. Pins
   the exact victim including cases the live workload never produces. Mutation-tested:
   flipping the implementation's farthest→nearest epoch pick makes it fail.
2. **Live-trace integration** (default geometry) — run each policy with `ECG_EVICT_TRACE`,
   parse every real eviction (per-way rrpv/epoch/dist/property/recency + victim), and
   assert it obeys the **tightened exact-victim rule**. The same `[EVICT L3 ...]` format is
   emitted by all three simulators.
3. **Epoch-coverage runs** — a forcing geometry (big L2 absorbs the edge stream → property
   reaches the L3; `ECG_STORED_REFRESH` keeps the L3 epoch live) makes the **epoch-property
   branch fire on the real simulator end-to-end**. Asserts the exact rule AND that the epoch
   value genuinely *selected* the victim ≥1× (a strict coverage gate, so the check cannot
   pass vacuously). Mutation-tested live as well.

**Scope (honest):** the *default* workload only evicts records, so the epoch branch is
exercised by layers 1 and 3. Layer 3 confirms `rrip_first`/`epoch_first`/`epoch_only` pick
the farthest-epoch property live; `shortcircuit` ranks property by *raw* distance (evicts
unstamped property first), so its stamped-epoch ranking is rarely operative live and is
covered by the synthetic test. cache_sim runs all three layers (strict epoch-value gate).
gem5 runs layers 2–3, but its layer-3 epoch-value gate is **informational**: gem5 has no
`ECG_STORED_REFRESH`, so property reaches its L3 unstamped and the epoch value cannot
discriminate live — gem5's epoch ranking is verified by the exact-rule check plus the
line-by-line mirror of the strictly-verified cache_sim block. Sniper runs layer 2.

```bash
make sim-pr
python3 scripts/experiments/ecg/verify_ecg.py            # synthetic + cache_sim live + epoch-coverage
python3 scripts/experiments/ecg/verify_ecg.py --gem5     # + gem5 live + gem5 epoch-coverage
python3 scripts/experiments/ecg/verify_ecg.py --sniper   # + Sniper live (guarded, prlimit)
# expected: each policy "N/N evictions obey spec [OK]" → "ALL POLICIES VERIFIED ✓"
```

Verified to date: **cache_sim** (7 policies × 40/40 evictions), **gem5** (5
ECG_GRASP_POPT variants × 40/40), and **Sniper** (4 ECG-specific variants × 40/40;
the guarded `sg_kernel` run is memory-capped via `--sniper-memory-limit-gb`). All
three emit the same `[EVICT L3 ...]` trace, so one checker verifies every backend.
Larger real-graph Sniper runs may still hit the documented SDE memory behavior;
the tiny email-Eu-core verification run completes cleanly under the cap.

## Prefetch path (DROPLET vs ECG prefetch)

ECG also ships a hint-driven prefetcher. The eviction policy above is the
*bandwidth* story; the prefetcher is the *latency* story, and the two are
deliberately separated:

- **DROPLET** (Basak et al., HPCA'19) — the literature baseline: prefetch the
  next-K edges in the stream (no target selection).
- **ECG_PFX** — prefetch the POPT-best target among the next-K in-neighbours,
  chosen by `ecg_mode6::selectPrefetchTarget` (`bench/include/ecg_mode6_builder.h`).
  Like the eviction decision, this target function is a **single shared header
  compiled into the cache_sim, gem5 and Sniper kernels**, so the ECG prefetch
  *decision* is identical across all three backends.

**The honest finding:** a prefetcher can only *relocate* traffic
(demand → prefetch); it can **never reduce total DRAM traffic** — only the
eviction policy can. So the prefetcher comparison is about *latency per unit of
bandwidth*, not bandwidth itself.

Prefetcher comparison at fixed LRU eviction (cache_sim, PageRank, `-o 0`,
lookahead 8; `demand2mem` = demand misses reaching memory = latency proxy;
`bandwidth` = total DRAM traffic; both lower = better):

| graph/L3 | prefetcher | demand2mem | bandwidth | fills | useful% |
|---|---|---|---|---|---|
| web-Google/512kB | none | 7.69M | 7.69M | 0 | – |
| web-Google/512kB | DROPLET | **1.45M** | 7.72M | 6.27M | 100 |
| web-Google/512kB | ECG_PFX | 5.41M | 7.69M | **2.28M** | 100 |
| cit-Patents/1MB | none | 33.73M | 33.73M | 0 | – |
| cit-Patents/1MB | DROPLET | **6.51M** | 33.73M | 27.22M | 100 |
| cit-Patents/1MB | ECG_PFX | 24.27M | 33.73M | **9.46M** | 100 |
| soc-pokec/1MB | none | 27.13M | 27.13M | 0 | – |
| soc-pokec/1MB | DROPLET | **3.50M** | 27.95M | 24.44M | 100 |
| soc-pokec/1MB | ECG_PFX | 20.10M | 27.26M | **7.16M** | 100 |
| com-orkut/2MB | none | 97.69M | 97.69M | 0 | – |
| com-orkut/2MB | DROPLET | **15.38M** | 104.39M | 89.01M | 100 |
| com-orkut/2MB | ECG_PFX | 73.14M | 98.65M | **25.51M** | 100 |
| roadNet-CA/512kB | none | 0.87M | 0.87M | 0 | – |
| roadNet-CA/512kB | DROPLET | 0.73M | 0.87M | 0.14M | 100 |
| roadNet-CA/512kB | ECG_PFX | 0.75M | 0.87M | 0.12M | 100 |

Reading it: **DROPLET** trades the most bandwidth for the biggest latency cut —
it issues 3–3.5× more prefetches and on com-orkut **over-fetches** (bandwidth
104.4M vs 97.7M baseline). **ECG_PFX** is bandwidth-efficient: it reaches the
same 100% useful-rate while issuing 2.7–3.5× fewer prefetches (it picks
POPT-best targets instead of every next-K edge), keeping total traffic at the
baseline. The full bandwidth win comes from stacking ECG_PFX on the
ECG_GRASP_POPT *eviction* (the table above is fixed-LRU to isolate the
prefetcher lever — see `docs/findings/prefetcher_saturation_under_eviction.md`
for the combined stack). Artifact: `wiki/data/ecg_prefetch_matrix.md`.

### Reproduce + verify the prefetch path

```bash
# performance matrix (rows = graph x prefetcher at fixed eviction)
python3 scripts/experiments/ecg/ecg_prefetch_matrix.py --suite cache-sim \
    --cells "web-Google:512kB cit-Patents:1MB soc-pokec:1MB com-orkut:2MB roadNet-CA:512kB" \
    --order 0 --eviction LRU --lookahead 8 --out /tmp/pfx_matrix.md

# correctness verification (synthetic shared decision + live per-prefetcher spec)
python3 scripts/experiments/ecg/verify_pfx.py
```

`verify_pfx.py` mirrors `verify_ecg.py` in two layers:

1. **Synthetic exact-target** (`bench/src_sim/test_ecg_prefetch.cc`) — asserts the
   exact target of `ecg_mode6::selectPrefetchTarget` (min re-reference among the
   next-K, window clipping, ties, invalid/out-of-range entries, disabled). Because
   that function is the single shared decision, this verifies the ECG prefetch
   target for **all three simulators**; mutation-proven (flipping min→max fails).
2. **Live behaviour** (cache_sim) — runs PageRank with each prefetcher and asserts
   its defining, falsifiable property: `none` issues nothing and traffic == demand;
   `DROPLET` fires, cuts demand-to-mem, and does **not** reduce total traffic
   (conserves bandwidth); `ECG_PFX` fires, cuts demand-to-mem, and issues strictly
   **fewer** prefetches than DROPLET at ≥50% useful-rate (selective targets).

**Cross-simulator status (honest):** the ECG prefetch *decision* (target
selection) is verified for all three backends via the shared function. *Live
firing* is verified end-to-end in cache_sim, which has no page-cross filter and
issues over the full property region, so both DROPLET and ECG_PFX property
prefetches fire there. In **gem5**, the generic `Queued::notify` page-cross
filter drops *any* cross-page `property[v]` prefetch unless the prefetcher has an
MMU plumbed (none does) — this filters **both** ECG_PFX *and* DROPLET's
indirect-property engine; only DROPLET's same-page *stride* engine still issues,
so the gem5 prefetch comparison is not yet apples-to-apples (one MMU-plumbing fix
un-blocks both; `docs/findings/gem5_implementation_audit_v1.md`, deferred). In
**Sniper**, ECG_PFX fires under the guarded `sg_kernel` path. cache_sim is
therefore the authoritative prefetch performance model.

## Related

- [[Cache-Simulation]] — simulator architecture, all `ECG_MODE` values, env vars
- [[ECG-Final-Runs]] — gem5/Sniper final-run profiles, charged P-OPT, topology caveats
- [[Baseline-Literature-Faithfulness]] — GRASP/P-OPT faithfulness audit
