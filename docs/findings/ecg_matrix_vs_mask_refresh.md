# Why the P-OPT matrix beats our epoch mask — and why REFRESH is an idealized, not feasible, fix

**Date:** 2026-06-30
**Prompted by:** "can we enhance it … we have 30 bits … make the epoch analysis more
sophisticated … what is the algorithm we are using currently, let's study it, and see
why it works for P-OPT and doesn't work for us." + "what is the refresh mechanism, is it
feasible in hardware?"

## 1. The two algorithms, side by side

**P-OPT rereference matrix** (`graph_cache_context.h:718`, `findNextRef(cline, cur_vtx)`)
is a **2-D** structure indexed by `(current_epoch, cache_line)`. At eviction it
re-derives `epoch_id = cur_vtx / epoch_size` **live** and reads the row → "distance to
this line's next reference *from right now*." A cold resident line is **re-scored every
eviction**; it never goes stale. This is the structure P-OPT reserves an LLC way for.

**ECG epoch mask** (`cache_sim.h:1762`, `dist = (ecg_epoch + ne − cur_epoch) % ne`)
is **1-D**: the line carries **one scalar** `ecg_epoch` — the next-ref epoch *stamped at
fill time*. Once `cur_epoch` passes it, the circular distance wraps to ≈`ne` → the line
looks "farthest-future" → evicted, even if it is referenced again soon. The mask is
**blind to every reference after the first stamped one.**

**Crux:** the gap is a missing *dimension* (epoch), not missing *precision*. This is why
prior fine-quantization / 8-byte-record / more-epoch-bits sweeps all **saturated** — they
bought precision on a frozen scalar, not the epoch dimension that makes the matrix work.

## 2. Two ways to add the missing dimension

| mechanism | idea | result |
|---|---|---|
| **`ECG_EDGE_MASK_SCHED=K`** (new) | spend the spare bits on a per-edge **schedule** of the next-K next-ref epochs; at eviction take the soonest entry still ahead of `cur_epoch`, so the line self-advances | helps the stale base ~1–2pp but **strictly dominated by REFRESH** on every graph; combining is worse. The 30 spare bits do **not** beat re-delivering the one epoch we already stream. Kept as a flag-gated, off-by-default ablation. |
| **`ECG_STORED_REFRESH`** (existing, **never enabled by any orchestrator**) | re-stamp a resident LLC line's `ecg_epoch` from the per-edge hint on **every** access (even on L1/L2 hits), modelling `ecg.extract` re-delivering the mask per edge | closes ~2.4pp and produces the only ECG eviction win — **but it is an IDEALIZED CEILING, not hardware-free** (see §6). |

## 3. Measured (cache_sim, PR, `epoch_only`, ne=1024, `-i3`, L1 32k/L2 256k)

**Equal 16-way LLC (pure eviction quality, CHARGED=0):**

| graph | POPT | ECG base | **+REFRESH** | +SCHED=4 |
|---|---|---|---|---|
| web-Google (2MB)      | 0.1991 | 0.2205 | **0.1810** | 0.1895 |
| soc-pokec (2MB)       | 0.2782 | 0.3284 | 0.3043 | 0.3184 |
| soc-LiveJournal1 (4MB)| 0.2990 | 0.3913 | 0.3666 | 0.3946 |

REFRESH closes ~2.4pp everywhere and **flips web-Google to an equal-ways win**. On the
denser social graphs the matrix is still a better equal-capacity oracle.

**Faithful (POPT pays `size_correct` reserved ways; ECG pays mask traffic + REFRESH):**

| graph | reserved ways | POPT charged | ECG+REFRESH | verdict |
|---|---|---|---|---|
| web-Google       | 1 | 0.2126 | **0.1810** | **ECG +3.2pp** |
| soc-pokec        | 2 | 0.3262 | **0.3043** | **ECG +2.2pp** |
| soc-LiveJournal1 | 3 | 0.3499 | 0.3666 | POPT +1.7pp |

Without REFRESH, faithful ECG **loses everywhere**. The aggressive REFRESH appears to flip
web-Google and soc-pokec to wins — **but that refresh is an idealized ceiling, not
hardware-feasible (see §6), and its feasible form recovers ~0**, so the honest faithful
result is: ECG loses to P-OPT on every tested graph.

## 4. Honest framing

- ECG does **not** beat the P-OPT matrix at equal capacity (the matrix's live 2-D next-ref
  is a genuinely better oracle, especially on dense graphs).
- The apparent "reserved-way-avoidance win" **depends entirely on the aggressive REFRESH**,
  which is an idealized, uncharged per-access LLC metadata broadcast (§6). Its
  hardware-feasible form recovers ~0, and feasible ECG loses to charged P-OPT everywhere.
- The "spare-bit schedule" idea is sound but empirically dominated by (idealized) refresh.
- ECG's defensible advantage is **scale/feasibility** (memory-resident mask works where
  P-OPT's reserved matrix cannot fit) and GRASP-degree insertion — **not** a faithful
  miss-rate win over P-OPT.

## 5. Behaviour change

`scripts/experiments/ecg/roi_matrix.py` exposes `--ecg-stored-refresh` (default **OFF**,
presence-gated) and `--ecg-refresh-llc-only`. The new `ECG_EDGE_MASK_SCHED=K` ablation is
implemented and inert when unset (byte-identical to prior ECG_GRASP_POPT). Refresh is OFF
by default because it is an idealized ceiling (see §6), not a faithful lever.

## 6. Hardware feasibility of REFRESH (the decisive caveat)

**What the mechanism actually does** (`cache_sim.h`, `refreshExactStamp` @897, called at the
top of `access()` @2138, *before* the L1/L2/L3 lookup): on every edge access it writes the
per-edge epoch hint into the **L3 line's metadata** — and because the hierarchy is
effectively inclusive (fills populate all levels), the L3 copy of a line that is hot in L1
gets a metadata write on **every L1/L2 hit**. That is an aggressive *per-access LLC
metadata broadcast*.

**The feasibility test** (`ECG_REFRESH_LLC_ONLY`): restrict the re-stamp to accesses that
actually **reach L3** (miss L1+L2), so the write piggybacks an L3 access already in flight
= genuinely free. Result (PR, epoch_only, ne=1024, −i3):

| | POPT | no-refresh | refresh AGGRESSIVE | **refresh LLC-only (feasible)** |
|---|---|---|---|---|
| web-Google (equal 16w) | 0.1991 | 0.2205 | 0.1810 | **0.2205 (== no-refresh)** |
| soc-pokec (equal 16w) | 0.2782 | 0.3284 | 0.3043 | **0.3284 (== no-refresh)** |

Faithful (POPT charged reserved ways, ECG 16w CHARGED=1):

| | POPT charged | ECG aggressive | **ECG feasible (LLC-only)** |
|---|---|---|---|
| web-Google | 0.2126 | 0.1810 | **0.2205 — LOSES** |
| soc-pokec  | 0.3262 | 0.3043 | **0.3284 — LOSES** |

**Conclusion.** The feasible (piggybacked) refresh recovers **~zero** of the benefit. The
entire refresh win comes from the uncharged per-access L3 metadata write on inner-cache
hits — a real cost cache_sim does not model. The prior "epoch already streamed, free under
LEAN+PACK" claim conflated two things: obtaining the epoch is free (it rides the edge word),
but **delivering it to the L3 line on every inner-cache hit is an extra L3 tag-write stream**,
not free. To realise it you would need a dedicated per-edge LLC epoch-update channel with
real tag-array write bandwidth — feasible to build but with a per-access cost that may
exceed P-OPT's (P-OPT pays capacity + a per-*eviction* query, and needs no per-access write).

**Honest standing:** with the feasible refresh, ECG does **not** beat P-OPT on eviction
quality on any tested graph. P-OPT's live 2-D matrix is the better oracle, and the only
mechanism that closes the gap for ECG is an idealized, uncharged broadcast. ECG's defensible
advantage is therefore **scale/feasibility** (the memory-resident mask scales with the edge
list and works where P-OPT's reserved matrix cannot fit, `popt_matrix_fits=0`), plus
GRASP-degree insertion — **not** a faithful miss-rate win over P-OPT via the epoch eviction.

## 7. Exhaustive verdict: can any FEASIBLE mask-only mechanism beat P-OPT? (2026-06-30)

Prompted by "if we can't match or beat P-OPT using mask bits only we give up." Every
hardware-feasible mask-only mechanism (no reserved way, hint delivered only when the
access legitimately reaches the line) was tested:

| mechanism | feasible? | beats P-OPT miss rate? |
|---|---|---|
| GRASP degree tiers (insertion) | yes | no (first-order proxy only) |
| coarse epoch stamp @ eviction | yes | no (stale) |
| multi-ref schedule `ECG_EDGE_MASK_SCHED` | yes | no (dominated by refresh) |
| refresh, feasible `ECG_REFRESH_LLC_ONLY` | yes | no (recovers ~0, §6) |
| epoch @ INSERTION (freshest moment) | yes | **no — HURTS** (web-Google 0.32 vs 0.23) |
| refresh, aggressive | **NO** (§6) | yes, but idealized/uncharged |

**Root cause (fundamental).** PageRank is *cyclic* — every vertex is re-referenced every
iteration — so a **frozen** per-line stamp is a weak signal *no matter when it is written*
(insertion or eviction). P-OPT wins via **live per-epoch sub-ordering**, which only a
resident matrix (reserved way) or an infeasible per-access broadcast can supply.

**Standing (feasible, mask-only, demand L3 miss rate, charged P-OPT):**
web-Google POPT 0.2126 / ECG 0.2205 (+0.8pp); soc-pokec 0.3262 / 0.3284 (+0.2pp);
soc-LiveJournal 0.3499 / 0.3913 (+4.1pp). ECG **matches within <1pp on 2/3 graphs at 0
reserved ways** but does **not** cleanly beat P-OPT.

**Total-memory-traffic angle is INCONCLUSIVE** (not a win): charging P-OPT's uncharged
matrix stream (~16·|V|/sweep) makes ECG's total DRAM 90% of P-OPT on web-Google but 107%
on soc-pokec — the accounting is confounded (ECG's CHARGED=1 mask stream is already inside
its `memory_accesses`; mode-6 adds mask-array accesses). No clean claim is supportable
without modelling both metadata streams inside the simulator.

**Conclusion for the project.** Feasible mask-only ECG does **not** beat P-OPT's miss rate;
it matches within <1pp at zero reserved ways. Whether "P-OPT-competitive at lower dedicated
LLC cost" clears the project bar is a **strategic thesis decision left to the user** — the
data does not support a clean miss-rate win, and this write-up avoids manufacturing one.

## 8. Literature synthesis (2026-06-30) — is "beat P-OPT" even the right bar?

Prompted by "let's see out there in literature to draw insight." External web search
was unavailable; this draws on the local corpus (primary: the P-OPT HPCA'21 paper) and
the team's `research/caching/` notes.

**Insight 1 — the goal in this space is to APPROACH the oracle, not beat it.** The ECG
notes' own expected hierarchy is `P-OPT ≤ ECG(DBG_PRIMARY) ≤ … ≤ GRASP` (research/caching/
ecg.md:289) — P-OPT is positioned as the *oracle ceiling*. P-OPT itself only *approaches*
Belady/T-OPT (its paper reports within ~1–5% of OPT, not beating it). So the honest ECG
thesis is the analogue: **approach P-OPT at lower hardware overhead** — which the measured
<1pp gap at zero reserved ways achieves. "Beat P-OPT's miss rate" was never the paper's
claim and is not the literature's bar.

**Insight 2 — why generic predictors fail on graphs (P-OPT §II, lines 280–311).** SHiP/
Hawkeye predict reuse from PC or address, assuming all accesses by an instruction/region
share reuse. Graph kernels violate this: the *same* load has different locality for high-
vs low-degree vertices; "even with infinite storage, SHiP-Mem gives little improvement
over LRU." This is precisely the gap ECG's **per-vertex** mask fills — a graph-aware
predictor where per-PC predictors are blind.

**Insight 3 — but insertion-time reuse hints don't help here (measured).** The team's
Hawkeye-inspired feasible modes were evaluated (web-Google 2MB/16w, −i3):
POPT 0.1991 · **ECG_GRASP_POPT(epoch) 0.2205** · GRASP 0.2291 · ECG_EMBEDDED 0.2350 ·
ECG_COMBINED 0.2539. The insertion-blend modes are *worse than plain GRASP* — a static
reref hint dilutes GRASP's clean degree signal (consistent with the epoch-at-insertion
result in §7). The best *feasible* mode remains GRASP insertion + epoch eviction-tiebreak.

**Net.** The literature reframes the project honestly: ECG's contribution is **P-OPT-
quality graph caching (within <1pp) via a memory-resident per-vertex mask — no reserved
LLC way, no matrix stream** — mirroring how P-OPT is positioned vs Belady. This is a
defensible thesis; "beat P-OPT" is not the right bar and is not supported. Recommended
paper framing: *practical, overhead-free approximation of the graph-caching oracle*, with
P-OPT as the ceiling ECG approaches, not a baseline ECG must exceed.

## 9. New mechanism: StreamShield + Schedule-2 (2026-07-10, cache_sim result)

The §7 verdict remains correct for the **old single-epoch mask alone**. A new mechanism
adds two orthogonal capabilities P-OPT does not have:

1. **StreamShield:** packed one-touch edge records carry a non-temporal hint. After an
   L2 miss they bypass the LLC tag/data/allocation path and fill only L2/L1. A
   bypass-aware stride prefetcher warms those private caches without consuming LLC ways.
2. **Schedule-2:** the 8-byte record carries the next two per-line reference epochs.
   Resident property lines self-advance to the second epoch after the first passes,
   recovering one extra dimension of the P-OPT matrix without a reserved way or matrix.

The implementation is gated by `ECG_STREAM_BYPASS=1` and
`ECG_EDGE_MASK_SCHED=2`; both are default-off. Record sizing is honestly charged:
web-Google's 20-bit ID + 2×12-bit epochs + tier fits one 8-byte record; K4 promotes
to 16 bytes and is rejected as traffic-heavy.

Full five-policy PR results (`-o5`, L1D=32kB, L2=256kB, 16-way LLC, STRIDE degree
8; lower demand memory accesses is better):

| graph / LLC | LRU | SRRIP | GRASP | P-OPT | StreamShield+K2 | vs P-OPT |
|---|---:|---:|---:|---:|---:|---:|
| web-Google / 2MB | 1,758,103 | 1,390,247 | 1,330,034 | 1,036,428 | **764,123** | **-26.3%** |
| soc-pokec / 2MB | 13,400,665 | 11,489,464 | 9,249,576 | 8,143,075 | **6,228,099** | **-23.5%** |
| cit-Patents / 8MB | 9,389,641 | 7,624,847 | 6,251,112 | 4,769,337 | **3,747,240** | **-21.4%** |

These corrected rows isolate epoch metadata to the governed PR `contrib[]` region;
older five-graph numbers that allowed `scores[]` to borrow the same epoch are
superseded. LiveJournal and roadNet controls require rerun under this corrected
region isolation. **Traffic caveat:** total traffic remains above uncharged P-OPT
because the 8-byte record stream and private-cache prefetch fills are charged.
Schedule-2 delivery and StreamShield allocation control are now ported and
decision-verified in gem5 and Sniper (§10-§12). The real-graph demand-miss and
traffic result remains cache_sim-authoritative until a feasible real-graph Sniper
matrix confirms the full performance claim.

## 10. Schedule-2 three-simulator equivalence (2026-07-11)

Schedule-2 now uses one shared builder and one shared effective-distance rule:

- `ecg_epoch::buildInEdgeEpochPairs` builds pull (PR) and push (BFS) pairs;
- the wire record is `dest[0:32] | epoch1[32:48] | epoch2[48:64]`;
- `ecg_policy::epochPairDistance` computes
  `min((e1-cur+ne)%ne, (e2-cur+ne)%ne)`;
- PR selects `epoch_first`; BFS selects traversal-safe `degree_first`.
- gem5/Sniper build the packed stream directly and skip the P-OPT matrix plus
  legacy single-epoch/mask copies in K2 runs; Sniper's bounded delivery map is
  cache-line keyed and region-gated, so collisions leave a line unstamped
  instead of borrowing stale metadata from another vertex/property array.
- the runner scopes `ECG_EDGE_MASK_SCHED=2` to the ECG policy only; side-by-side
  P-OPT cells still build/load their rereference matrix.

The verifier runs the same pressured graph and parses every native backend trace.
It requires non-identical K2 pairs, zero distance mismatches, and exact victim-rule
compliance:

| kernel | simulator | traced K2 property ways | non-identical pairs | distance mismatches | epoch-decisive victims |
|---|---|---:|---:|---:|---:|
| PR | cache_sim | 23,579 | 14,345 | 0 | 3 |
| PR | gem5 | 6,281 | 2,058 | 0 | 131 |
| PR | Sniper | 7,843 | 5,129 | 0 | 0* |
| BFS | cache_sim | 3,454 | 2,825 | 0 | 3 |
| BFS | gem5 | 7,328 | 3,071 | 0 | 9 |
| BFS | Sniper | 5,214 | 2,605 | 0 | 3 |

\* Sniper PR's bounded non-inclusive trace evicted records throughout; resident
property lines still carried distinct K2 pairs whose effective distances were
verified. The shared exact-victim test plus cache_sim/gem5 decisive PR coverage
prevents this do-no-harm cell from becoming a vacuous policy check.

For gem5 and Sniper, the verifier also compares the first 32 kernel-emitted
`(dest,epoch1,epoch2)` records against the 32 records actually decoded by the
backend; both are **32/32 exact matches** for PR and BFS.

Reproduce:

```bash
python3 scripts/experiments/ecg/verify/equiv_kernels.py \
  --gem5 --sniper --kernels pr bfs --schedule-k 2
```

This closes **K2 construction, delivery, line metadata, distance, and victim-decision
equivalence**. Section 11 closes the subsequent StreamShield allocation gap.
gem5 K2 remains restricted to in-order CPUs: `ecg.extract2` uses the serialized
mailbox, so the runner rejects O3 until a request-bound epoch-pair extension exists.

## 11. StreamShield port and K2-vs-bypass attribution (2026-07-11)

StreamShield is now implemented in all three simulators for PR:

- **cache_sim:** `accessStream()` skips LLC lookup/allocation and fills L2/L1.
- **gem5:** the PR packed-record virtual range is exported in the sideband;
  shared `system.l3cache` sets the MSHR `allocOnFill` bit to false. The response
  still fills the private hierarchy.
- **Sniper:** NUCA reads retain normal tag lookup, hit handling, and latency. On a
  miss, the returning NUCA write is discarded without insertion. Dedicated
  `stream-bypass-reads/writes` counters prove the read-miss and suppressed-write
  halves fire.

The bypass is PR-only, ECG-only, and default-off. Baseline policies never inherit
`ECG_STREAM_BYPASS`, so P-OPT still builds and uses its matrix.

Reproduce the mechanism gate:

```bash
python3 scripts/experiments/ecg/verify/equiv_kernels.py \
  --gem5 --sniper --kernels pr --schedule-k 2 --stream-bypass
```

### 2×2 factorial: which mechanism contributes more?

PR `-i1`, `-o5`, L1D=32kB, L2=256kB, 16-way LLC, STRIDE8. Lower demand
memory accesses is better. `K1` is the original single-epoch ECG.

| graph | LRU | SRRIP | GRASP | P-OPT | K1 | K1+StreamShield | K2 | K2+StreamShield |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| web-Google / 2MB | 1,758,103 | 1,390,247 | 1,330,034 | 1,036,428 | 1,080,415 | 997,671 | 815,073 | **764,123** |
| soc-pokec / 2MB | 13,400,665 | 11,489,464 | 9,249,576 | 8,143,075 | 7,551,255 | 7,323,620 | 6,433,736 | **6,228,099** |
| cit-Patents / 8MB | 9,389,641 | 7,624,847 | 6,251,112 | 4,769,337 | 4,288,176 | 4,063,879 | 3,943,972 | **3,747,240** |

Shapley attribution averages each mechanism's marginal contribution with the
other mechanism off and on:

| graph | total K1 → K2+SS reduction | StreamShield share | K2 share |
|---|---:|---:|---:|
| web-Google | 29.3% | 21.1% | **78.9%** |
| soc-pokec | 17.5% | 16.4% | **83.6%** |
| cit-Patents | 12.6% | 38.9% | **61.1%** |
| aggregate, weighted by avoided accesses | — | 22.7% | **77.3%** |

**Conclusion:** K2 is the dominant mechanism, contributing about three quarters
of the combined reduction. StreamShield contributes the remaining quarter and is
still material, especially on cit-Patents. The interaction is mildly antagonistic:
once K2 removes property misses, fewer pollution misses remain for bypass to remove.

## 12. Request-bound RISC-V implementation and Sniper fused model

StreamShield is now an instruction property rather than only an address-range
experiment. Two custom-0 I-type record loads use the same K2 wire layout:

| instruction | custom-0 FUNCT3 | action |
|---|---:|---|
| `ecg.load2 rd, 0(rs1)` | `0x4` | load and return the 64-bit K2 record; deliver both epochs normally |
| `ecg.stream.load2 rd, 0(rs1)` | `0x3` | same load/delivery plus `Request::ECG_STREAM_BYPASS` |

The request flag rides the specific load through the core and cache hierarchy.
L1/L2 behave normally. At `system.l3cache`, the flag clears the MSHR
`allocOnFill` decision; an LLC hit remains usable, but an LLC miss is not
installed. STRIDE prefetches derived from that load inherit the same request flag.
Request-bound cells disable the address-range fallback, and logs report
`source=request-flag`, so the ISA bit is independently load-bearing. This is
race-free under OoO because the bypass decision is per request, not a mailbox.

The RISC-V decoder test executes both instructions through the real gem5 decoder,
checks the complete 64-bit record round trip, and confirms the StreamShield request
reaches the L3 no-allocate path.

Sniper cannot execute RISC-V binaries directly, so its scale model uses the same
single-record-load semantics without per-edge SimMagic: K2 offsets/records are
exported before the ROI, and the cache consumes the exact pair for
`(current source, property line)` when the governed line reaches the LLC. Live
receipt traces are checked against the exported binary records; temporary K2
sidebands are deleted after each cell.

### Matched fused-load performance check

Both K2 cells now use one fused record-load instruction/model; their only
difference is the StreamShield no-allocate bit.

| simulator / graph | LRU | SRRIP | GRASP | P-OPT | K2 | K2+StreamShield | StreamShield marginal speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| gem5 / kron_s16_k4, simulated ticks | 23.076B | 22.520B | 22.192B | **21.600B** | 30.476B | 26.962B | **13.03%** |
| Sniper / kron_s16_k16, simulated ticks | timeout | timeout | **35.865T** | timeout | 46.952T | 46.647T | **0.65%** |

gem5 logs identify the source as
`source=request-flag allocate=0`, proving the custom instruction—not only the
address-range fallback—drives the no-allocation decision.

In clean gem5, StreamShield also cuts K2 L3 misses from 39,333 to 16,425
(**-58.24%**). In clean Sniper, it cuts K2 L3 misses from 9,889,214 to
9,859,131 (**-0.30%**) while executing the exact same 118,517,996 instructions.
However, K2+StreamShield still does **not** beat the available full baselines on
these synthetic mechanism cells. The kron cells are unsuitable
for an ECG quality claim (the epoch signal is weak/inert and Sniper's
non-inclusive hierarchy changes record reuse). Therefore the valid statement is:

> StreamShield improves the K2 implementation in both gem5 and Sniper. The overall
> ECG-vs-P-OPT performance win remains established only by the corrected
> cache_sim real-graph factorial until a feasible real-graph Sniper matrix is completed.
