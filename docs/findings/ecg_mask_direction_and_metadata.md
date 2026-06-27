# ECG mask: which metadata is load-bearing, and graph-direction correctness

**Date:** 2026-06-21
**Audit prompted by:** "are we using all the masks?" + "when we measure epoch/POPT we
must be aware of graph vs inverse graph (BFS pull/push); POPT should load a different
matrix per direction; review how the paper handled it."

Two independent findings: (1) **which packed mask fields are actually read** by each
policy, and (2) **whether the epoch/POPT next-reference matrix uses the correct graph
direction** for each kernel. Both are correctness/honesty boundaries for the artifact.

---

## 1. Which metadata is load-bearing (verified)

The ECG per-edge mask packs `dest | DBG-tier(2b) | POPT-quant(7b) | epoch(16b) |
prefetch_target`. What the cache actually **reads at runtime**:

| Policy (matrix column) | recency | rrpv | DBG tier | **epoch** | POPT-7b | prefetch tgt |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| LRU | ✅ | – | – | – | – | – |
| GRASP | – | ✅ | ✅ (insertion rrpv) | – | – | – |
| POPT (baseline) | – | – | – | *its own rereference matrix* | – | – |
| ECG:grasp_only | – | ✅ | (insertion) | – | – | – |
| ECG:epoch_only | ✅ | – | – | ✅ | – | – |
| ECG:epoch_first | ✅ | – | – | ✅ | – | – |
| ECG:rrip_first (default) | ✅ | ✅ | – | ✅ | – | – |
| ECG:shortcircuit | (set order) | – | ✅ (tiebreak) | ✅ (raw) | – | – |

Verified against the shared `ecg_policy::selectVictim`: its `WayState` is
`{prop, rrpv, recency, dbg, dist(=epoch), stamped}` — **there is no POPT field**.

**Takeaways:**
- **The epoch is the load-bearing ECG eviction field.** Every ECG variant that beats
  the baselines wins via the per-edge next-reference **epoch** (`dist`).
- **The 7-bit POPT-quant field is vestigial for the ECG_GRASP_POPT headline eviction**
  — no `selectVictim` variant reads it. (It is *not* globally dead: cache_sim's
  `ECG_EXACT_MASK`/`ECG_COMBINED`/`ECG_EMBEDDED` modes and gem5's ISA decode/`ECG_RP`
  still carry/consume `ecg_popt_hint`, so we **document it as vestigial-for-headline,
  not remove it** — removal blast radius is not worth it.)
- **POPT *data* is build-time only.** The POPT rereference matrix feeds (a) the epoch
  values and (b) the ECG_PFX target choice (`selectPrefetchTarget` reads
  `avg_reref_by_line`). The *cache* only ever reads the epoch (eviction) and the target
  (prefetch).
- The **DBG/degree tier** is used by GRASP's insertion RRPV and as a shortcircuit
  tiebreak (≈0 contribution); the **prefetch-target field** is used only in the
  prefetch experiments.

---

## 2. Graph-direction correctness of the rereference matrix

**Principle (and this is exactly the P-OPT paper's thesis).** The next reference to a
property line accessed while traversing edge-direction *D* is determined by the
**opposite** adjacency (the graph **transpose**). If a kernel reads `prop[v]` while
visiting `u`'s *D*-neighbours, then `prop[v]` is next read at the next `u'` whose
*D*-list also contains `v` — i.e. the next *transpose*-neighbour of `v`.

P-OPT (Balaji & Lucia, HPCA'21) is literally *"Transpose-based Cache Replacement"*:
its Rereference Matrix summarises the graph **transpose**, and the system stores
**both CSR and CSC** so either direction is available
(`research/POPT_HPCA21_CameraReady.txt`). So "use a per-direction matrix" is not new —
it is P-OPT's core idea; our per-edge mask is a re-derivation of the same flexibility.

**Our mechanism.** `makeOffsetMatrix(g, …, traverseCSR=true)` builds the matrix from
`out_neigh` when `traverseCSR=true` (the default) and from `in_neigh` when `false`
(`popt.h:422-438`); undirected graphs force `true` (in==out). **Every kernel currently
calls it with the default `true` (out_neigh).**

**Audit (cache_sim kernels):**

| Kernel / phase | traverses | reads property | needs next-ref from | matrix used | verdict |
|---|---|---|---|---|---|
| **PageRank** (pull) | `in_neigh(u)` | `contrib[v]` | `out_neigh(v)` (transpose) | `out_neigh` (default) | ✅ **correct** |
| **CC** | `out_neigh(u)` | `comp[v]` | `in_neigh(v)` | `out_neigh`, **assumes out==in** | ✅ undirected-only (documented in cc.cc) |
| **BFS top-down** (push) | `out_neigh(u)` | `parent[v]` | `in_neigh(v)` | `out_neigh` default **or** BFS visit-order skeleton clock | ⚠️ default is wrong-direction; the `ECG_EXACT_BFS` skeleton models BFS order instead |
| **BFS bottom-up** (pull) | `in_neigh(u)` | frontier check | — | — | (frontier bitmap; little property-mask reuse) |
| **SSSP** | `out_neigh(u)` | `dist[v]` | `in_neigh(v)` | `out_neigh` (default) | ⚠️ direction-uncertified on **directed** graphs |
| **BC** | `out_neigh(u)` | `depths[v]`,`path_counts[v]` | `in_neigh(v)` | `out_neigh` (default) | ⚠️ direction-uncertified on **directed** graphs |

**What this means (honest boundaries):**
- **PageRank — the headline kernel — is direction-correct.** Its in-pull + `out_neigh`
  matrix is exactly the transpose P-OPT prescribes, on any graph. The headline eviction
  and prefetch matrices are PR, so **there is no headline-results direction bug.**
- **CC** is correct only for **undirected/symmetric** graphs (it explicitly assumes
  `out==in`); this matches the existing "CC-ECG valid only for undirected graphs" note.
- **SSSP / BC / BFS-top-down traverse out-edges**, so the faithful next-ref is the
  transpose (`in_neigh`), but they use the default `out_neigh` matrix. On **directed**
  graphs this is the wrong direction. Both the P-OPT *baseline* and ECG read the *same*
  matrix, so their head-to-head comparison stays internally consistent (apples-to-apples)
  — but neither is *transpose-faithful* for these kernels on directed graphs. These
  kernels are **not** headlined as ECG wins (ECG's niche is PR on power-law graphs), so
  this is a **correctness boundary to certify before promoting them**, not a results bug.

**One-argument fix path (if/when these kernels are promoted):** pass
`makeOffsetMatrix(g, …, /*traverseCSR=*/false)` for the out-traversal kernels so the
rereference matrix is built from `in_neigh` (the transpose). We do **not** apply it now
(it changes those kernels' numbers and they are out of the headline scope) — and we do
**not** build both fwd+inverse matrices globally (doubling the ~16·V-byte matrix + the
8·E-byte mask stream for no headline benefit). Build the single correct direction per
kernel only when that kernel is certified.

---

## 3. Decisions (rubber-duck-gated, `rd-direction-plan`)

- **Keep the 7-bit POPT field** (document as vestigial-for-headline; it is still read by
  non-headline cache_sim modes + gem5 decode — removal blast radius not worth it).
- **Do not build dual fwd+inverse masks/matrices** globally (over-engineering;
  per-kernel single correct direction suffices).
- **PR is the direction-correct headline**; CC is undirected-only; SSSP/BC/BFS-TD are
  *direction-uncertified on directed graphs* and must use `traverseCSR=false` before
  being promoted as transpose-faithful ECG results.
- This is an **audit + documentation** outcome — no working path was perturbed.

---

## 4. IMPLEMENTED: per-kernel transpose direction (main+inverse), 2026-06-21

The audit boundary above is now **handled in code** (user request: "enable main and
inverse masking so we capture both ways; P-OPT does this too"). Added
`ecgRerefTraverseCSR(natural_csr, g, kernel)` in `popt.h`: it picks the kernel's
transpose-correct rereference direction, honours `ECG_REREF_TRANSPOSE=AUTO|OUT|IN`
(direction-transfer override), forces CSR on undirected graphs (in==out), logs the
choice, and **aborts loudly** if IN/CSC is requested but the inverse is unmaterialised
(no silent empty matrix). Wired per kernel:
- **PR** → `natural=true` (out_neigh) — identical to the old default ⇒ **byte-identical**
  (verified PR POPT web-Google/512kB/o0 = 0.6591, unchanged).
- **SSSP, BC** → `natural=false` (in_neigh, the transpose of their out-edge push).
- **CC** → `natural=false`, but undirected forces CSR (CC is undirected-only).
- **BFS** → conservative: default CSR for mixed DOBFS; `ECG_BFS_FORCE_TD` opts into the
  TD transpose; `ECG_EXACT_BFS` still uses its own visit-order skeleton clock.

**Key empirical finding — our eval corpus is (almost all) symmetric.** Probing
`g.directed()`: web-Google, cit-Patents, soc-pokec, com-orkut, kron_s16, soc-LiveJournal1
all load **undirected** (in==out); roadNet-CA loads "directed" but is **structurally
symmetric**. So the direction distinction is **moot** on the benchmark graphs, the
original `out_neigh` default was already correct for them, and the new per-kernel
direction is **inert** on them (no result perturbation — PR and all current numbers
unchanged). On a *genuinely asymmetric* directed graph the helper correctly selects
IN/CSC for push kernels (verified via the `[P-OPT reref] … IN/CSC [AUTO]` log on a
constructed asymmetric graph; the matrices differ by construction — out_neigh vs
in_neigh — though SSSP's POPT-eviction miss-rate was insensitive to the flip on the
tested graphs). The value is **forward-looking correctness + P-OPT parity + the
direction-transfer knob**, with zero risk to the symmetric-corpus headline results.

---

## 5. Per-edge-list dual-direction masking — invariant + why deferred (2026-06-21)

The masking is **per-edge-list**, and the graph materialises **both** adjacencies
(`out_neigh`/CSR and `in_neigh`/CSC, `invert=true`). The correct invariant is:

> A per-edge mask is tied to a specific edge list and uses the **transpose** adjacency
> for its next-reference: a kernel that traverses `in_neigh` (PR pull) needs IN-edge
> masks whose epoch is computed from `out_neigh`; a kernel that traverses `out_neigh`
> (SSSP/BC/BFS-TD push) needs OUT-edge masks whose epoch is computed from `in_neigh`.

`buildInEdgeMasks_PR` implements the IN case (iterates `in_neigh`, epoch from
`exact_nbr` built on `out_neigh`, line 1262) and is correct for PR pull. An OUT-edge
per-edge builder is the mirror (iterate `out_neigh`, epoch from `in_neigh`).

**Correction to a tempting claim.** "Run the same algorithm on the inverse graph and it
should be the same" is **only true for symmetric graphs**. PageRank on `G` vs `Gᵀ` is a
*different computation* (ranks of the reverse graph) unless `G` is symmetric. The valid
goal is therefore **mask correctness for whichever direction a kernel traverses**, not
"same result on the inverse graph."

**Status: capability documented, full build DEFERRED** (rubber-duck `rd-dual-mask-plan`),
because on the current setup it has **no consumer and no observable effect**:
1. The eval corpus is symmetric (`in_neigh == out_neigh` on every benchmark graph), so
   IN-masks and OUT-masks are identical — the OUT builder would change nothing.
2. SSSP/BC/BFS use **per-vertex** masks (`computeVertexMasks`), not the per-edge path, so
   nothing would consume OUT per-edge masks until a push kernel is converted to the
   per-edge demand path.
The reref-matrix direction (the load-bearing knob) is already per-kernel transpose-correct
(`ecgRerefTraverseCSR`). **When** a genuinely asymmetric directed benchmark and a
per-edge push kernel exist, build the OUT masks as a **non-owning transposed-view adapter**
over the existing IN builder (view where `out_neigh`↔`in_neigh`) and validate with a tiny
hand-constructed directed-graph oracle (symmetric-graph equality is only a weak smoke test).

---

## 6. IMPLEMENTED for BFS: dual-direction masking (2026-06-21)

Per the user request ("fully implement for BFS"), the dual-direction masking is now
in code, in two parts (rubber-duck `rd-bfs-mask-design`):

**(a) The load-bearing correctness fix (per-vertex path).** BFS's only masked read is
TD's `parent[v]` over `out_neigh(u)`; the next reader of `parent[v]` is `in_neigh(v)`,
so the transpose-correct rereference direction is **IN/CSC**. BFS now defaults to
`natural_csr=false` (was conservative CSR), making the existing per-vertex
`vertex_masks[v]` transpose-correct for TD. `ECG_BFS_FORCE_OUT` reverts for transfer
experiments. (Inert on the symmetric corpus; correct on directed graphs.)

**(b) The per-edge "mask both edge lists" capability.** `buildOutEdgeMasks(g)`
(`graph_cache_context.h`) is the self-contained OUT-edge mirror of `buildInEdgeMasks_PR`:
it stores into `out_edge_masks_by_src` / `out_edge_epoch_by_src`, derives each edge's
epoch from its own IN-adjacency arrays (`exact_in_off`/`exact_in_nbr`, built from
`g.in_neigh`), and **never touches the PR in-edge path or the shared `exact_off`** — so
PR stays byte-identical (verified POPT web-Google/512kB/o0 = 0.6591). `ECG_BFS_EDGE_MASKS=1`
builds it and makes TDStep carry the per-edge, **src-iteration-aware** epoch (the soonest
in-neighbour of dest > u) instead of the single per-vertex value — the one thing the
per-vertex mask cannot encode. BU masking is added in **§7** (2026-06-21 follow-up:
"BU and TD both should have their own masks").

**Validation.** `bench/src_sim/test_ecg_out_edge_mask.cc` validates the builder on a tiny
**directed** graph against a hand-computed oracle (edges 0→2=1, 1→2=3, 3→2=0, 2→4=2,
4→1=4); **mutation-proven** (flipping `in_neigh`→`out_neigh` fails all 5). Symmetric-graph
equality would have been only a weak smoke test, so the oracle is directed by construction.
On the symmetric eval corpus the per-edge path is ~inert (web-Google BFS hit-rate 0.8564
per-vertex vs 0.8579 per-edge — the small delta is the src-aware epoch), as expected; the
value is forward-looking correctness for directed graphs + the dual-mask capability the
user asked for. PR and all current headline numbers are unchanged.

---

## 7. IMPLEMENTED for BFS bottom-up: frontier-bitmap masking (2026-06-21)

Per the user ("we should have for bfs BU TD both should have their own masks why we left
it alone"), the bottom-up (BU) phase now carries its own mask, symmetric to TD. Rubber-duck
`rd-bu-mask-design`.

**What BU actually accesses (the crux).** Unlike TD (push), BU (pull) traverses
`in_neigh(u)` and its accesses are: `parent[u]` (SEQUENTIAL over u — regular, perfect
locality, no mask), the in-edge stream (streaming), and `front.get_bit(v)` — a **frontier
bitmap** membership probe. The bitmap probe is BU's *only* data-dependent/irregular access,
and it was **not cache-modeled** at all (a plain method call). That is why BU had "no masked
read": its irregular access is a compact bitmap, not a property array. There is no
`parent[v]`-style property read in BU to mask (modeling `parent[v]` would defeat the whole
point of direction-optimizing BFS, which uses the compact bitmap *to avoid* TD's property
traffic — rejected).

**What was implemented.** Under `ECG_BFS_EDGE_MASKS` (the same gate as TD), BU now models
the frontier probe as a cache access — `SIM_CACHE_READ_MASKED` on `front.data()[v/64]` (the
real address of v's bitmap word; `Bitmap::data()` is a new read-only accessor) — carrying
the **IN-edge** per-edge epoch. The direction is the mirror of TD: v's frontier bit is next
read when v's next `out_neigh(v) > u` is processed, so the transpose-correct epoch is derived
from `g.out_neigh`. `buildInEdgeMasks(g)` is the **generic** self-contained IN-edge
(inverse-graph) mirror of `buildOutEdgeMasks` (LOCAL sorted out-adjacency; fills only
`in_edge_*`; fills the epoch **unconditionally**, unlike `buildInEdgeMasks_PR` which only
fills it under `ECG_EDGE_MASK_EPOCH`). So with the flag set: **TD uses OUT-edge masks, BU
uses IN-edge masks** — both phases masked per their own edge list.

**Gating + no regression (verified).** The masked bitmap read is strictly inside
`if (use_in_edge_masks)`, which is false unless `ECG_BFS_EDGE_MASKS` built the IN masks, so
the default BFS access stream is unchanged. A/B clean-vs-changed binaries:
PR L3 misses 26,131,464 == 26,131,464 (byte-identical); default BFS L3 misses 30,016 ==
30,016 (byte-identical). With the flag on, both `OUT-edge` and `IN-edge` mask builds fire and
both phases are masked. (BFS verification FAILs identically with and without the change — a
pre-existing cache-sim BFS harness condition, not a regression.)

**Validation.** `bench/src_sim/test_ecg_in_edge_mask.cc` is the directed-graph oracle
(mirror of the out-edge oracle): in-edge epochs derived from `out_neigh(dest)`. A single
`dest=2` reached from `src=1,3,4` yields **three different** epochs (3, 4, 1) — proving the
epoch is src-iteration-aware, not a per-vertex constant. 5/5 pass.

**HONEST CAVEATS (rubber-duck `rd-bu-mask-design`).**
1. **Granularity.** A 64 B cache line holds 8 words = **512 vertices'** frontier bits, so the
   line's true reuse is the min next-ref over ~512 vertices; the per-edge v-epoch is the same
   *kind* of approximation TD makes (16 vertices/property-line) but **coarser**. The IN-edge
   *signal* (when is v's bit next read) is correct; only the line granularity is loose.
2. **Value.** The frontier bitmap is small (n/8 bytes — web-Google ~114 KB) and LLC-resident
   **by design** (BU exists to avoid property traffic), and its lines are uniformly hot (each
   512-vertex line almost always has a near-future reader), so BU masking is **do-no-harm with
   ~nil measurable benefit** on the eval corpus. Additionally the bitmap is not a registered
   property region, so the ECG_GRASP_POPT epoch tiebreak (property-only) does not act on it;
   the mask mainly affects insertion RRPV. This is a **symmetry/completeness** feature (like
   the inert dual-direction masks on the symmetric corpus), not a headline-mover.
3. **Scope.** Inert on the symmetric eval corpus (in==out); the dual mask matters only on
   directed graphs. Default BFS (flag off) and all headline numbers are unchanged.

---

## 8. The per-edge masking is GENERIC (inverse/main graph), not BFS-specific

The dual-direction per-edge masking is a **generic "mask per edge list"** capability, not a
BFS feature. There are two edge lists and one mask set each:

| Edge list | Graph | Builder | Storage | Transpose epoch source |
|-----------|-------|---------|---------|------------------------|
| OUT (CSR) | **main** graph (`g.out_neigh`) | `buildOutEdgeMasks(g)` | `out_edge_*_by_src` | `g.in_neigh(dest)` |
| IN (CSC)  | **inverse** graph (`g.in_neigh`) | `buildInEdgeMasks(g)` *(plain)* or `buildInEdgeMasks_PR(g,k)` *(+prefetch/EXACT/EPOCH/PACK)* | `in_edge_*_by_src` | `g.out_neigh(dest)` |

The builders contain **zero kernel-specific logic** — only direction math (the next-ref of a
datum read via direction D is the graph transpose, so OUT-edge epochs come from `in_neigh`
and IN-edge epochs come from `out_neigh`). A kernel simply consumes the mask set for
whichever graph it traverses:

| Kernel / phase | Traverses | Reads | Mask set used |
|----------------|-----------|-------|---------------|
| PR (pull)      | `in_neigh`  | `property[dest]`   | IN-edge (`buildInEdgeMasks_PR`) |
| CC (Afforest)  | `in_neigh`  | `comp[dest]`       | IN-edge (`buildInEdgeMasks_PR`) |
| BFS top-down (push)   | `out_neigh` | `parent[v]`        | OUT-edge (`buildOutEdgeMasks`) |
| BFS bottom-up (pull)  | `in_neigh`  | frontier bit of v  | IN-edge (`buildInEdgeMasks`) |
| SSSP relax (push)     | `out_neigh` | `dist[v]`          | OUT-edge (`buildOutEdgeMasks`) |
| BC forward (push)     | `out_neigh` | `depths[v]`,`path_counts[v]` | OUT-edge (`buildOutEdgeMasks`) |
| BC backward           | `succ` DAG  | `path_counts`,`deltas` | per-vertex (compacted DAG, no static edge list) |

So PR and CC use the **inverse-graph** masks; BFS/SSSP/BC use the **main-graph** (OUT) masks
for their push reads (BFS also IN for BU). The only genuinely BFS-specific piece in §7 is
*modeling the frontier-bitmap access* (the bitmap is unique to BFS) — the **masks themselves
are generic**. The earlier `buildInEdgeMasksBFS` name was misleading and has been renamed to
`buildInEdgeMasks`.

### §8.1 SSSP/BC wiring (2026-06-21, for the final matrix run)

`ECG_EDGE_MASKS` (generic, single matrix knob; per-kernel aliases `ECG_BFS_/SSSP_/BC_EDGE_MASKS`)
now switches **SSSP** (`RelaxEdges_Sim`, both `dist[wn.v]` reads) and **BC forward**
(`depths[v]`+`path_counts[v]`) from per-vertex to per-edge OUT masks — a 1:1 mirror of BFS-TD.
BC's backward phase keeps per-vertex accesses (the `succ`/`succ_start` successor DAG is a
runtime-built, source-dependent compaction, not a static edge list, so there is no stable
per-edge position to mask).

**Sticky-epoch hygiene (important).** `cache_sim.h` stamps a filled line's `ecg_epoch` from
`hints.edge_epoch` on **every** `ECG_GRASP_POPT` fill. Because `edge_epoch` is a sticky
per-thread hint, a SEQUENTIAL source read (`dist[u]`/`depths[u]`/`parent[u]`) issued between
edge-masked reads would otherwise inherit the *previous* vertex's last edge epoch. So each
kernel resets `hints.edge_epoch = 0` (guarded by `!out/in_edge_masks_by_src.empty()`, i.e.
no-op when masks are off) before its source reads: BFS-BU + BC-forward + BC-backward at the
top of the outer loop body, SSSP at the end of `RelaxEdges_Sim` (its source read is in the
caller). BFS-BU additionally resets before `SIM_CACHE_WRITE(parent[u])` because that write
targets the OUTER vertex u (not the masked dest v), so it must not carry v's frontier epoch
(rubber-duck rd-sbm-impl). Verified flag-off byte-identical (SSSP/BC/BFS L3 misses unchanged)
and flag-on verification PASS.

**Known limitation (prefetch + edge masks).** The runtime prefetch block (`ECG_PREFETCH_LOOKAHEAD>0`)
runs *before* the current edge's epoch is set, so a prefetched line inherits the previous
edge's (or zero) epoch — a second-order imprecision present identically across all per-edge
kernels (pre-existing, not specific to this wiring). Demand reads are unaffected.

**Caveats.** (1) On the symmetric eval corpus in==out, so these masks are **inert** (no result
change) — value is matrix completeness + forward-looking directed support, not a new headline.
(2) BC builds the OUT masks **per source** (`BCBFS_Sim` runs per source and already rebuilds
`makeOffsetMatrix` per source — same O(E)); acceptable for small BC source counts, hoist/cache
if it becomes a preprocessing bottleneck on large graphs.

### §8.2 SSOT consumption helpers (2026-06-21)

The per-edge consumption logic (build-readiness guard + `edgeMaskPOPT` extraction +
next-ref epoch stamp + sticky-epoch reset) was copy-pasted across BFS-TD/BU, SSSP, BC.
It is now a **single source of truth** in `GraphCacheContext` (the owner of both mask
sets), so a kernel just names the edge list it traverses:

```cpp
enum class EdgeMaskDir { OUT, IN };                       // main / inverse graph
bool     edgeMaskReady(dir, src, degree) const;           // is this dir's row built+sized?
uint32_t resolveEdgeMaskAndEpoch(dir, src, degree, edge_pos, vertex_fallback);  // mask + sets epoch
void     clearEdgeEpoch();                                 // sticky-epoch hygiene
```

Each push kernel collapses to one call per edge:
`m = graph_ctx.resolveEdgeMaskAndEpoch(EdgeMaskDir::OUT, u, out_degree, edge_pos, vertex_masks[v]); SIM_CACHE_READ_MASKED(...)`.
This directly realizes the design intent — *"both edge lists are separate; when an
algorithm needs the in-degree edge list the mask is ready"* — a future pull kernel uses
`EdgeMaskDir::IN` and the IN-edge mask is served with no new plumbing. `edgeMaskReady` is
used where the masked **access itself** is conditional (BFS-BU's frontier probe, which has
no cache read when masks are off — so the gate must stay to keep the default stream).

**Pure refactor — verified byte-identical** to the pre-refactor commit in *both* flag
states (flag-off AND flag-on): BFS 29710/30579, SSSP 4096/4096, BC 1990743/1891741 (pre==post);
PR anchor 26131464 unchanged; oracles 5/5. PR's specialized mode-6/7 path (LEAN/PACK/charging/
record-width) is intentionally **not** routed through the helper (it models extra traffic the
generic helper would hide). Builder unification (`buildOutEdgeMasks`+`buildInEdgeMasks` are
near-identical mirrors) was **deferred** as headline-adjacent/transpose-sensitive (rubber-duck
rd-ssot) — a future clean-up once the consumption SSOT has settled.

---

## 9. P-OPT rereference matrix: SSOT build + real-time per-direction load (2026-06-21)

"Do we have the same [dual-direction] for P-OPT?" — investigated + rubber-ducked
(rd-popt-dual). Two parts:

### §9.1 SSOT build+register helper (the genuinely valuable cleanup)
The reref build+register block was copy-pasted across **5 kernels** (PR/BFS/SSSP/BC/CC):
`makeOffsetMatrix(...) + numCacheLines + initRereference(...) + exact_vtx_per_line`. It is
now SSOT in `popt.h`:
```cpp
buildRerefMatrix(g, natural_csr, kernel, vtxPerLine, numEpochs, storage);        // build only
buildAndRegisterReref(g, ctx, natural_csr, kernel, vtxPerLine, numEpochs, storage); // + register
```
`buildAndRegisterReref` is duck-typed on the context, so `popt.h` keeps no cache_sim
dependency. Verified byte-identical (POPT): PR 26131464, BFS 30012, SSSP 4096, BC 2176466,
CC 4096 (pre==post). The kernel-specific `ECG_EXACT_REREF` block stays per-kernel.

### §9.2 Real-time per-direction load (POPT_DUAL_REREF) — and why it's forward-looking
`GraphCacheContext::setActiveRerefMatrix(const uint8_t*)` repoints the **single** reserved
reref way at a pre-built matrix of the same dims (the matrix is non-owned → a pointer swap).
Under `POPT_DUAL_REREF` (default off), BFS pre-builds **both** the TD matrix (in-transpose /
CSC) and the BU matrix (out-transpose / CSR) and swaps the active one per phase — keeping the
1-way cost model (vs reserving a 2nd way).

**HONEST FINDING (rubber-duck rd-popt-dual): this is NOT analogous to the ECG dual edge
masks, and it does NOT improve `parent[]` for BFS.** The asymmetry:

| | manages in BU | BU access pattern |
|---|---|---|
| ECG edge masks | the frontier probe `front.get_bit(v)` | **irregular** (v ∈ in_neigh) → dual mask meaningful |
| P-OPT reref | the registered property `parent[]` | **sequential** (`parent[u]`, u in ID order) → edge-list matrix moot |

In BU the only edge-list-driven access is the frontier **bitmap** (handled by the IN-edge
masks); `parent[]` is read in plain sequential ID order. So neither CSC nor CSR models BU-
`parent[]` reuse (a sequential/identity model would). Activating CSR during BU therefore does
not make `parent`'s P-OPT management more correct — on directed graphs it would feed P-OPT an
edge-list oracle unrelated to the sequential parent stream. `parent[]` is *already* correctly
P-OPT-managed: TD uses the transpose-correct CSC for its irregular `parent[v]`; BU-parent is
sequential (direction-independent).

So `POPT_DUAL_REREF` is **forward-looking**: the mechanism is ready for a future direction-
optimizing kernel whose property access is **irregular in both directions** (BFS is not such a
kernel). It is **inert on the symmetric corpus** (in==out → CSR==CSC → swap is a no-op;
verified BFS POPT_DUAL_REREF on==off, L3 misses 30012==30012). The mechanism itself is proven
on a **directed** graph by `bench/src_sim/test_popt_dual_reref.cc` (CSC≠CSR; swap repoints) —
5/5. Default-off keeps every headline path byte-identical.

### §9.3 Functional verification (is the swap live?)
`setActiveRerefMatrix` increments `reref_swap_count` (BFS prints it under POPT_DUAL_REREF),
so the real-time load is observable. Evidence the mechanism is FUNCTIONAL — it fires AND
takes effect:
- **Swap fires:** BFS on web-Google.sg / soc-pokec enters BU and reports `loads this run = 2`
  (swap to the BU matrix and back to TD).
- **Takes effect on directed graphs:** BFS POPT on the DIRECTED soc-pokec.el (1.6M nodes;
  TD=CSC, BU=CSR genuinely differ) gives L3 misses **OFF=1,647,565 vs ON=1,648,167** — the
  swap changes the result.
- **Inert on the symmetric corpus:** web-Google.sg OFF==ON (1,309,233) even though the swap
  fires (count=2) — because in==out → CSR==CSC (identical content).
- **Not beneficial for BFS:** the directed effect is slightly WORSE (+602 misses, +0.04%),
  exactly as predicted — CSR-during-BU is a *less* faithful oracle for the SEQUENTIAL
  `parent[]` than the TD-correct CSC. Confirms §9.2: functional mechanism, forward-looking
  value (a future kernel with irregular property in both directions would benefit; BFS does
  not). NOTE: the flag is presence-based (`getenv != nullptr`), so `POPT_DUAL_REREF=0` still
  ENABLES it — UNSET the var to disable.

### §9.4 Correctness verification
The dual-reref swap is correct (does the right thing, never corrupts the algorithm):
- **Reref machinery correct:** SSSP and BC verify **PASS** with `CACHE_POLICY=POPT` (the
  SSOT `buildAndRegisterReref` builds/registers/consumes the matrix correctly).
- **Transpose-correct per phase:** the build logs show TD activates `IN/CSC(in_neigh)` and
  BU activates `OUT/CSR(out_neigh)` — the transpose of each phase's traversal; the unit test
  (`test_popt_dual_reref.cc`) proves the two matrices differ on a directed graph and the swap
  repoints to the right one.
- **Algorithm output unaffected:** the reref matrix is READ-ONLY cache-eviction metadata
  (`setActiveRerefMatrix` only repoints `rereference.matrix`; `findNextRef` only reads it for
  eviction priority — it never touches `parent[]`). BFS's verification result is IDENTICAL
  with the swap on vs off, so the swap cannot change the BFS tree.
- **BFS now verifies PASS (a real pre-existing bug, FIXED):** the cache-sim `DOBFS_Sim` was
  missing the canonical GAPBS parent finalization — unreached vertices kept InitParent's
  `-out_degree(n)` encoding (< -1) instead of `-1`, so the verifier's `depth[u]==parent[u]`
  reachability check FAILed on any graph with unreached vertices (independent of P-OPT/reref/
  dual — plain LRU FAILed too). Added the finalization loop (matches `bench/src/bfs.cc`); it is
  post-processing with NO cache accesses, so cache stats are byte-identical (BFS POPT flag-off
  L3 misses 30012 unchanged). BFS now verifies PASS for LRU/POPT/GRASP/ECG_GRASP_POPT across
  directed (soc-pokec.el) + symmetric (web-Google.sg, cit-Patents) graphs — and crucially
  **POPT_DUAL_REREF=1 verifies PASS** on directed soc-pokec, positively confirming the dual
  swap is algorithm-correct (not merely "identically failing").

---

## 10. Cross-simulator equivalency (cache_sim / gem5 / Sniper) — 2026-06-21

Rubber-duck `rd-equiv`. The three simulators have SEPARATE kernel trees
(`bench/src_sim/`, `bench/src_gem5/`, `bench/src_sniper/`) and deliver ECG metadata by
different mechanisms (cache_sim = host-side mask arrays; gem5/Sniper = hardware ISA path,
`ecg.extract` / packed edge word). Equivalency is about the SHARED decision + correctness
contract, not identical kernels.

### VERIFIED
- **Shared ECG decision parity:** `ecg_mode6::selectPrefetchTarget` (used by all 3 sims) +
  the packMask field layout are unchanged this session; `verify_pfx.py` synthetic
  shared-decision test = **10/10** ("covers cache_sim + gem5 + Sniper"), live prefetch checks
  pass.
- **This session is cache_sim-only:** no change to `ecg_mode6_builder.h`, `bench/src_gem5/*`,
  `bench/src_sniper/*`, or `bench/include/gem5_sim/*` (git log over the session range is empty
  for those paths). The cache_sim SSOT refactor was byte-identical, so cache_sim still consumes
  the same mask bits (`edgeMaskPOPT` = [26:33], separate epoch) it did before.
- **BFS/SSSP verification now PASSES across ALL 3 sims** (it FAILed in all 3 before, for TWO
  different reasons):
  - cache_sim FAILed from a missing parent finalization (direction-optimizing GAPBS keeps the
    `-out_degree` unvisited encoding) — FIXED (§9.4).
  - gem5/Sniper FAILed from a verify-harness **source mismatch**: `BFSBound`/`SSSPBound` and
    `VerifyBound` shared ONE `SourcePicker`, so the kernel and verifier drew DIFFERENT sources.
    cache_sim/canonical use TWO pickers seeded identically (`sp` + `vsp`). FIXED by adding the
    second `vsp` picker to gem5/Sniper `bfs.cc` + `sssp.cc` (verified host builds PASS without
    `-r`). The kernel is unchanged, so measured cache/cycle results are unaffected — only the
    correctness check is corrected.

### OPEN / LIMITED (known, documented)
- **Full gem5/Sniper RUNTIME parity is NOT proven by the synthetic test** — `verify_pfx.py`
  validates the pure target-selection decision; real parity also needs actual gem5/Sniper
  sim runs + artifact checks of ISA decode, field packing, and emitted hints.
- **BFS algorithm differs structurally:** cache_sim is direction-optimizing (TD push + BU pull,
  bitmap frontier, in-neigh traversal in BU); gem5/Sniper are SIMPLE TD-only queue BFS. So BFS
  *cache behavior* is NOT cross-sim equivalent (different access stream / mask & prefetch
  opportunity). This is pre-existing and does not affect the PR headline. To compare BFS across
  sims, either port DOBFS to gem5/Sniper or force cache_sim TD-only (`ECG_BFS_FORCE_TD`).
- **gem5/Sniper BFS verifier is LENIENT** (checks only reachability sign `parent[n] < 0`),
  vs cache_sim/canonical strict (parent is a valid depth-1 predecessor). The simple BFS is
  correct (passes the strict contract), but a future wrong tree could slip past the lenient
  check. Porting the strict verifier is deferred (needs `in_neigh` availability in those
  kernels) and flagged here.
- **gem5 ECG_PFX PREFETCH: the 15-bit truncation is FIXED (§10.2 Path B + §11 Path A); epoch
  eviction was always faithful (VERIFIED, Phase 0 audit `rd-eqh-plan`):** two gem5 delivery paths:
  - **Epoch eviction (the headline ECG_GRASP_POPT mechanism):** delivered via a packed-flat
    4-byte record `(dest | epoch<<id_bits)` — web-Google: 20-bit dest + 12-bit epoch = 32 bits,
    "packed record ON", **NO truncation**, faithful at scale.
  - **Prefetch target (history → fixed):** was a 15-bit `packMaskEpoch` field [49:64] → silent
    truncation on >32767-vertex graphs (web-Google **99.5%**, soc-pokec **97.0%**, com-orkut
    **98.9%**). Path B widened to 24 bits (≤16M ids, §10.2, actual-sim validated); Path A packs
    only the 12-bit epoch and prefetches the streamed edge id, so it is **size-independent**
    (§11, actual-sim validated). `ECG_PFX_STRICT_TARGET=1` still aborts on any >24-bit residual.
    **Current validity matrix (eviction + the two prefetch mechanisms):**

    | feature | cache_sim | gem5 | Sniper |
    |---------|-----------|------|--------|
    | epoch eviction (headline replacement) | yes | yes (packed-flat) | yes |
    | DROPLET baseline prefetch | yes | yes | yes |
    | Path B prefetch (single selective target) | yes (31-bit) | yes (24-bit, ≤16M) | yes (8-byte) |
    | Path A prefetch (epoch-filtered next-K lookahead, **headline**) | yes | yes (§11) | **yes (§13)** — `SNIPER_ECG_EXTRACT` + batch-drain |

  **Sniper faithfulness (§12/§13 UPDATE):** the per-edge epoch delivery (`SNIPER_ECG_EXTRACT`,
  opt-in) is IMPLEMENTED + VALIDATED, so Sniper's ECG_GRASP_POPT eviction is delivery-faithful
  like gem5/cache_sim (its host-side `findNextRef` matrix remains the default/oracle path). On
  that foundation, **Sniper Path A is now IMPLEMENTED + mechanism-validated (§13)**: the kernel
  emits the next-K epoch-filtered survivors (each carrying its epoch via `SNIPER_ECG_EXTRACT`
  so the prefetch-filled line is stamped) and the `ecg_pfx` prefetcher batch-drains them. Path A
  consumes ~17× more target hints than Path B (next-K vs single), distinct and with no ring
  starvation. Known limitation: `pf_issued=0` on the no-pressure validation cell is the
  pre-existing Sniper L2 enqueue filter (§13); gem5 is the cycle-accurate Path A reference with
  real fills, cache_sim is authoritative for traffic.
- **cache_sim-only research features** (BU frontier masks, SSSP/BC OUT masks, `POPT_DUAL_REREF`)
  have NO gem5/Sniper analog and are inert on the symmetric corpus / default-off. They are
  equivalent across sims ONLY because they are disabled or demonstrably inert for the evaluated
  configs; if ever used for cross-sim numbers they would need an ISA-path analog.

### §10.1 Cross-sim validity contract (equivalency-hardening sprint, 2026-06-21)

Actionable rules for any cross-simulator (cache_sim / gem5 / Sniper) comparison, derived
from the Phase 0 audit + the parity tests:

1. **PR epoch eviction (the headline ECG_GRASP_POPT mechanism): all 3 sims valid.** gem5
   delivers the epoch via the packed-flat 4-byte record (dest + epoch), faithful at scale.
2. **ECG_PFX prefetch:** all 3 sims valid. **Path B** (single selective target): cache_sim/Sniper
   31-bit, gem5 24-bit (≤16M ids, §10.2, validated). **Path A** (epoch-filtered next-K lookahead,
   the headline): cache_sim + gem5 (§11, size-independent) + **Sniper (§13, `SNIPER_ECG_EXTRACT` +
   batch-drain, mechanism-validated; fills gated by the pre-existing Sniper L2 enqueue filter —
   gem5 is the cycle-accurate Path A reference with real fills, cache_sim authoritative for traffic).
   `ECG_PFX_STRICT_TARGET=1` still guards any residual gem5 >24-bit truncation.
3. **BFS is NOT cross-sim comparable for cache behavior** (cache_sim is direction-optimizing
   TD+BU; gem5/Sniper are simple TD-only). Use cache_sim `ECG_BFS_FORCE_TD` for a smoke-level
   comparison only; do NOT report BFS as a cross-sim result. PR is the cross-sim headline.
4. **cache_sim-only research features** (`ECG_EDGE_MASKS` BU/SSSP/BC masks, `POPT_DUAL_REREF`)
   must be OFF (default) for cross-sim runs — they have no gem5/Sniper analog.
5. **Verification:** after this session all 3 sims verify PASS (cache_sim parent finalization
   §9.4; gem5/Sniper verify source-mismatch fix). gem5/Sniper keep a lenient reachability-sign
   verifier (the simple BFS is correct; strict-verifier port deferred).

**Parity evidence (VERIFIED):**
- Decision parity: `verify_pfx.py` synthetic `selectPrefetchTarget` = 10/10 (all 3 sims).
- Field-delivery parity: `bench/src_sim/test_ecg_packed_field_parity.cc` = **48/48** — pins the
  31-bit (`packMask`), the legacy 15-bit (`packMaskEpoch`, truncates >32767), and the 24-bit wide
  (`packMaskEpochWide`, ≤16M) layouts against silent repack regressions.
- Runtime mechanism parity: gem5 ECG_PFX validated end-to-end at scale (kron_s16_k4 = 65,536
  verts >32K): Path B 97% useful, Path A 82.6% useful, no truncation (§11); epoch eviction
  faithful at scale (packed-flat). **Sniper Path A (§13):** mechanism-validated — next-K
  epoch-filtered lookahead fires (226K hints vs Path B 13K, ~17×, no ring starvation); fills
  gated by the pre-existing Sniper L2 enqueue filter (gem5 = cycle-accurate Path A with fills).

### §10.2 IMPLEMENTED + actual-sim VALIDATED: fix the gem5 ECG_PFX 32K limit by reclaiming vestigial mask bits

**Status (2026-06-21):** IMPLEMENTED (commit `3fc8eb6b`) and actual-sim VALIDATED — the Path B
wide pfx-target [40:64] (24-bit) was exercised on a >32K-vertex graph (kron_s16_k4 = 65,536)
with `ecg.extract` instruction delivery: 229,673 prefetches, **222,840 useful (97%)**, no
truncation (a 15-bit-clamped target would collapse useful-rate — see §11). Field-parity test
48/48; overlay-hash registry (incl. `decoder_ecg_extract.isa`) refreshed.

Original proposal/analysis below.

The gem5 ECG_PFX 15-bit prefetch-target limit (§10, ~97-99.5% truncation on headline graphs)
is fixable by reclaiming the **vestigial POPT field** — the epoch is the only load-bearing
eviction field (ECG_GRASP_POPT uses `ecg_epoch`, cache_sim.h:1256; `ecg_popt_hint` is read
only by the legacy `ECG_EMBEDDED` mode, :1662). Rubber-duck `rd-widen`.

**Proposed (gem5-only) wide layout** — a SEPARATE `packMaskEpochWide`, NOT a mutation of
`packMaskEpoch` (which `ECG_EMBEDDED`/`ECG_COMBINED` still need on gem5):
```
packMaskEpochWide (64-bit):  dest[0:24] | epoch[24:40] (16) | pfx[40:64] (24)
```
Reclaims popt(7) + dbg(2) (+ shift) -> prefetch target **15 -> 24 bits = 16,777,216 ids**,
covering every current headline graph (web-Google 916K, soc-pokec 1.6M, com-orkut 3M,
kron-s24 = 2^24 -> max id 16,777,215 fits) with **NO wider record / zero extra bandwidth**.
`packMask` (cache_sim/Sniper, 31-bit) and the shared constants (`kPrefetchShift`,
`extractPrefetchTarget`) stay UNCHANGED.

**Coordinated change required (any mismatch silently recreates the truncation bug):**
1. `bench/include/ecg_mode6_builder.h` — add `packMaskEpochWide` + `extractPrefetchTargetWide`.
2. `bench/src_gem5/pr.cc` — repack/extract via the wide layout + the truncation guard.
3. `bench/include/gem5_sim/overlays/arch/riscv/.../decoder_ecg_extract.isa` — the `ecg.extract`
   ISA op decodes the new bit position/width in lockstep.
4. `bench/src_sim/test_ecg_packed_field_parity.cc` — extend to pin the wide layout.
5. overlay-hash registry (`lit_faith_*_overlay_hash_registry.py --update`) if enforced.
Guards: abort if `dest >= 1<<24` or `pfx >= 1<<24`; note the `pfx==0`="no prefetch" sentinel
(vertex 0 cannot be a prefetch target — pre-existing).

**Alternatives (deferred):**
- **Relative offset** (encode "which of src's next-K in-neighbours", ~6 bits, graph-size-
  independent): elegant + fits the current field, BUT the gem5 prefetcher (`ecg_pfx.cc`) has no
  neighbour-list context to resolve an offset — needs the kernel to resolve offset->absolute or
  a prefetch-interface redesign. Promising future design, larger than it looks.
- **16-byte record** (>16M verts): NOT worth it now — `ecg.extract` takes ONE 64-bit register;
  128-bit is an ABI expansion (two registers/instructions), and doubles the mask stream on the
  bandwidth-sensitive large-graph path.

**DECISION (taken 2026-06-21):** implemented — the user chose full 3-sim PFX parity ("both are
headline variants"), so gem5 cycle-accurate ECG_PFX on >32K headline graphs is now first-class
(Path B wide target here + Path A in §11). The §10.1 validity-matrix routing
(cache_sim/Sniper authoritative for large-graph PFX; gem5 faithful for epoch eviction at scale)
remains the documented fallback.

## 11. The TWO ECG prefetch mechanisms + gem5 Path A port (2026-06-21)

User question: *"are we using the 24-bit pfx field to prefetch, or the edge list with
epoch as a filter?"* The answer is **both are real, distinct headline mechanisms**, and
before this work gem5 only had one of them.

### The two mechanisms (cache_sim `bench/src_sim/pr.cc`)

| | **Path A — epoch-filtered DROPLET lookahead** | **Path B — single packed target** |
|---|---|---|
| Where | `pr.cc:241-285` (`ECG_EDGE_MASK_LEAN` + `ECG_EDGE_MASK_PREFETCH=K`) | `pr.cc:287-295` (fat-mask) |
| Targets | the **next-K in-neighbours from the edge stream** (DROPLET-style), each kept/dropped by an **epoch filter** | **one** POPT-best `selectPrefetchTarget`, packed into the mask |
| Target width | the **streamed edge id** (full graph width) — only the **12-bit epoch** is packed | the packed prefetch field (15-bit → 24-bit after §10.2) |
| Size limit | **NONE** (target is the streamed edge; size-independent) | yes — bounded by the packed field width |
| Headline | the combined-stack bandwidth win ("epoch-stamped lookahead beats DROPLET") | the selective ECG_PFX (fewer fills than DROPLET) |

So Path A is exactly "the edge list with epoch as a filter" and it **has no 32K/16M
limit** — the §10.2 widening fixed the *secondary* Path B field; the headline Path A never
had that limit. (This realises the "relative offset / edge-list" alternative flagged in §10.2.)

> **§14 update:** Path A stores **nothing** for prefetch — the next-K targets are the CSR
> edge stream (= DROPLET's read-ahead) and the prefetched line's epoch comes from the same
> edge entry. The Path B stored prefetch-target field is therefore **dead weight** and is
> dropped from the honest record (`packMaskEpochOnly`, no prefetch field); the epoch saturates
> at ~10 bits. See §14 for the HW-scalability + bit-budget analysis.

### What each simulator ran (before this work)
- **cache_sim**: BOTH. Headline = Path A.
- **gem5**: **Path B only** (both its prefetch loops select ONE target → `GEM5_ECG_PFX_TARGET`).
- **Sniper**: full 8-byte mask target (Path B style).

→ gem5 did **not** run the headline Path A. Fixed below for full 3-sim equivalency.

### gem5 Path A implementation (HW-faithful, rule-1 clean)
Gated by `ECG_EDGE_MASK_PREFETCH=K` in the mode-6 kernel (`bench/src_gem5/pr.cc`), mirroring
cache_sim `pr.cc:241-285`: walk the next-K in-neighbours, read each candidate's epoch from
the packed-flat record, apply `ECG_PREFETCH_EPOCH_FILTER`/`_THRESH_PCT`, emit each survivor.

The hard part is HW-faithfully giving a **prefetched** line its candidate's epoch so it evicts
correctly (cache_sim stamps the per-line `ecg_epoch` synchronously; gem5's prefetch FILL is
async, and the in-order `ecg.extract` single-slot mailbox is stale by then). The fix is **NOT**
an O(V) per-vertex table (that is the P-OPT-class cost ECG avoids — rejected on rule 1):

1. **Per-line epoch tag** (`EcgReplData::ecg_epoch`, already present) — HW-realizable, ~12 bits/line.
2. **Dedicated `(target,epoch)` hint** — new `GRAPHBREW_ECG_PFX_TARGET_EPOCH_WORK_ID`
   (`graph_cache_context_gem5.hh`), `threadid = target | epoch<<32` via
   `GEM5_ECG_PFX_TARGET_EPOCH` (m5op). Distinct from the fat-mask work-id (no `>>24`
   ambiguity, **no single-slot/demand-epoch corruption, no 24-bit truncation**).
3. **Bounded in-flight prefetch-epoch buffer** (`recordPendingPrefetchEpoch` /
   `consumePendingPrefetchEpoch`, 256-entry direct-mapped, drop-counter) — models the
   prefetch engine "carrying the epoch it read from the edge word"; sized like an MSHR /
   prefetch-metadata array, **NOT O(V)**.
4. **`ecg_rp.cc reset()`**: for ECG_GRASP_POPT, when the single-slot misses (i.e. a prefetch
   fill), recover the carried epoch from the in-flight buffer and stamp the line.
5. **`ecg_pfx.cc`**: batch-drain up to `GEM5_ECG_PFX_DRAIN_BATCH` (default 8) hints/call
   (DROPLET-style multi-address-per-call) so K-per-edge pushes are issued, not dropped.

`setup_gem5.py` reproduces the new `pseudo_inst.cc` work-id handler. Path B (single 24-bit
target) is untouched and mutually exclusive (taken only when `ECG_EDGE_MASK_PREFETCH=0`).

### Validation (gem5 RISCV, kron_s16_k4 = 65,536 verts, L3=128kB, ECG_GRASP_POPT)
**Env-forwarding fix (required):** gem5 SE mode does NOT inherit the host env — a kernel
`getenv()` reads the simulated process env, which `graph_se.py` builds from an explicit
allowlist. `ECG_EDGE_MASK_PREFETCH` (+ `ECG_PREFETCH_EPOCH_FILTER`/`_THRESH_PCT`) had to be
added there, or the kernel silently falls back to Path B (`lean_pfx_k=0`).

Three runs, all status=ok, clean exit:

| run | L3 miss-rate | IPC | pf_issued | pf_useful |
|---|---|---|---|---|
| eviction-only (no pfx) | 0.4013 | 0.2185 | — | — |
| **Path B** (single wide target, §10.2) | 0.3969 | 0.2323 | 229,673 | 222,840 (97%) |
| **Path A** (epoch-filtered K=8) | **0.3406** | **0.2892** | **794,333** | **656,183 (82.6%)** |

- Path A is byte-distinct from Path B (first target 5338 vs 19876; simTicks 595B vs 219B;
  ~3.5× more prefetches), achieves the **lowest L3 miss-rate and highest IPC** — the headline.
- **No-truncation proof:** 82.6% useful on 794K Path A prefetches (and 97% on Path B's wide
  target) over a **>32K-vertex** graph — a 15/24-bit-clamped target would hit wrong lines and
  collapse useful-rate. Path B's run also validates §10.2's wide pfx-target in actual sim.
- No-regression: eviction-only is clean and unchanged (the reset() pending fallback only fires
  on prefetch fills).

→ gem5 now runs **both** ECG prefetch mechanisms; Path A (the headline edge-list+epoch-filter
lookahead) is graph-size-independent and HW-faithful with a bounded (non-O(V)) in-flight buffer.

## 12. IMPLEMENTED + VALIDATED: Sniper delivery-faithful epoch (SNIPER_ECG_EXTRACT, 2026-06-21)

Sniper's ECG_GRASP_POPT eviction can now use a per-edge epoch **delivered through the memory
hierarchy** (HW-faithful, like gem5/cache_sim) instead of the host-side `findNextRef` matrix
oracle — closing the faithfulness gap §10 documented. Opt-in via `SNIPER_ENABLE_ECG_EXTRACT`
(default = the existing `findNextRef` path, untouched). Naming mirrors gem5
(`GEM5_ENABLE_ECG_EXTRACT`) for SSOT.

**Mechanism:** kernel `SNIPER_ECG_EXTRACT(dest, epoch)` per demand edge → `notify_user` →
`magic_server.cc` dispatch → bounded per-core epoch map (`recordEcgEpoch`); the cache stamps the
property line's `m_ecg_epoch[]` at fill + re-stamps at eviction from the current map (non-invasive
refresh, no `cache.cc` core patch). The eviction ranks property lines by
`dist=(epoch+ne−cur_ep)%ne` (cur_ep from `currentVertexForPopt`), `stamped=isProp&&valid`; the
shared `ecg_policy::selectVictim` DECISION is unchanged. Epoch map keyed per-vertex; the lookup
scans the line's `blocksize/elem` vertices (kNumVtxPerLine=16; linemin ⇒ all agree).

**Validation (Sniper, full `pr` binary — NOT pr_kernel_smoke — email-Eu-core, ECG_GRASP_POPT,
L3=2kB):** delivered epoch **correct and varying** — `u=0→0, 150→39, 300→76, 450→115, 600→153,
750→191, 900→229`, i.e. exactly `u·ne/N` (ne=256, N=1005), the same next-ref model gem5/cache_sim
use. Property-fill stamp rate **100%** (19998/20000) after the code-review fix (see below) — BOTH
the `scores` and `contrib` property arrays carry the delivered epoch. Compiles end-to-end
(snipersim + kernel). Gotchas found: `--sniper-workload pr_kernel_smoke` builds a DIFFERENT binary
without the emit (use `benchmark` + `--allow-sniper-benchmark-workload`); the `[EVICT L3]` trace's
`epoch=` column is the property MARKER, the epoch value is in `dist=`.

**Code-review fix (fcbf6870):** `vertexForAddress` originally checked only `regions[0]` (=`scores`),
but PR's eviction-protected array is `regions[1]` (=`contrib`) — so contrib lines never got a
delivered-epoch stamp and the feature was silently inert for its target array. Fixed to search ALL
property regions (mirroring `isPropertyData`/`findNextRef`); `edge_epoch_count` clamped to
`[2,65535]` to match the kernel.

**Updated cross-sim eviction faithfulness:** all 3 sims now deliver the per-edge epoch through the
hierarchy (cache_sim packed record, gem5 packed-flat/ecg.extract, Sniper SNIPER_ECG_EXTRACT).
Sniper's `findNextRef` matrix remains the default/oracle path.

## 13. IMPLEMENTED + VALIDATED: Sniper Path A (epoch-filtered next-K lookahead, 2026-06-21)

Sniper now runs the **headline Path A** combined-stack prefetcher (epoch-filtered next-K
edge-list lookahead), matching cache_sim and gem5 (§11). This closes the last 3-sim
prefetch gap: Path A was previously cache_sim + gem5 only; Sniper ran Path B (single
packed target) exclusively.

**Mechanism (kernel `bench/src_sniper/pr.cc`, mode-6 loop):** gated by
`ECG_EDGE_MASK_PREFETCH=K` (>0). For each demand edge, walk the next-K `in_neigh`,
compute `cur_ep = ecg_epoch::currentEpoch(u, N, ne)` and the candidate epoch
`cand_ep = in_edge_epochs_by_src[u][cpos]`, keep survivors via the SSOT
`ecg_epoch::prefetchKeep`, and for each survivor emit BOTH `SNIPER_ECG_EXTRACT(cand,
cand_ep)` (records the candidate's epoch in the per-core map so the prefetched line is
stamped at fill — the only channel to a prefetch-filled line's epoch in Sniper) AND
`SNIPER_ECG_PFX_TARGET(cand)` (issues the prefetch). Mutually exclusive with the Path B
single-target block (K==0). `cand` is the streamed edge id at full width (no size limit,
unlike gem5's 15/24-bit packed-target field).

**Prefetcher batch-drain (`ecg_pfx_prefetcher.cc::getNextAddress`):** Path A pushes up
to K hints per demand edge, so the prefetcher now drains up to
`SNIPER_ECG_PFX_DRAIN_BATCH` (default 8) hints per call instead of exactly one —
otherwise the 256-entry ring overflows and drops most Path A targets. The batch bound is
on *consumed*-per-call (not issued), so a run of duplicates/invalids cannot spin. Path B
(single target) is unaffected.

**Dedup bypass for parity:** Path A emits every survivor (like cache_sim/gem5), so the
emit-side dedup `should_emit_ecg_pfx_hint` must be disabled (`SNIPER_ECG_PFX_HINT_FILTER=0`,
wired by `roi_matrix.py --ecg-pfx-hint-filter 0`). The kernel warns if `K>0` while the
filter is non-zero, or if `K>0` without the mode-6 prefetch loop (so Path A would not fire).

**Validation (Sniper, full `pr` binary, email-Eu-core, ECG_GRASP_POPT, L3=256kB,
`--prefetcher ECG_PFX --ecg-pfx-mode per_edge --ecg-pfx-hint-filter 0`,
`SNIPER_ENABLE_ECG_EXTRACT=1`):**

| config | `ecg_pfx_target_hints_seen` | `pf_issued` | `invalid_target` |
| ------ | --------------------------: | ----------: | ---------------: |
| Path B (K=0, single target) | 13,254 | 11 | 0 |
| Path A (K=8, next-K lookahead) | **226,039** | 0 | 0 |

Path A consumes **~17× more target hints** than Path B (next-K vs single), confirming the
lookahead fires and is **distinct** from Path B. `invalid_target=0` ⇒ the batch-drain kept
up with the high volume (no ring starvation). The `nuca replacement_policy=ecg` +
`l2_cache/prefetcher=ecg_pfx` are correctly wired.

**Honest scope / known limitation:** `pf_issued=0` for Path A (status `active_no_fill`) is
the **pre-existing Sniper L2 prefetch-enqueue filter** (documented since sprint 6b-2), NOT
a Path A regression — Path B's batch-drain path still issues fills (11). This validation
cell also has **no eviction pressure** (email-Eu-core property = 4 KB ≪ 256 KB L3,
`l3_mr≈0.0002`), so fills would not help regardless. Sniper therefore validates the Path A
**delivery+consumption mechanism** (next-K epoch-filtered hint generation and drain);
**gem5** is the cycle-accurate Path A reference *with* real fills (pf 794333 issued, 82.6%
useful, on kron_s16_k4 >32K verts, §11), and **cache_sim** is authoritative for total
traffic/miss-rate. This matches the §10 cross-sim contract (cache_sim = traffic;
gem5/Sniper = cycle-accurate mechanism with documented per-tool limitations).

## 14. HW-scalability + honest bit budget: epoch-only record, storage-free prefetch (2026-06-22)

Driven by the reviewer-critical question *"at deployment scale (twitter 41M, friendster
65M, kron-s27 134M — the P-OPT/GRASP/DROPLET corpus) can ECG even fit its bits in HW?"*

### 14.1 Eviction: scale-independent, same HW class as P-OPT
The eviction metadata is **per-LLC-line**, not per-edge: `m_ecg_epoch` is ~10 bits/way in
the tag array (like RRIP's 2 b or P-OPT's per-line bits). A 2 MB LLC = 32 K lines × 10 b ≈
**40 KB, independent of graph size**. The decision is a W-way epoch-distance compare
(`(epoch+ne−cur_ep) mod ne`, max over 16 ways) — identical complexity to P-OPT's
rereference compare. The bounded per-core delivery map (`recordEcgEpoch`,
`kEcgEpochMapSize`) is fixed-size, **not O(V)**. So eviction is realizable at any scale, and
ECG keeps all 16 LLC ways whereas P-OPT reserves 1 — ECG's runtime LLC cost is *lower*.

### 14.2 The bit-fit worry is the packed *delivery*, not the mechanism
"Can't fit the bits at 134 M" applies **only** to the 4-byte zero-traffic packing, where the
~10-bit epoch shares the 32-bit edge word with the dest id. The epoch width is
**scale-invariant** (see §14.4); the id is what grows. So decouple them — the `ecgRecordBytes`
auto-switch already does: 4 B (≤~1 M with 10 b epoch, zero traffic) → 8 B (id ∥ epoch as
separate fields, any scale to ~4 B verts) → 16 B beyond. At >4 B verts 64-bit ids are needed
anyway, so the epoch rides free in the high half. The 4-byte zero-traffic case is a
small-graph **bonus**, not the foundation.

### 14.3 Prefetch stores NOTHING — Path B field removed from the honest layout
The stored prefetch-**target** field (Path B, a full vertex id — `kPrefetchBits=31` in the
old mask) is the *only* metadata that grows with N AND can't be derived. It is **dead
weight**: the headline Path A reads the next-K targets straight from the CSR edge stream
(= DROPLET's read-ahead engine, Basak HPCA'19), so nothing is stored; the prefetched line's
epoch comes from the **same edge entry** the prefetcher already read. Evidence: the cache_sim
record already defaults `ECG_RECORD_PREFETCH_BITS=0` / `ECG_RECORD_POPT_BITS=0`, and runs
report `pfx_encoded=0 pfx_no_candidate=ALL` yet still beat DROPLET. New honest SSOT layout
`ecg_mode6::packMaskEpochOnly` (`ecg_mode6_builder.h`): `dest[0:28] | dbg[28:30] |
epoch[30:64]` — **no prefetch field**, dest covers 268 M verts (twitter/friendster/kron-s27),
pinned by `test_ecg_packed_field_parity` (now **58/58**, +10 epoch-only checks). The gem5
`ecg.extract` ISA decoder stays in lockstep (a `funct7=0x1` epoch-only variant; runtime
rip-out of the gem5/Sniper Path B mask field tracked separately — cache_sim is already honest).

### 14.4 Epoch precision SATURATES at ~10 bits — do NOT widen (measured)
Sweeping `ECG_EDGE_MASK_EPOCHS` on the `-o5` shortcircuit headline (eviction-only):

| graph (L3) | ne=64 (6b) | ne=256 (8b) | **ne=1024 (10b)** | ne=4096 (12b) | ne≥16384 | POPT |
|---|---|---|---|---|---|---|
| web-Google (512 kB) | 0.6923 | 0.6328 | **0.6050** | 0.6190 | 0.6190 | 0.6284 |
| cit-Patents (1 MB) | — | 0.7327 | **0.6773** | 0.6773 | 0.6773 | 0.7449 |

~10 bits (ne=1024) is the sweet spot and **beats P-OPT** on both. More bits give **no
eviction gain**: cit-Patents is flat (all 0.6773); web-Google's 0.605→0.619 step is purely the
**charged record width** flipping 4 B→8 B (`id 20 + epoch 10 + tier 2 = 32` packs in 4 B at
ne=1024, but `+epoch 12 = 34` needs 8 B at ne=4096 → more mask-stream traffic counted in
l3_mr), **not** worse eviction. So the answer to *"give more bits for a more precise epoch?"*
is **no** — pin the epoch at ~10 bits (saturates quality, keeps the record at 4 B where the id
allows); spend the reclaimed Path-B bits on staying compact, not on a wider epoch.

### 14.5 Honest minimal record + HW-cost vs the baselines
Canonical ECG per-edge metadata = **`dest + 2-bit degree tier + ~10-bit epoch`** — no popt
rank, no prefetch target. Offline precompute is O(E) parallel (= P-OPT's offline matrix build).
The genuinely software-visible cost is the **ISA load-hint** that carries the epoch to the LLC
(`ecg.extract`) — ECG's explicit design point ("software-visible ISA hints"), vs P-OPT/DROPLET
being transparent. Net: ECG's offline metadata ≈/< P-OPT's O(V·epochs) matrix, ECG reserves
**zero** LLC ways (vs P-OPT's 1), prefetch stores **nothing** (vs Path B's vestigial id field).

## 15. Simulator tiering + 3-sim equivalence showcase (2026-06-22)

### 15.1 The three simulators and their paper roles (decided)
Mirrors how the baselines were evaluated (P-OPT shipped a functional cache sim
`CMUAbstract/POPT-CacheSim-HPCA21` + Sniper for timing; GRASP/DROPLET used Sniper):

| sim | role | graph scale | why |
|---|---|---|---|
| **cache_sim** | fast **prototyping** + functional authority (miss-rate/traffic) | any (memory-bound) | deterministic cache metrics, = P-OPT's POPT-CacheSim role; iterate ideas in minutes |
| **gem5** | the **ISA case study** (`ecg.extract` HW-faithful epoch delivery, cycle-accurate) | small/medium | demonstrates the mechanism is real silicon, not a host-side shortcut |
| **Sniper** | **scale** demonstration | large (40–134M, like GRASP/DROPLET) | interval sim runs large graphs in hours–days (gem5 cannot) |

Caveat: GraphBrew's Sniper currently runs full-detail under SDE (~20 min on 1005-node
email-Eu-core), so reaching 60M needs sampling / native-pinball reconfig (tracked). cache_sim
is the practical large-graph path today and is methodologically faithful (P-OPT did the same).

### 15.2 Equivalence is proven by shared-DECISION parity, not identical kernels
The three kernels are separate (`bench/src_{sim,gem5,sniper}`) and deliver the epoch by
different mechanisms (host mask arrays vs `ecg.extract` ISA vs `SNIPER_ECG_EXTRACT`), but the
eviction/prefetch DECISION is SSOT. Verified end-to-end this session:

| check | cache_sim | gem5 | Sniper |
|---|---|---|---|
| eviction victims obey spec (`verify_ecg.py`) | **7×40/40** + 2070/2070 coverage | **5×40/40** + 4000/4000 | **4×40/40** |
| epoch-property eviction branch fired (real run) | yes | yes | yes |
| prefetch-target decision (`verify_pfx.py`) | 10/10 (covers all 3) | ✓ | ✓ |
| field-delivery layout (`test_ecg_packed_field_parity`) | **58/58** (pins all layouts incl epoch-only) | (shared header) | (shared header) |
| Path A epoch-filtered lookahead fires | yes (§11) | yes (§11) | yes (§13) |

So every ECG variant (grasp_only / epoch_only / rrip_first / epoch_first / shortcircuit)
selects the SAME victim under the SAME access stream in all three simulators — equivalence
holds. (gem5 numeric common-cell on kron_s16_k4 in progress as the intuition showcase; the
cache_sim common-cell already shows ECG:shortcirc 0.8136 < POPT 0.8932 < GRASP 0.8391 there.)

## 16. ECG eviction CONTRACT: tiebreaker + non-property + stamped (SSOT, 3-sim verified, 2026-06-22)

The eviction DECISION is the single shared header `bench/include/ecg_victim_policy.h`
(`ecg_policy::selectVictim`), **byte-identical** across cache_sim / gem5 / Sniper (md5
verified; synced on every change). Each sim's thin adapter populates
`WayState{prop, rrpv, recency, dbg, dist, stamped}` from its native lines; the decision is
NOT re-implemented per sim. `verify_ecg.py` (+`--gem5`/`--sniper`) asserts every victim obeys
this contract in all three.

### 16.1 What the fields mean
- **prop** — `true` for a PROPERTY line (vertex/score/contrib — the hot graph data we protect),
  `false` for a RECORD line (edge-stream/metadata — streamed ~once, low reuse).
- **dist** — raw circular next-reference distance `(stored_epoch + ne − cur_epoch) % ne`
  (larger = referenced farther in the future = better eviction candidate).
- **stamped** — a per-edge epoch was DELIVERED to this line. As of the valid-bit fix (§ commit
  a01e8e2e/e3f7933e) this is an **explicit per-line valid bit**, NOT `epoch != 0`: a real
  epoch-bucket-0 line (low-ID next-referencer) that WAS delivered is correctly `stamped`.
- **effDist** — `stamped ? dist : 0`. An UNSTAMPED property line contributes distance 0
  (treated as "kept", never "farthest"), so only genuinely stamped property competes on epoch.

### 16.2 Non-property handling (CONTRACT — identical in all variants)
**Property lines are protected; record (non-property) lines are evicted first.** This is the
GRASP-derived invariant. Precisely, per variant:
- **grasp_only** — pure RRIP; `prop` ignored (== GRASP sanity baseline).
- **shortcircuit** (the headline) — evict the **first non-property line by set order**; only if
  the whole set is property do we rank by epoch.
- **epoch_first / epoch_only** — evict the **oldest non-property line by recency**; else stamped
  property by farthest dist; else LRU fallback.
- **rrip_first** (default) — within the max-RRPV set: oldest non-property by recency; else
  property by farthest effDist; age + retry if no candidate.

### 16.3 Tiebreaker hierarchy (shortcircuit — the canonical headline)
When the set is **all property** (no record to evict first), pick the victim by, in order:
1. **effDist** — largest effective next-ref distance (farthest-future). Unstamped property = 0.
2. **DBG degree tier** — on an effDist tie, evict the higher `dbg` tier. (Empirically vestigial:
   the epoch subsumes degree; kept as a cheap, deterministic tiebreak — see §14.4.)
3. **set order** — on a full tie, the first such way index (deterministic).

### 16.4 Cross-sim equivalence scope (IMPORTANT)
Equivalence holds on the **ISA-delivered epoch path** (cache_sim host arrays = gem5 `ecg.extract`
= Sniper `SNIPER_ECG_EXTRACT`): all three feed the same `stamped`/`dist` into the same
`selectVictim`. gem5 and Sniper ALSO have a `findNextRef` host-matrix FALLBACK that quantizes
distance differently (`min(dist,127)>>3`); that path is a diagnostic, NOT part of the
equivalence contract. Report cross-sim numbers only in delivered-epoch mode
(`SNIPER_ENABLE_ECG_EXTRACT=1` / gem5 `ecg.extract`).

### 16.5 Verification (all PASS, 2026-06-22)
- Synthetic exact-victim unit test `test_ecg_victim.cc` (sets the valid bit) — PASS.
- Live-trace spec `verify_ecg.py`: cache_sim 7×40/40 + 2070/2070 coverage; gem5 5×40/40 +
  4000/4000; Sniper 4×40/40. The trace now carries an explicit `stamped=` column; the verifier
  models `stamped` (not `epoch != 0`).
- Field-delivery parity `test_ecg_packed_field_parity` — 58/58.

## 17. CANONICAL config (pin ONE — stop hand-rolling env, 2026-06-22)

There are ~50 ECG_* env knobs from the prototyping phase; several are LOAD-BEARING and
silently change l3_mr by 0.1+ if omitted. **Headline numbers MUST be produced via the committed
scripts** (`ecg_variant_matrix.py` for eviction, `combined_stack_matrix.py` for prefetch/combined,
both wrapping `roi_matrix.py`), NEVER by hand-rolled `env`. Hand-rolling reproduced 0.60 / 0.74 /
0.76 for the "same" web-Google/o5 cell by omitting `CACHE_ULTRAFAST=0`, `CACHE_L1_POLICY=LRU`,
`CACHE_L2_POLICY=LRU`, or `CHARGED=1`. The committed scripts set the full correct env.

### 17.1 The one canonical config
| axis | value |
|---|---|
| eviction policy | `CACHE_POLICY=ECG`, `ECG_MODE=ECG_GRASP_POPT` |
| variant | `ECG_VARIANT=shortcircuit` (the headline; epoch_only/epoch_first tie it post-§16) |
| epoch | `ECG_EDGE_MASK_EPOCHS` (10 bits ne=1024 saturates; operationally ≤ 65535 to fit 16-bit storage) |
| record | epoch-only: `dest + 2-bit DBG tier + epoch` — NO popt field, NO prefetch-target field |
| prefetch | Path A only (`ECG_EDGE_MASK_PREFETCH=K`, storage-free CSR read-ahead); Path B dropped |
| cache | L1 32 kB/8w (LRU), L2 256 kB/8w (LRU), L3 16-way/64 B (size swept), `CACHE_ULTRAFAST=0` |
| reorder | report BOTH `-o5` (DBG; GRASP/ECG-tier need it) and `-o0` (un-reordered robustness) |
| threads | `OMP_NUM_THREADS=1` (cache_sim determinism) |

### 17.2 The 64-bit ISA record (SSOT delivery format)
```
[ dest : 28 ][ dbg : 2 ][ epoch : 34 ]    (one 64-bit ecg.extract word)
```
- **dest 28 bits** → 268 M vertices (covers twitter 41 M / friendster 65 M / kron-s27 134 M).
- **dbg 2 bits** → 4 degree tiers (empirically vestigial — the epoch subsumes it; kept as a free
  tiebreak, §14.4). Do NOT spend more bits here.
- **epoch 34 bits** → generous safety headroom (saturates ~10; free in the 64-bit word).
- **No prefetch-target field** — Path A reads the next-K targets from the CSR edge stream
  (= DROPLET read-ahead), storing nothing (§14.3). Path B's stored target was the only field
  that grew with N and is dropped.
- **All 3 sims MODEL this format**: gem5 delivers it via the real `ecg.extract` op; cache_sim
  (host arrays) and Sniper (`SNIPER_ECG_EXTRACT`) model the identical layout so equivalence is on
  the proposed ISA format, not each sim's native shortcut.

### 17.3 Load-bearing knobs the canonical scripts set (do not omit)
`CACHE_ULTRAFAST=0`, `CACHE_L1_POLICY=LRU`, `CACHE_L2_POLICY=LRU`, `ECG_EXACT_REREF=1`,
`ECG_PREFETCH_MODE=6`, `ECG_EDGE_MASK_{EPOCH,LINEMIN,LEAN,PACK,CHARGED}=1`. `CHARGED=1` is the
sharpest: it enables epoch DELIVERY; without it eviction degenerates toward GRASP
(web-Google/o5 0.76 vs 0.62).

## 18. HONEST prefetch/combined framing — eviction is the bandwidth headline (2026-06-22)

Rubber-duck (rd-plan2) on the fresh combined-stack matrix. The clean both-order data (headline
graphs, `combined_stack_matrix.py`) **refutes** any "ECG+PathA combined stack beats everything"
claim. Report this honestly.

### 18.1 The data (bandwidth = total DRAM traffic; demand2mem = latency proxy; both lower=better)
| cell (-o5) | LRU bw / dem | LRU+DROPLET bw / dem | **ECG-evict bw / dem** | ECG+PathA bw / dem |
|---|---|---|---|---|
| web-Google 512kB | 17.76 / 17.76 | 18.03 / 3.00 | **14.39 / 14.39** | 16.78 / 4.63 |
| cit-Patents 1MB | 91.93 / 91.93 | 92.55 / 17.24 | **74.65 / 74.65** | 83.36 / 21.49 |
| soc-pokec 1MB | 74.81 / 74.81 | 78.48 / 10.15 | **58.74 / 58.74** | 72.45 / 19.05 |

### 18.2 What the data actually says (three separate, honest claims)
1. **EVICTION is the bandwidth + miss-rate headline.** ECG eviction-only has the LOWEST bandwidth
   in EVERY cell (web-Google −19% vs LRU, soc-pokec −21%) AND the lowest miss-rate (§16). This is
   the strong, clean, single-metric win.
2. **Prefetch (Path A) is a LATENCY trade, not a bandwidth win.** Adding Path A to ECG *increases*
   bandwidth (web-Google 14.39→16.78, soc-pokec 58.74→72.45) while slashing demand2mem
   (14.39→4.63). A prefetcher RELOCATES demand→prefetch; it does not cut total traffic. (DROPLET is
   the same: LRU 17.76 → LRU+DROPLET 18.03, bandwidth flat/up.)
3. **No single arm dominates (Pareto).** Per cell the non-dominated frontier is:
   **ECG-evict (bandwidth-optimal) — ECG+PathA (balanced) — LRU+DROPLET (latency-optimal)**; plain
   LRU is dominated by ECG-evict on both axes. ECG+PathA beats LRU+DROPLET on bandwidth but
   LRU+DROPLET beats ECG+PathA on demand-latency (web-Google 3.00 vs 4.63).

### 18.3 Defensible paper framing (do NOT overclaim)
- **Headline:** "ECG eviction reduces LLC traffic and miss-rate vs LRU/GRASP/POPT."
- **Prefetch:** "ECG's epoch-stamped Path A prefetch trades a little bandwidth for a large
  demand-miss (latency) reduction, and at equal latency uses less bandwidth than LRU+DROPLET."
- **M3 = interaction / Pareto matrix**, NOT a "combined-stack win." Annotate each row with the
  bandwidth winner, the demand2mem winner, and whether ECG+PathA is non-dominated.
- Eviction is cache_sim/gem5/Sniper-equivalent on traffic; the prefetch LATENCY claim needs a gem5
  cycle-accurate spot-check (cache_sim models traffic, not MSHR/timeliness), and Sniper is parity
  corroboration only (its L2 enqueue filter suppresses fills). See todos cu-gem5-pfx-spotcheck.

## 19. Consolidated ECG ISA: ONE instruction, mode-controlled caching (2026-06-26)

The prototyping forms (`ecg.extract` reg-hint FUNCT3=0x0, `ecg.load` side-record FUNCT3=0x1)
are SUBSUMED into a SINGLE headline instruction. The paper presents one custom-0 op; a FUNCT7
MODE field selects which caching axis the per-edge metadata drives.

### 19.1 The instruction
```
ecg.load  rd, rs1, rs2        custom-0 (opcode 0x0b), FUNCT3=0x2, R-type
    rs1 = property base (&prop[0]);  rs2 = fat edge (mode-6 record);  rd = prop[dest]
    dest = rs2 & ((1<<W)-1);  EA = rs1 + dest*elem_size;  deliver metadata per MODE BEFORE
    the fill (so the line is stamped on insertion);  rd = Mem[EA].
```
The instruction word is a fixed 32 bits; the 64-bit metadata lives in the register `rs2`, never
in the instruction. `W` is the configurable dest-field width (§19.5). Decoder SSOT:
`overlays/arch/riscv/isa/decoder_ecg_extract.isa` + `formats/ecg.isa` (the `ECG_MODE`/`ECG_WIDTH`
bitfields), mirrored byte-identical to the gem5/src build tree by `setup_gem5.py`.

### 19.2 FUNCT7 = ECG_MODE<31:27> | ECG_WIDTH<26:25>  (mode picks the axis, width sizes dest)
| ECG_MODE | mode | SSOT layout (bits of rs2) | caching effect |
|---|---|---|---|
| 0x00 | **EVICT** (headline) | `dest[0:W] | epoch[W:W+16]` (`packEvict`) | next-ref epoch eviction (P-OPT-class) |
| 0x01 | EVICT+PFX | `dest[0:W] | epoch[W:W+16] | pfx[W+16:64]` (`packEvictPfx`) | + Path-B prefetch target |
| 0x02 | EMBEDDED | NARROW `packMaskEpoch`: `dest[0:24] | dbg[24:26] | popt[26:33] | epoch[33:49] | pfx[49:64]` | full legacy metadata (fixed 24-bit dest) |

`FUNCT7 = (ECG_MODE<<2) | ECG_WIDTH`. The shifts come from the builder constants, so the gem5
decoder, the cache_sim/Sniper models, and the kernel packer read the SAME bits.

### 19.3 Bug fixed this session (the consolidation)
- An earlier hand-edit that added `ecg.load`/`ecg.pload` as FUNCT3 cases **dropped the `}` that
  closes the custom-0 `0x02: decode FUNCT3 {` block**, so the standard LOAD opcode (and everything
  after) was wrongly nested inside custom-0. The gem5 ISA parser reported only `At 0: unknown
  syntax error` (EOF). Restored the missing brace; gem5/src is now byte-identical to the overlay.
- The dbg-delivery modes had a **layout collision**: they put `dbg` at bit 40, which the WIDE layout
  uses for `pfx`. Reconciled: dbg is carried ONLY by the NARROW `packMaskEpoch` (EMBEDDED, 0x01);
  the WIDE modes (EVICT/EVICT+PFX) carry no dbg (it is vestigial and reclaimed for the wider pfx,
  §10.2/§14). The non-SSOT FULL (0x03) mode was dropped (EMBEDDED already carries every field).

### 19.4 Validation (no bugs)
- ISA parses clean standalone (`isa_parser` → `PARSE_OK`); gem5 RISCV `gem5.opt` rebuilds rc=0
  (`[ISA DESC]` regenerates + compiles + links). The generated decoder resolves `ECG_WIDTH` to
  `bits(machInst,26,25)` (verified in the generated `.inc`).
- **Real-decoder gate** (`test_ecg_load_modes.cc`, wired into `verify_ecg.py --gem5`): issues EVERY
  `(mode × dest-width)` `ecg.load` through the ACTUAL gem5 RISC-V decoder and checks the decoded dest
  via `rd = prop[dest]`. The field-parity test only checks a C++ MIRROR of the shifts and the 3-sim
  verify drives eviction via the X86 m5op path, so this is the only thing that runs the real decoded
  instruction for the new modes/widths. Includes a TEETH proof: forcing the emitted width wrong
  (`ECG_TEST_FORCE_WC`) while packing correctly makes the decoder mis-extract dest → the test FAILs,
  proving `ECG_WIDTH` is load-bearing (non-vacuous). Both pass: normal=PASS, forced-wrong=FAIL.
- Field-layout parity + drift guard (`test_ecg_packed_field_parity.cc`) = 95/0, pinning the WIDE
  shifts AND the configurable-width `W = 8*(wc+1)` decoder logic against the builder SSOT for every
  width class; EMBEDDED's NARROW shifts verified equal to `packMaskEpoch`.
- `experiments.py verify` (cache_sim) = ALL POLICIES VERIFIED; `--gem5 --sniper` = ALL POLICIES
  VERIFIED — 3-sim eviction equivalence (shared `ecg_victim_policy.h` decision + shared
  `ecg_mode6_builder.h` layout), cache_sim + gem5 + sniper all agree GRASP helps. Epoch width is
  analytical (§14.4): `b ≈ log2(N·ρ/C)` ⇒ ~10-bit saturation on the eval corpus and ≤16 bits for
  ~100M nodes @ 1MB LLC.

### 19.5 Configurable dest width (8/16/24/32) — one op scales to 4.29B vertices
The `ECG_WIDTH` field (FUNCT7 bits[26:25]) selects the dest-field width class
`wc ∈ {0,1,2,3} → W = 8/16/24/32` bits; the next-ref EPOCH always rides `[W:W+16]`, and EVICT+PFX
puts the prefetch target above `[W+16:64]`. So a single instruction sizes its dest to the graph:
W8 ≤ 256 verts, W16 ≤ 65 K, W24 ≤ 16.7 M (the headline default == the prior 24-bit WIDE layout),
W32 ≤ 4.29 B — covering twitter/friendster/kron-s27 and beyond without a wider register or a
second op. SSOT helpers in `ecg_mode6_builder.h`: `ecgEvictWidthClass(N)`, `ecgEvictWidthBits(wc)`,
`packEvict`/`packEvictPfx`, `extractEvict{Dest,Epoch,PfxTarget}`. `.insn r` needs a constant funct7,
so the kernel emitter (`gem5_ecg_load_evict`) is a 4-way switch over the constant FUNCT7 per width;
the PR kernel computes `wc = ecgEvictWidthClass(g.num_nodes())` once. This is the scaling axis the
analytical bit budget motivates: with a fixed cache the dest grows with the graph, not the epoch.

### 19.6 OoO request-sideband — the race-free, HW-realizable delivery (final piece)
The epoch must reach the LLC fill associated with the RIGHT demand. Two in-order models existed:
the single-slot mailbox (`setDecodedEcgExtractHint`) and the per-vertex table
(`storeEcgMetadataByVertex`). The mailbox **races under an out-of-order CPU** (a later `ecg.load`'s
epoch can overwrite an earlier one before its fill stamps the line); the table is **O(num_vertices)**
— the very cost ECG avoids. The correct OoO + HW-realizable delivery is a per-REQUEST sideband: the
`ecg.load` AGU tags the demand `Request` with `{dest, epoch}` (a few tag bits riding the in-flight
load), and the LLC reads it on the fill. The epoch travels WITH the specific request → no shared
structure to race, no per-vertex storage.

Implemented as a first-class gem5 `Request::Extension` (`Request` is `Extensible<Request>`):
`overlays/mem/cache/replacement_policies/ecg_epoch_request_ext.hh` defines `EcgEpochExtension` +
`attachEcgEpoch(req,…)` (the O3 AGU side) + `readEcgEpoch(req,…)` (the LLC side). `ecg_rp.cc` reset
now consults `readEcgEpoch(pkt->req,…)` FIRST and falls back to the in-order mailbox/table. Compiles
+ links into RISCV `gem5.opt` (rc=0); a standalone unit check confirms the attach/read round-trip.

In-order equivalence (why the case study is valid): on the serialized TimingSimpleCPU the mailbox
holds exactly the demanded vertex's epoch when its fill reaches the LLC, so it is mathematically
equivalent to the sideband (no race possible) — the replacement policy is validated in-order via the
mailbox, and the `EcgEpochExtension` is the SAME information delivered race-free for the O3CPU /
multicore form. The remaining O3 integration is the AGU attach (a custom `ecg.load` format's
`initiateAcc` calling `attachEcgEpoch` on the request it issues); the read side + extension are in
place so that path is correct the moment it is wired.

## 20. Three-simulator equivalence showcase + debug proof (2026-06-26)

`scripts/experiments/ecg/three_sim_showcase.py` runs the same policies on the same cell across
cache_sim / gem5 / Sniper and prints (a) an L3 miss-rate table and (b) the per-sim `[ECG-CONFIG …]`
banner. It drives the committed `roi_matrix` with the verified cell geometry, so the ECG headline is
NOT hand-rolled — a raw `roi_matrix --policies ECG:…` that omits the load-bearing knobs
(`ECG_EDGE_MASK_CHARGED`, `CACHE_ULTRAFAST=0`, the L1/L2 sizes) makes ECG DEGENERATE (kron@128kB
ECG 0.67 > LRU 0.66). With the correct geometry ECG matches verify exactly (0.5718).

**Two distinct claims — do not conflate them:**
- **(A) ECG ADVANTAGE** — ECG beats GRASP *and* P-OPT. This is a REAL-GRAPH claim on cache_sim (the
  functional authority), §20.0.
- **(B) 3-SIM EQUIVALENCE** — the same policy moves the miss rate the same DIRECTION in all three
  simulators. This needs a gem5/Sniper-feasible cell, so it uses the SYNTHETIC kron_s16_k4 (§20.1).
  kron's Kronecker structure does NOT reward the next-reference epoch, so ECG does NOT beat GRASP on
  kron — that cell certifies cross-sim agreement, NOT the ECG advantage.

### 20.0 ECG ADVANTAGE (real graphs, cache_sim, -o5, ECG_VARIANT=shortcircuit): ECG beats GRASP AND P-OPT
ECG_GRASP_POPT layers GRASP's degree-aware INSERTION + P-OPT's next-reference EVICTION (the epoch is
strictly more information than degree), so the design target is `LRU > GRASP > P-OPT ≥ ECG` (lower
miss rate is better). On graphs with exploitable next-reference structure it holds:

| cell (cache_sim, -o5) | LRU | GRASP | P-OPT | **ECG** |
|---|---|---|---|---|
| web-Google @ 512kB (L1=32k,L2=256k) | 0.8440 | 0.6733 | 0.6326 | **0.6229** |
| cit-Patents @ 1MB (L1=32k,L2=256k)  | 0.8958 | 0.8196 | 0.7471 | **0.6795** |

ECG is the LOWEST miss rate in both — it beats GRASP and the (all-ways, uncharged) P-OPT, using a
memory-resident per-edge mask and NO reserved LLC way (§14, §21 on P-OPT's 1-way charge makes the
margin larger still). This is the headline; the kron cell below is ONLY for cross-sim equivalence.

### 20.1 Equivalence cell — NOT the ECG advantage (kron_s16_k4 @ L3=128kB/16w, L1d=16kB, L2=64kB, -o5, PR, ECG_VARIANT=rrip_first)
| policy | cache_sim | gem5 | Sniper |
|---|---|---|---|
| LRU | 0.6606 | 0.6475 | 0.5695 |
| GRASP | **0.5319** | **0.5655** | **0.4771** |
| ECG_GRASP_POPT | **0.5718** | (eviction-spec) | (eviction-spec) |

Read as DIRECTION vs LRU, not absolute (absolute rates are NOT comparable across simulators: gem5/
Sniper see the full ISA access stream, cache_sim sees graph accesses only — §10/§16.4). GRASP HELPS
in ALL THREE simulators (−0.129 / −0.082 / −0.092); cache_sim ECG helps (−0.089). This is the same
result the verify GATE 1+2 asserts (`helps=['cache_sim','gem5','sniper']`).

### 20.2 gem5/Sniper ECG on a PRESSURED cell = eviction-spec, not full-run
gem5/Sniper ECG_GRASP_POPT on kron@128kB exceeds the per-run sim timeout (≈15 min of detailed sim;
the ECG mask/epoch path is heavier than LRU/GRASP). Their ECG correctness is therefore established by
the eviction-SPEC checks (`verify_ecg.py --gem5 --sniper`: 40/40 each), the byte-identical shared
`ecg_victim_policy.h` decision, the field-parity drift guard, and the real-decoder gate — NOT by a
full miss-rate run on the pressured cell. cache_sim (the functional authority) carries the ECG
miss-rate headline.

### 20.3 Small-cache confound (why the cell matters)
On a TINY L3 (email-Eu-core @ 8kB) cache_sim shows GRASP/ECG helping (0.657→0.534/0.536) but gem5
shows them HURTING — the gem5 L3 is swamped by the full-ISA stream, so the graph-property retention
signal is drowned out. This is the documented access-population confound (§10), not an equivalence
failure: equivalence must be read on a cell large enough that the property region is the L3 working
set (kron@128kB), where all three simulators agree.

### 20.4 Debug proof (the runs are what they claim)
- `ECG_DEBUG=1` → each sim emits ONE `[ECG-CONFIG sim=… policy=… mode=… variant=… …]` line at
  policy init, proving the resolved config. Verified identical across all three on kron:
  `sim=cache_sim policy=ECG mode=ECG_GRASP_POPT variant=rrip_first charged=1`,
  `sim=gem5 … llc=131072B`, `sim=sniper … policy=ECG mode=ECG_GRASP_POPT variant=rrip_first`.
  Banners live in `cache_sim` `MaskConfig::initFromEnv`, gem5 `GraphEcgRP` ctor, Sniper `CacheSetECG`
  ctor (one-shot).
- `ECG_EVICT_TRACE=N` → each sim dumps the first N L3 evictions (`[EVICT L3 pol=… curEpoch=… ->
  victim=way… reason=…]`), proving the policy ACTS (already present in cache_sim + gem5 + Sniper).

## 21. FULL multi-kernel equivalence + full debug (2026-06-26)

The ECG eviction DECISION (`ecg_victim_policy.h`) is kernel-AGNOSTIC and byte-identical across the
three simulators, so the policy must obey the same eviction spec for EVERY kernel in EVERY
simulator — not only PageRank. `scripts/experiments/ecg/verify/equiv_kernels.py`
(`experiments.py verify --kernels [--gem5 --sniper]`) runs each kernel under ECG_GRASP_POPT with the
eviction trace on, asserts every L3 eviction obeys the policy spec (reusing `verify_ecg.py`'s
kernel-agnostic `verify_trace`), AND captures the per-sim `[ECG-CONFIG …]` debug banner.

### 21.1 Result (email-Eu-core, coverage geometry, ECG_VARIANT=rrip_first) — ALL PASS
| kernel | cache_sim | gem5 (X86) | Sniper |
|---|---|---|---|
| PR  | ok (2070/2070) | ok (4000/4000) | ok (40/40) |
| BFS | ok (489/489)   | ok (2637/2637) | ok (40/40) |
| BC  | ok (503/503)   | ok (629/629)   | n/a |

Every cell: eviction-spec PASS **and** the `[ECG-CONFIG sim=… policy=ECG mode=ECG_GRASP_POPT
variant=rrip_first …]` banner present. So the eviction-decision equivalence holds across kernels
*and* simulators, not just PR.

Honest gaps (documented, not silently skipped):
- **SSSP** needs a WEIGHTED graph (`.wsg`). On the unweighted `email-Eu-core.sg` it sees ~63 accesses
  and never pressures the L3 (≈0 evictions); the eval corpus ships only unweighted `.sg` and the
  `converter` has no weight-generation flag. SSSP is therefore omitted from the matrix.
- **BC on Sniper**: the Sniper `sg_kernel` driver has no `bc` target (cache_sim + gem5 only).

### 21.2 Scope: this is the DECISION equivalence; the mask DIRECTION is still PR-tuned
`verify_trace` certifies that each eviction obeys the spec *given the masks the kernel delivered* —
which is kernel-agnostic and direction-independent. It does NOT certify that the per-edge mask
direction is OPTIMAL for BFS/BC: the next-ref matrix defaults to `out_neigh` (PR's in-pull
transpose), while BFS-top-down/BC traverse out-edges (§ "graph-direction correctness"; uncertified
on directed graphs). That is a miss-rate concern, not a spec one — PR remains the direction-correct
performance headline (§20.0).

### 21.3 Full debug
Two independent proofs fire for every kernel × sim:
- `ECG_DEBUG=1` → one-shot `[ECG-CONFIG sim=… policy=… mode=… variant=…]` at policy init. cache_sim's
  banner now lives in `traceEvict` (the first L3 eviction — universal across PR/BFS/BC, unlike the
  former PR-only `MaskConfig::initFromEnv` placement); gem5 `GraphEcgRP` ctor; Sniper `CacheSetECG` ctor.
- `ECG_EVICT_TRACE=N` → the first N L3 evictions with every candidate's fields + chosen victim +
  reason (`[EVICT L3 pol=ECG:rrip_first … -> victim=way… reason=…]`), proving the policy ACTS.
