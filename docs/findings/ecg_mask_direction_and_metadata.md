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
    | Path A prefetch (epoch-filtered next-K lookahead, **headline**) | yes | yes (§11) | **NO — Path B only** |
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
   the headline): cache_sim + gem5 (§11, size-independent); **Sniper has Path B only** — do NOT
   report a Sniper Path A number until it is ported. `ECG_PFX_STRICT_TARGET=1` still guards any
   residual gem5 >24-bit (>16M-vertex) truncation.
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
  faithful at scale (packed-flat). **Open:** Sniper Path A (epoch-filtered lookahead) not yet
  ported — Sniper currently runs Path B (hub-ranked single target) only.

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
