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
from `g.out_neigh`. `buildInEdgeMasksBFS(g)` is the self-contained IN-edge mirror of
`buildOutEdgeMasks` (LOCAL sorted out-adjacency; fills only `in_edge_*`; fills the epoch
**unconditionally**, unlike `buildInEdgeMasks_PR` which only fills it under
`ECG_EDGE_MASK_EPOCH`). So with the flag set: **TD uses OUT-edge masks, BU uses IN-edge
masks** — both phases masked per their own edge list.

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
