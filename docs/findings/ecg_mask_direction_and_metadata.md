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
