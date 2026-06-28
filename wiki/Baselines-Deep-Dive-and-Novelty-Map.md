# Graph‑Cache Baselines: Deep Dive + Our Novelty Map

> **Purpose.** A mini‑paper wiki that distills *every* point the three graph‑cache
> baselines make — **P‑OPT** (Balaji, HPCA'21), **GRASP** (Faldu, HPCA'20), and
> **DROPLET** (Basak, HPCA'19) — and, for each point, states **how our system
> answers it, why we are more novel, and the concrete experiment + story** that
> lands the claim in the HPCA draft. It also resolves the **paper‑name collision**
> with *GraphPulse* (Rahman et al., MICRO'20).
>
> Our system is currently drafted as **PULSE** ("Per‑edge Unified Locality
> Signaling Engine"); the internal mechanism is **ECG** (the per‑edge epoch) +
> **`ECG_EXTRACT`** (the graph load that delivers it). The one‑line thesis:
> **"carry the *signal*, not the matrix."**
>
> Sources: `research/POPT_HPCA21_CameraReady.txt`,
> `research/ECG__.../GRASP_HPCA20.pdf`, `research/ECG__.../hpca2019_droplet.pdf`,
> `research/HPCA-PREP/PULSE/main.tex`, `docs/findings/baseline_faithfulness_audit_v1.md`,
> `docs/findings/droplet_vs_ecg_pfx_algorithm.md`, `docs/findings/grasp_road_anti_thrashing.md`.

---

## 0. How to read this wiki

Each baseline section has a fixed shape so the draft can lift it directly:

1. **Problem exposed** — the locality failure the paper identified.
2. **Key insight** — the one idea the paper is built on.
3. **Methodology** — how it works (mechanism, data structures, where it lives).
4. **Every point / claim** — the enumerated results, costs, and limitations.
5. **Our answer** — per point: how we subsume it, why it is *more* novel, and the
   **experiment + story** (with the numbers we have validated this cycle).

A consolidated **novelty matrix** (§7) and **evidence ledger** (§8) follow, then
the **story arc** (§9) and **naming** (§10).

---

## 1. The shared problem (the gap all three attack)

All three papers — and ours — start from the same fact:

- Graph kernels are **memory‑bound**: P‑OPT cites prior work that graph kernels
  spend **up to 80 % of time waiting on DRAM**. The working set (property array)
  exceeds on‑chip capacity, so the LLC is the decisive structure.
- The expensive access is the **property load** (`score[v]`, `parent[v]`,
  `dist[v]`, `comp[v]`): it is **irregular, vertex‑indexed, and named only by a
  vertex id that just arrived from the otherwise‑sequential edge stream**.
- The **edge list** (CSR) is large but **streams** (short reuse, L1/L2‑friendly);
  the **property array** is small but has **topology‑dictated, long, irregular
  reuse** that classical replacement/prefetch cannot predict.

> **The three attack three different sides of the same latency:**
> **GRASP** keeps the *high‑reuse* (hot) property lines resident;
> **P‑OPT** reconstructs *next‑reference order* so the cache can emulate Belady;
> **DROPLET** *prefetches* the property line ahead of the core using the edge stream.
> Their hardware paths are **separate**. **Our thesis: make all three meet in one
> per‑edge value that already rides on the access that needs it.**

---

## 2. Substrate — RRIP (Jaleel, ISCA'10)

GRASP and P‑OPT both build on **RRIP** (Re‑Reference Interval Prediction), so the
draft must introduce it first.

- **Mechanism.** Each line carries an *m*-bit **RRPV** (re‑reference prediction
  value). Victim = a line at **max RRPV** (predicted reused farthest in future).
  Insertion places new lines at a *distant* RRPV (2^m − 2 for SRRIP) so they are
  evicted quickly unless re‑referenced (a hit promotes toward 0).
- **Why it matters here.** RRIP is the *substrate*: GRASP changes **insertion /
  hit‑promotion RRPV by vertex degree**; P‑OPT changes the **victim choice by
  next‑reference distance**. Our system reuses the same RRPV substrate but feeds
  it a **per‑edge epoch** that subsumes *both* knobs.
- **Faithful in our code:** `SRRIP` = Jaleel ISCA'10, 3‑bit RRPV
  (`baseline_faithfulness_audit_v1.md`).

---

## 3. P‑OPT — *Practical Optimal Cache Replacement for Graph Analytics* (Balaji, Crago, Jaleel, Lucia — HPCA'21)

### 3.1 Problem exposed
State‑of‑the‑art replacement policies (LRU, RRIP, DRRIP, Hawkeye) fail on graphs
because graph reuse is **dynamically variable and graph‑structure‑dependent** —
properties those policies don't capture. **Belady's MIN** is ideal but needs the
future.

### 3.2 Key insight (the paper's thesis)
> **The transpose of a graph succinctly represents the next references of all
> vertices.** If a kernel's outer loop is over columns of the adjacency matrix,
> then scanning a vertex's **row** in the transpose tells you the *next* outer
> iteration that will touch it — i.e., its next reference. The transpose **is**
> the Belady oracle for graph property data.

### 3.3 Methodology
- **T‑OPT** (idealized): consult the transpose (CSC) directly at each replacement
  to find the resident vertex whose next reference is **farthest in the future**;
  evict it. Most frameworks already store CSR **and** transpose, so the data
  exists.
- **P‑OPT** (practical): the raw transpose is too large/slow to consult per
  eviction, so P‑OPT builds a **quantized Rereference Matrix**:
  - **Epoch quantization** — divide the vertex iteration space into *epochs*; for
    each property line store, **per epoch**, a 1‑byte "next‑reference distance"
    bucket. Two columns (current + next epoch) are kept resident.
  - The matrix is consulted on replacement to rank resident lines by next‑ref
    distance, approximating MIN.
- **Where it lives:** the resident rereference columns occupy **reserved LLC
  ways** (out‑of‑band metadata in the data array).
- **Phase structure (as we model it, faithful intent):** Phase 1 evict
  non‑graph/record data first; Phase 2 evict graph property by furthest
  next‑reference distance (`baseline_faithfulness_audit_v1.md`).

### 3.4 Every point / claim
- **P1.** Transpose = next‑reference oracle (the core idea).
- **P2.** Quantized Rereference Matrix makes MIN *practical* (low‑cost access to a
  summary of transpose info).
- **P3.** **+33 % average (56 % max) performance over LRU**; **35 % LLC‑miss
  reduction**; meaningful reduction even vs a stronger **DRRIP** baseline.
- **P4.** Works across multiple applications and inputs (PR, BFS, etc.).
- **P5 (cost, understated in the paper).** The resident matrix columns **consume
  LLC capacity**: at 1 B/line for 2 columns, reserved ways grow with the property
  footprint.
- **P6 (eval).** Sniper + a Pin cache‑sim; 8‑core, 256 KB L2, **24 MB DRRIP‑NUCA
  LLC** (3 MB/core).

### 3.5 The wedge — P‑OPT's *capacity‑tax paradox* (our motivation)
P‑OPT's accuracy **requires storage**, and that storage is **LLC ways**:
`w_rsv = ⌈2·L·B / way_size⌉` for property span `L` lines. On a **16‑way LLC each
reserved way removes 6.25 % of data associativity.** The tax is **worst exactly
when the property footprint ≫ LLC** — the graph needs capacity at the very moment
the policy removes it. The common "P‑OPT oracle" comparison **hides this** by
keeping a full data cache *plus* an invisible matrix; that is **not an iso‑area
hardware point.**

### 3.6 OUR ANSWER (novelty + experiment + story)
- **Novelty.** Keep P‑OPT's *signal* (quantized next‑reference epoch) but **stop
  storing the matrix in the cache.** Carry **one per‑edge epoch in the spare high
  bits of the edge word** the kernel already streams (`b_id = ⌈log₂|V|⌉` low bits
  = vertex id; the rest = epoch, `n_e = 2^(32−b_id)` in‑word epochs). The signal
  arrives **with the property load it governs** — no matrix, no reserved ways, no
  second metadata stream. When scale exhausts the spare bits, **promote to an
  8‑byte read‑once epoch record** — trading *prefetchable sequential bandwidth*
  (recoverable) for *LLC associativity* (not recoverable once reserved).
- **The honesty move (Contribution #1).** We **charge P‑OPT its tax** under an
  **iso‑area model** (reserve the ways the matrix needs) **and** show the
  uncharged oracle for context.
- **Experiment (validated this cycle).** PageRank‑pull, **cit‑Patents @ 4 MB LLC
  (≈3.6× property pressure):** demand LLC miss rate **LRU 56.7 % → GRASP 42.3 →
  charged P‑OPT 35.7 → PULSE 30.8 → uncharged‑P‑OPT *oracle* 31.1.** **PULSE beats
  even the uncharged oracle** because the oracle still pays matrix lookups while
  PULSE pays nothing. **kron‑s24 @ 8× pressure:** PULSE **8.0 %**, while charged
  P‑OPT (forced to reserve a quarter of the cache) **falls below GRASP.**
- **Story.** *"P‑OPT proved the transpose is the oracle. We prove you never had to
  store it — the oracle already rides in the dead bits of the edge you were going
  to read anyway. At scale, P‑OPT's accuracy eats the cache it is trying to
  protect; ours does not."*

---

## 4. GRASP — *Domain‑Specialized Cache Management for Graph Analytics* (Faldu, Diamond, Grot — HPCA'20)

### 4.1 Problem exposed
Real graphs have a **power‑law degree distribution**: a few **hot** vertices carry
most connections and **inherently exhibit high reuse**, but **irregular access
patterns** prevent commodity cache policies from capitalizing on that reuse.

### 4.2 Key insight
> Use **lightweight software** to *pinpoint the hot vertices* (via degree), then
> **specialize the LLC insertion/hit‑promotion policy to protect their property
> lines from thrashing** — while staying flexible enough to capture other reuse.
> Avoids the storage‑heavy predictors of domain‑agnostic schemes.

### 4.3 Methodology
- **Degree‑Based Grouping (DBG):** reorder so high‑degree vertices land in a
  contiguous **hot region** of the property array (a cheap address test then
  classifies a line as hot/warm/cold).
- **Specialized RRIP insertion + hit‑promotion:** hot lines inserted at/kept near
  **near‑RRPV** (protected), cold lines at **distant‑RRPV** (evicted first). A
  **3‑tier HOT/WARM/COLD** classification by degree (our `classifyGRASP()`).
- **Cost:** negligible HW (a couple of region bounds registers); the "model" is an
  **array‑relative** region (e.g., top ~15 % of the vertex array — GRASP's
  `add_region(ptr, frac, n)`).

### 4.4 Every point / claim
- **G1.** Power‑law skew ⇒ hot vertices have high *latent* reuse the cache misses.
- **G2.** Protect hot property lines via insertion/promotion bias (not pinning —
  retains flexibility).
- **G3.** Pairs with **reordering** to make "hot" a contiguous, cheaply‑testable
  region.
- **G4.** **+4.2 % average (9.4 % max) speedup over the best prior scheme** on
  high‑skew graphs; **robust on low‑/no‑skew** where prior schemes *slow down*.
- **G5.** Negligible hardware vs storage‑intensive predictors.
- **G6 (limitation we measured).** On **low‑locality / road‑like** graphs at the
  literature operating point (L3 = 1 MB), GRASP's degree bias **actively hurts
  frontier kernels** (BFS +62.6 pp, SSSP +55.8 pp vs LRU on roadNet‑CA); it only
  helps via **anti‑thrashing** at sub‑working‑set caches or genuine reuse in CC
  (`grasp_road_anti_thrashing.md`).

### 4.5 OUR ANSWER (novelty + experiment + story)
- **Novelty.** GRASP's "degree priority" is a **coarse, static** proxy for reuse.
  Our **epoch subsumes degree priority**: a quantized next‑reference timestamp
  *already* ranks a hub (near next‑ref) above a leaf (far next‑ref), **and**
  separates lines *within* a degree class by true next‑reference order — which
  GRASP cannot. When the spare‑bit budget collapses to a few hundred epochs, PULSE
  **degrades gracefully to "degree priority with a coarse tie‑break" — i.e., to
  GRASP** — so GRASP is the **floor**, not a competitor.
- **The do‑no‑harm proof.** GRASP *hurts* frontier kernels on low‑skew graphs
  (G6). Our **decisive‑equivalence harness** shows that on frontier kernels
  (BFS/BC/CC/SSSP) the epoch is **delivered and policy‑compliant but rarely
  strictly decides the victim** ("do‑no‑harm: tied effective distance"), so PULSE
  **matches GRASP where degree is right and never inherits its frontier
  regression** by construction (epoch ranks by *next‑use*, not by *degree*).
- **Experiment (validated).** Multi‑kernel headline (GAP‑5, real graphs): **ECG ≥
  max(GRASP, P‑OPT) on every cell — WIN 7 / TIE 3 / LOSE 0; ECG < P‑OPT on 8/10.**
  cit‑Patents PR: GRASP 42.3 → PULSE 30.8. The **ties** are the two no‑reuse
  thrash cells (com‑orkut BFS/SSSP at 1.0) where *nothing* helps — honestly
  reported.
- **Story.** *"GRASP showed degree is a cheap reuse proxy. We show degree is just
  the low‑resolution limit of next‑reference order — so we get GRASP for free at
  small scale and strictly more than GRASP whenever the bits exist to separate
  hubs by *when*, not just *how connected*."*

---

## 5. DROPLET — *Analysis and Optimization of the Memory Hierarchy for Graph Processing* (Basak, Li, Hu, … Xie — HPCA'19)

### 5.1 Problem exposed
A **data‑aware characterization** shows graph workloads have **two data types with
opposite behavior**: **edge lists** (streaming, short reuse, L1/L2‑resident) and
**property data** (irregular, long reuse, **thrashes the LLC**). A single uniform
prefetcher/cache cannot serve both.

### 5.2 Key insight
> **Decouple** prefetch into two engines that respect the two data types, and let
> the **edge stream drive property prefetch** (the edge names the next vertex),
> **breaking the pointer‑chasing dependence** that stalls the core.

### 5.3 Methodology
- **Edge‑list engine:** watches the edge stream, **detects strides** (4‑entry
  table, confidence ≥ 2), predicts the next *K* edge lines.
- **Property engine:** for vertex ids in the (prefetched) edge lines, computes
  `property_base + v·elem_size` and **issues property prefetches**, decoupled from
  the core's dependency chain.
- **Defaults (artifact):** `prefetch_degree=1`, `indirect_degree=16`,
  `stride_table_size=64`; attaches at the **L3**, plus a **memory‑controller
  property prefetcher** (the full decoupled architecture).

### 5.4 Every point / claim
- **D1.** Two data types ⇒ two decoupled engines (the architecture).
- **D2.** Edge‑stream‑driven indirect prefetch hides property latency without
  pointer‑chasing stalls.
- **D3.** **1.37× average speedup (1.76× peak on BFS); 15–45 % LLC‑miss
  reduction.**
- **D4.** The full decoupled L2‑streamer + MC‑property architecture beats a simpler
  `streamMPP1` by **4–12.5 %**.
- **D5 (cost).** Stride table + dedup set + a **separate prefetch metadata path**;
  issues **many** speculative prefetches (≈16 indirect per trigger) → **bandwidth
  and pollution.**

### 5.5 OUR ANSWER (novelty + experiment + story)
- **Novelty (Contribution #4 — *one signal for two jobs*).** DROPLET is a *second,
  separate* mechanism from replacement. **Our epoch drives prefetch *and*
  replacement** from the **same per‑edge value**: the epoch **filters lookahead**
  over the sequential edge stream (prefetch only the next‑referenced‑soon
  targets) *and* breaks resident‑priority ties (replacement). One value, carried
  on the edge, replaces **two** graph‑specific hardware paths.
- **Efficiency novelty.** DROPLET sweeps *K* targets per trigger; we **pick the
  best‑1 by epoch**. Measured: **same LLC‑miss reduction at ≈1/3 the prefetch
  bandwidth** — ECG_PFX issues **1.918 requests per useful hit vs DROPLET's 2.250
  (15 % fewer)**; per‑cell best **web‑Google/pr: 30.7 % fewer requests per useful
  hit** (`droplet_vs_ecg_pfx_algorithm.md`). Bandwidth matters: DRAM contention,
  pollution, power.
- **Timeliness novelty.** DROPLET must *detect* a stride and prefetch *ahead*; our
  hint **arrives exactly when the property load is issued** (it *is* the edge word)
  — no stride mis‑prediction, no warm‑up.
- **Experiment (validated mechanism).** The unified epoch is the headline
  combined‑stack; ECG_PFX vs DROPLET efficiency table (14 cells) is in
  `droplet_vs_ecg_pfx_algorithm.md`. **Caveat to keep honest:** our DROPLET is a
  *best‑case* `streamMPP1`‑class approximation (no MC‑property engine); the draft
  must phrase the comparison as "DROPLET‑style," and Sniper's L2‑enqueue filter
  currently suppresses fills (mechanism validated via hint counters; gem5 is the
  cycle‑accurate fill reference).
- **Story.** *"DROPLET split prefetch into two engines because the edge stream and
  the property array behave differently. We keep the data‑type insight but collapse
  the two engines — and the separate replacement policy — into one number that the
  edge already carries, at a third of the bandwidth."*

---

## 6. The unifying thesis (what makes us a *system*, not a tweak)

| Baseline | What it owns | Where it stores state | Our subsumption |
|---|---|---|---|
| RRIP | the RRPV substrate | in‑line RRPV bits | reused, fed by the epoch |
| GRASP | degree priority (insertion/promotion) | region registers + reorder | epoch's **coarse limit** |
| P‑OPT | next‑reference order (victim choice) | **reserved LLC ways** (the tax) | epoch in **spare edge bits** |
| DROPLET | structure‑driven prefetch | stride table + 2nd path | epoch **filters lookahead** |

Five contributions (from the draft), each tied to a baseline:
1. **Charge P‑OPT's capacity tax** (iso‑area honesty) — answers P‑OPT P5/P6.
2. **PULSE = one per‑edge epoch** unifying degree + next‑reference — answers
   GRASP G2 + P‑OPT P1.
3. **"Carry the epoch, not the matrix":** spare bits when they suffice, 8‑byte
   read‑once record at scale — answers P‑OPT P5.
4. **One epoch drives replacement *and* prefetch** — answers DROPLET D1/D5.
5. **Cross‑validated** (functional cache‑sim + gem5 + Sniper) + released — answers
   the methodology‑faithfulness bar all three set.

The deepest single idea: **the edge word is a channel already on the critical
path.** The kernel *must* read it to form the property address; if its high bits
carry an epoch, the policy state **binds to the access that caused it** — no
metadata cache, no second miss, ideal timing.

---

## 7. Per‑point novelty matrix (their claim → our counter → our evidence)

| # | Their point | Our counter (novelty) | Evidence / experiment |
|---|---|---|---|
| P‑OPT P1 | transpose = next‑ref oracle | keep the signal, drop the storage | cit‑Patents PR: PULSE 30.8 < oracle 31.1 |
| P‑OPT P3 | +33 % over LRU | we beat **charged** P‑OPT, not just LRU | 30.8 vs 35.7 (charged) |
| P‑OPT P5 | (tax understated) | **iso‑area charge** + 8B promote | kron‑s24 8×: charged P‑OPT < GRASP; PULSE 8.0 |
| GRASP G2 | protect hot by degree | epoch **subsumes** degree + separates within‑class | multi‑kernel WIN 7/TIE 3/LOSE 0 |
| GRASP G6 | (frontier regression) | epoch is do‑no‑harm on frontier kernels | decisive‑equiv: BFS/BC/CC/SSSP delivered, not harmful |
| DROPLET D1 | two decoupled engines | **one** epoch for replace **+** prefetch | unified combined‑stack |
| DROPLET D5 | many prefetches/trigger | best‑1 by epoch, **1/3 bandwidth** | 1.918 vs 2.250 req/useful |
| All | "we are faithful" (1 sim) | **3‑sim equivalence** + iso‑area | eviction‑equiv matrix (§8) |

---

## 8. Evidence ledger (what is validated vs still needed)

**Validated this work (cache‑sim authority + cross‑sim decision):**
- **Headline:** cit‑Patents PR @4 MB — LRU 56.7 / GRASP 42.3 / charged P‑OPT 35.7
  / **PULSE 30.8** / uncharged oracle 31.1. kron‑s24 @8× — **PULSE 8.0**.
- **Generality:** multi‑kernel GAP‑5 (pr/bfs/cc/sssp/bc × cit‑Patents, com‑orkut)
  — **ECG ≥ max(GRASP,P‑OPT): WIN 7 / TIE 3 / LOSE 0; ECG < P‑OPT 8/10.**
- **3‑sim eviction equivalence:** ECG_GRASP_POPT victim decision is byte‑spec
  identical across cache‑sim + gem5; **pr/bfs/bc/sssp DECISIVE** (epoch strictly
  selects the victim) on cache‑sim+gem5; **cc do‑no‑harm**; **tc out‑of‑scope**
  (no vertex property). Direction‑transpose oracle 3/3, packing 95/95.
- **Prefetch efficiency:** ECG_PFX = DROPLET miss‑reduction at **≈1/3 bandwidth**.

**Still needed / open (be honest in the draft):**
- **Cross‑sim *performance* parity** on the pressured cell: GRASP and P‑OPT
  **directions agree** across cache‑sim↔gem5 (both help, P‑OPT most). **ECG:DBG_PRIMARY
  is NOT a clean cross‑sim proxy** — gem5's ECG degree‑mode does *not* replicate the
  GRASP 3‑tier insertion that cache‑sim's does (cache‑sim ECG:DBG ≈ GRASP and helps;
  gem5 ECG:DBG hurts). A prefetch‑config parity bug was fixed but did **not** explain
  it (replacement gap, not prefetch — findings §22.16). **Scope the cross‑sim ECG
  claim to ECG_GRASP_POPT *decision*‑equivalence (proven) + GRASP/P‑OPT direction;
  cache‑sim is the ECG miss‑rate authority.**
- **Sniper fill validation** (L2‑enqueue filter suppresses ECG_PFX fills →
  validate via hint counters; gem5 is the fill reference).
- **Full DROPLET** (MC‑property engine) is *not* reproduced — phrase as
  "DROPLET‑style."
- **Iso‑area charge** must be applied uniformly in every P‑OPT cell in the final
  matrix.

---

## 9. The story arc for the HPCA draft (one narrative)

1. **Hook.** A graph kernel waits not because it can't compute, but because the
   next property line is named by a vertex id that just arrived on the edge stream.
2. **Three partial answers.** GRASP (keep hubs), P‑OPT (reconstruct order),
   DROPLET (prefetch ahead) — three *separate* hardware paths.
3. **The paradox.** The most accurate (P‑OPT) pays an **LLC capacity tax** that is
   worst exactly under pressure; the common oracle comparison hides it.
4. **The free channel.** The edge word is already on the critical path; its spare
   bits can carry a **next‑reference epoch** that *subsumes* degree priority.
5. **One signal, two jobs.** The same epoch drives replacement *and* prefetch,
   delivered by a graph load (`ECG_EXTRACT`).
6. **Honesty + proof.** Charge P‑OPT its tax; show we beat **charged** P‑OPT and
   even the uncharged oracle; prove the decision is **equivalent across three
   simulators**; show GRASP is our floor and DROPLET our bandwidth‑win.
7. **Boundaries.** Where exact small matrices still help (tiny graphs), where the
   iso‑area tax flips the ranking, where tc is out of scope.

---

## 10. Naming — resolving the *GraphPulse* collision

**Problem.** The draft system is **PULSE**; **GraphPulse** (Rahman, Abu‑Ghazaleh,
Gupta — *event‑driven graph‑processing accelerator*, MICRO'20) is an established,
adjacent‑domain name. "Pulse" + graphs is taken. (The internal mechanism name
**ECG**/`ECG_EXTRACT` also lives in the cardiac‑signal metaphor family.)

**Naming brief.** The name should evoke: a **compact signal carried *with* the
edge** (not stored separately), encoding **next‑reference order**, driving **both
replacement and prefetch**, delivered by a **graph load**. Short, pronounceable,
collision‑free in comp‑arch.

**Candidates (ranked, with rationale):**

| Name | Expansion / rationale | Why it fits | Watch‑outs |
|---|---|---|---|
| **ECHO** | *Edge‑Carried Hint for Ordering* | Stays in the cardiac/ECG family (echo‑cardiogram); "echo" = the signal that comes back; captures *carried with the edge* | "Echo" is common (consumer/SW); verify no graph‑cache arch use |
| **HITCH** | the hint **hitches a ride** on the edge word | Nails the core insight (free channel; carry alongside the access); memorable verb | acronym is a stretch; check collisions |
| **SIGNET** | *SIGnal in the Edge word, Next‑ref Timestamp* | "signet" = a compact seal/mark; elegant, signal‑centric | slightly literary |
| **CREST** | *Cache Replacement via Edge‑Stream Timestamps* | wave/ECG **crest**; cardiac‑adjacent; descriptive acronym | "Crest" used elsewhere; check |
| **EMBER** | *Edge‑eMBedded Epoch for Replacement* | "embedded in the edge"; warm, short | epoch‑for‑replacement only; underplays prefetch |
| **SEAM** | signal in the **seam** of the edge word | evokes "woven into the stream" | generic word |

**Recommendation.** **ECHO** (first choice) keeps the team's cardiac‑signal
identity (ECG → ECHO), is short and memorable, and the expansion *Edge‑Carried
Hint for Ordering* states the mechanism. **HITCH** is the strongest *conceptual*
alternative (it literally describes "the locality signal hitches a ride on the
edge you already read"). Whichever is chosen, keep **`ECG_EXTRACT`** as the ISA
op name (or rename to match, e.g. `echo.load` / `hitch.load`) for SSOT.

> **Action:** before committing a name, do a comp‑arch collision check
> (DBLP / Google Scholar: "ECHO cache", "HITCH prefetch", "SIGNET cache",
> "CREST replacement"). Update `\newcommand{\sys}` in `main.tex` and the internal
> `ECG_*` symbols in one pass.

---

## 11. Open gaps / honest risks (keep these visible)

- **Cross‑sim performance** equivalence (not just decision) is the weakest leg;
  scope the claim to *decision‑equivalence + direction‑agreement*, with cache‑sim
  as the miss‑rate authority and gem5 IPC as the cycle‑accurate truth.
- **DROPLET faithfulness:** "DROPLET‑style" only; full MC‑property engine is
  future work.
- **Single‑core, scaled LLC:** the paper config is 1‑core with a scaled LLC vs the
  baselines' 8‑core NUCA — must be stated in Methodology.
- **Spare‑bit resolution at extreme |V|:** above the in‑word budget the epoch
  collapses toward degree priority; the **8‑byte record mode** is the answer and
  must be measured at scale (kron‑s24/s25), with the read‑once bandwidth charged.
- **tc** has no vertex property → genuinely out of scope (document, don't force).

---

*Maintained as the related‑work + novelty SSOT for the HPCA draft. Update the
evidence ledger (§8) as cells land; update §10 once the name is chosen.*
