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
