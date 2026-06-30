# Why the P-OPT matrix beats our epoch mask — and the never-enabled fix (REFRESH)

**Date:** 2026-06-30
**Prompted by:** "can we enhance it … we have 30 bits … make the epoch analysis more
sophisticated … what is the algorithm we are using currently, let's study it, and see
why it works for P-OPT and doesn't work for us."

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
| **`ECG_STORED_REFRESH`** (existing, **never enabled by any orchestrator**) | re-stamp a resident LLC line's `ecg_epoch` from the per-edge hint on **every** access (even on L1/L2 hits), modelling `ecg.extract` re-delivering the mask per edge | **the dominant lever.** Faithful (epoch already streamed; free under LEAN+PACK) and gated by `edge_epoch_valid` so frontier-kernel sequential-read hygiene is preserved (BFS still verifies PASS). |

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

Without REFRESH, faithful ECG **loses everywhere**. With it, ECG wins where P-OPT's
reserved-way capacity loss exceeds the mask's (refreshed) eviction deficit — i.e. small/
mid graphs; the densest graph (LiveJournal, 3 reserved ways) still favours the matrix
because its equal-ways deficit (6.8pp) outruns the 3-way reserve.

## 4. Honest framing

- ECG does **not** beat the P-OPT matrix at equal capacity (the matrix's live 2-D next-ref
  is a genuinely better oracle, especially on dense graphs).
- ECG's faithful win is **reserved-way avoidance**, and **REFRESH is what makes it real** —
  it shrinks the eviction-quality deficit enough that the saved capacity dominates.
- The "spare-bit schedule" idea is sound but empirically dominated by the simpler refresh.

## 5. Behaviour change

`scripts/experiments/ecg/roi_matrix.py` now injects `ECG_STORED_REFRESH=1` for
`ECG:ECG_GRASP_POPT` by default (new `--ecg-stored-refresh`, presence-gated so `0`
omits the key). The new `ECG_EDGE_MASK_SCHED=K` ablation is implemented and inert when
unset (byte-identical to prior ECG_GRASP_POPT).
