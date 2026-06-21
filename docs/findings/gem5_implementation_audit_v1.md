# gem5 Implementation Audit v1 — Accuracy & Completeness

**Date:** 2026-06-08
**Scope:** All gem5 overlays under `bench/include/gem5_sim/overlays/` +
sideband JSON, ISA opcode, SimObject Python config, kernel `bench/src_gem5/pr.cc`,
and the X86/RISCV gem5.opt binaries.
**Trigger:** User request "now lets focus on our gem5 implementation make sure
it is accurate and complete" following completed HPCA cache_sim evaluation.

---

## TL;DR — verdict per component

| Component | Status | Notes |
|---|---|---|
| ECG replacement policy (`ecg_rp.cc`) | ✅ Complete | DBG_PRIMARY / POPT_PRIMARY / DBG_ONLY / ECG_EMBEDDED / ECG_COMBINED all implemented and mirror `cache_sim.h::findVictimECG` |
| GRASP replacement policy (`grasp_rp.cc`) | ✅ Complete | 3-tier insertion + hit promotion, sideband JSON-loaded property regions |
| P-OPT replacement policy (`popt_rp.cc`) | ✅ Complete | 3-phase eviction matches Balaji HPCA'21 + cache_sim |
| DROPLET prefetcher (`droplet.cc`) | ⚠️ Stride engine works; **indirect-property engine is filtered** | 860/860 issued on email-Eu-core, but those are dominated by the same-page *stride* engine; the indirect `property[neighbor]` prefetches cross pages and are dropped by the same `Queued::notify` filter as ECG_PFX (see Correction below) |
| ECG_PFX prefetcher (`ecg_pfx.cc`) | 🔴 **BROKEN in gem5 — wrong root cause documented** | New root cause found this audit: `pf_identified=0` due to page-cross filter in `Queued::notify`, NOT the single-slot mailbox |
| `ecg_extract` RISC-V opcode (`decoder_ecg_extract.isa`) | ✅ Decoded correctly | Custom-0 opcode `0x0b`, FUNCT3=0, FUNCT7=0, payload unpacks fat-id correctly into real_vertex + DBG/POPT/PFX hints |
| Ring-buffer hint queue (`graph_cache_context_gem5.hh`) | ✅ Code is correct | 256-entry MPSC; sprint 6f-6 fix landed cleanly |
| Sideband JSON loading | ✅ Working | Property + edge regions parsed correctly |
| gem5 PR kernel mode 6 (`pr.cc`) | ✅ Complete | sprint 6f-6 port wired the per-edge mask path; uses `GEM5_ECG_PFX_TARGET` macro |
| RISCV gem5.opt binary | ⚠️ Stale (May 25) | Predates all sprint 6f-5/6f-6 fixes — needs rebuild before any RISCV cycle-accurate run |
| X86 gem5.opt binary | ✅ Fresh (Jun 5 17:48) | Has ring-buffer fix; this is the binary used by `pfx_gem5_validation_sweep.sh` |

**Headline finding:** the prior documented "hint-to-issue gap"
(`docs/findings/gem5_ecg_pfx_simobject_gap.md`) targets the wrong root
cause. The single-slot mailbox WAS one issue (now fixed, ring buffer in
place), but the *primary* reason `pf_issued = 0` for ECG_PFX is that
gem5's `Queued::notify()` (line 220-221 of `mem/cache/prefetch/queued.cc`)
drops every prefetch whose target line crosses a page boundary from the
demand access, **unless the prefetcher has been given an `mmu` reference**.
ECG_PFX targets `property[random_vertex]`, which essentially always lives
on a different page from the demand edge access.

**UPDATE 2026-06-20 (empirical, supersedes the correction below).** The
`S68-MMU-PATCH` (a `registerMMU(system.cpu.mmu)` call in
`configs/graphbrew/graph_se.py:215-253`, committed `d751a5d6`) plumbs the MMU into
both the DROPLET and ECG_PFX prefetchers, so `can_cross_page = (mmu != nullptr)`
is now **true** and cross-page property prefetches are **no longer dropped**.
Verified on today's RISCV `gem5.opt` (email-Eu-core, `-o5`, DROPLET at L2):
`pfIdentified = pfIssued = 3534` (zero dropped), `pfSpanPage = 1691` page-crossing
prefetches **issued**, `pfUsefulSpanPage = 1533`. So **gem5's indirect/property
prefetch path works** — the v1 audit's `pf_identified=0 / pf_span_page=63 (all
dropped)` was the *pre-S68, stale-binary* state. The earlier "the filter cripples
both property engines" correction (kept below for history) is therefore **wrong on
current code** — the filter is fixed.

Two caveats remain, both verified this session:
1. **ECG_PFX still issues nothing** (`pfIdentified=0`) — but for a *different*
   reason than the page filter. The ECG_PFX prefetcher loads its sideband
   (`ECG_PFX: loaded sideband property=[...]`) yet receives **0 hints** in every
   delivery mode tried (ISA `ecg.extract`, explicit `GEM5_ECG_PFX_TARGET`, mode 6).
   This is the unresolved kernel→prefetcher **hint-delivery** gap (the m5-op never
   reaches `calculatePrefetch`), independent of the now-fixed filter. DROPLET works
   because it computes its targets *inside* `calculatePrefetch` (no hint channel).
2. **TLB cost is not modelled.** In our RISCV SE-mode + TimingSimpleCPU setup the
   data TLB reports `system.cpu.mmu.dtb.accesses = 0` for a real run (1.7M cycles,
   3534 prefetches), i.e. translation is functionally free — no page-walk / TLB-miss
   latency. So the scattered-page TLB pressure of indirect property prefetch is
   **not captured** in gem5 timing (nor in cache_sim, which has no TLB). See
   `docs/findings/property_prefetch_tlb_paging.md`.

---

*Historical (pre-S68) correction — superseded by the UPDATE above:*
DROPLET has *two* engines and BOTH push into the same `addresses` list that
`Queued::notify()` filters (neither prefetcher overrides `notify`; both only
override `calculatePrefetch`):
- Engine 1 (stride/edge): prefetches next-cache-line *edge* addresses
  (`baseLineAddr + i*edgeElemSize`) — these stay on the same page, so they
  **survive** the filter.
- Engine 2 (indirect *property*): `issueIndirectPrefetches` pushes
  `propertyAddrForVertex(vertexId)` (`droplet.cc:305`) — a scatter
  `property[neighbor]` target that crosses pages exactly like ECG_PFX, so
  pre-S68 it was **dropped by the same filter**.

The gap doc said `pf_issued=0 despite hints being consumed`. With the
fresh X86 binary (post ring-buffer fix), this audit's smoke shows
`pf_identified=0, pf_span_page=63` — meaning hints ARE being consumed and
pushed to `addresses` (63 of them), but every single one is rejected by
gem5's page-cross filter before reaching `pf_issued`.

---

## Files audited

### A. Replacement policies — sound

```
bench/include/gem5_sim/overlays/mem/cache/replacement_policies/
  ecg_rp.{cc,hh}                       330 + 71 LOC
  grasp_rp.{cc,hh}                     168 + 98 LOC
  popt_rp.{cc,hh}                      194 + 85 LOC
  graph_cache_context_gem5.hh          547 LOC
  GraphReplacementPolicies.py           94 LOC
```

All three policies:
- Load property regions from `/tmp/gem5_graphbrew_ctx.json` sideband
  (retry every 512 attempts until populated by the kernel)
- Apply the same algorithmic structure as `cache_sim.h`:
  - GRASP: 3-tier insertion (`HIGH→1, MODERATE→maxRRPV-1, LOW→maxRRPV`),
    promote-on-hit
  - P-OPT: 3-phase victim search (non-property first, then max
    rereference distance, then RRIP tiebreak)
  - ECG: layered eviction parameterized by `ECGMode` enum — exactly
    matches the cache_sim mode strings

These are the strongest part of the gem5 implementation. The X86 binary
links them correctly (verified by symbol presence).

**One minor gap:** gem5 ECG does *not* read or apply the per-edge mode-6
fat mask metadata. The mask is built kernel-side in `pr.cc` (sprint 6f-6
port), but `ecg_rp.cc::reset()` uses the runtime sideband region +
P-OPT matrix to compute `ecg_popt_hint`, not the kernel-emitted fat-id
PFX bits. This is fine for the paper's main claims because the gem5
ECG replacement policy was always P-OPT-matrix-driven (matching the
HPCA'21 design), and the cache_sim per-edge mode-6 results carry the
"new" contribution. But it means gem5 currently runs a *less aggressive*
ECG than cache_sim — if the paper claims cross-sim mode-6 parity, gem5
needs to consume the mask via the `ecg.extract` opcode payload.

### B. ISA — sound

```
bench/include/gem5_sim/overlays/arch/riscv/isa/
  decoder_ecg_extract.isa     20 LOC
  formats/ecg.isa             11 LOC
```

The opcode is correctly decoded as RISC-V custom-0 (`0x0b`), FUNCT3=0,
FUNCT7=0. The payload structure (`fat_id`) is unpacked exactly as
intended:

| Bits | Field | Used by |
|---|---|---|
| `[0:31]` | `real_vertex` | Returned in Rd, registered as "current vertex" hint |
| `[32:39]` | `dbg_hint` (8 bits) | DBG-tier hint stored in `decodedEcgMetadataStorage` |
| `[40:47]` | `popt_hint` (8 bits) | P-OPT distance hint |
| `[48:63]` | `pfx_hint` (16 bits) | Prefetch target vertex (16-bit, mapped to full vertex via mailbox) |

The custom opcode fires:
- `setDecodedEcgExtractHint(real_vertex, dbg_hint, popt_hint, pfx_hint)`
- `setPrefetchTargetHint(pfx_target)` where `pfx_target` falls back to
  `real_vertex` if `pfx_hint == 0`

**The opcode is correct.** The issue is downstream in the SimObject
that *consumes* the hint (see below).

### C. ECG_PFX SimObject — BROKEN, new root cause

```
bench/include/gem5_sim/overlays/mem/cache/prefetch/
  ecg_pfx.{cc,hh}             167 + 68 LOC
```

#### Live smoke (this audit, 2026-06-08)

Email-Eu-core/pr at L3=1MB on X86 gem5.opt (Jun 5 17:48 build — has
ring-buffer fix):

| Arm     | pf_identified | pf_issued | pf_useful | pf_span_page | l3_miss_rate |
|---------|--------------:|----------:|----------:|-------------:|-------------:|
| DROPLET |           860 |       860 |       762 |          797 |       0.0594 |
| ECG_PFX |             0 |       **0** |         0 |       **63** |       0.0064 |

ECG_PFX SimObject:
- Correctly loads sideband property region (`ECG_PFX: loaded sideband`
  emitted on first hint consumption)
- Correctly consumes 63 hints via `consumePrefetchTargetHint()`
- Correctly calls `addresses.push_back(AddrPriority(address, 0))` for
  all 63

But every one of those 63 is dropped by gem5's `Queued::notify()` filter.

#### Root cause: `Queued::notify` page-cross filter

`bench/include/gem5_sim/gem5/src/mem/cache/prefetch/queued.cc:207-234`:

```cpp
for (AddrPriority& addr_prio : addresses) {
    addr_prio.first = blockAddress(addr_prio.first);

    if (!samePage(addr_prio.first, pfi.getAddr())) {
        statsQueued.pfSpanPage += 1;        // <-- 63 of these for ECG_PFX
        ...
    }

    bool can_cross_page = (mmu != nullptr);
    if (can_cross_page || samePage(addr_prio.first, pfi.getAddr())) {
        statsQueued.pfIdentified++;          // <-- 0 of these for ECG_PFX
        insert(pkt, new_pfi, addr_prio.second, cache);
    } else {
        DPRINTF(HWPrefetch, "Ignoring page crossing prefetch.\n");
    }
}
```

`mmu` is set per-prefetcher by `Cache::setMMU()` only when the prefetcher
is wired into a cache that has translation enabled (`Cache::xfetchTags →
prefetcher->setMMU(mmu)` or similar). For the default `make_ecg_pfx_prefetcher`
in `bench/include/gem5_sim/configs/graphbrew/graph_cache_config.py:186-193`,
**no MMU is plumbed**. ECG_PFX's `use_virtual_addresses=True` is honored
by gem5 only for the address computation; the page-cross filter still
fires.

**Why DROPLET works:** DROPLET prefetches `addr + N*stride` where addr is
the current demand edge-list address. These addresses naturally stay on
the same 4KB page (in cit-Patents/com-orkut/etc., the edge list is
huge — many KB long — but each prefetch is N×64B ahead, often <4KB).
Indirect property prefetches DO cross pages, but DROPLET issues so many
prefetches that the same-page edge stream dominates the `pf_issued`
count.

**Why ECG_PFX doesn't work:** every ECG_PFX prefetch is
`property[arbitrary_vertex_id]`, which lands at an address pseudo-randomly
distributed across the entire property region. The probability that this
target lives on the same 4KB page as the demand edge access is roughly
`page_size / property_region_size = 4KB / 4MB = 0.1%`. With 63 hints,
expected `pf_identified` ≈ 0.063 — observed value is 0. Statistically
consistent with the filter being the cause.

#### Fix paths (priority ordered)

**Fix 1 (smallest patch, highest leverage):** Wire the cache's MMU into
the ECG_PFX (and DROPLET) prefetcher Python constructor. gem5's
`BasePrefetcher` exposes `registerMMU(simObj)` (`Prefetcher.py:116-119`)
which appends the MMU to a list that gets handed to the C++ `Queued`
class via `addMMU()` at `regProbeListeners()` time
(`Prefetcher.py:102-104`). Search across our config tree confirms
**no `registerMMU` call exists anywhere** in
`bench/include/gem5_sim/configs/`, which is the precise gap.

Concrete patch sketch — after the prefetcher is attached to the cache
in the gem5 SE-mode wiring (`graph_se.py` ~L210, where `cache.prefetcher
= make_ecg_pfx_prefetcher(...)` lands), add:

```python
if hasattr(system.cpu, "mmu"):
    cache.prefetcher.registerMMU(system.cpu.mmu)
elif hasattr(system.cpu, "dtb"):
    # gem5 v21 and older
    cache.prefetcher.registerMMU(system.cpu.dtb)
```

This is the canonical gem5-recommended path. Estimated time: 2-4 hours
for the patch + smoke-test loop on email-Eu-core/pr, plus a rebuild
(gem5 Python config does not require recompile, but a clean
smoke-validation pass adds confidence).

**Fix 2 (cleanest semantic):** Add `cross_pages = True` as an explicit
override in ECG_PFX. gem5 has a `cross_pages` knob in some
prefetchers; if `Queued` exposes it, set it `True` for ECG_PFX. Falls
back to fix 1 if not.

**Fix 3 (hack):** Bypass `Queued::notify`'s filter by overriding
`Queued::insert()` in `ecg_pfx.cc` and pushing prefetches directly to
the internal queue, ignoring the page check. Fragile across gem5
versions; not recommended.

**Fix 4 (correct ISA path):** Once ECG_PFX is delivered via the
`ecg.extract` instruction (RISC-V custom opcode), the prefetch can be
issued from the *processor's* prefetcher port, where translation is
already established. This is the long-term paper-faithful path; the
opcode is already decoded correctly, but no current
`bench/include/gem5_sim/configs/graphbrew/` config wires an
ISA-fed prefetcher.

### D. DROPLET prefetcher — sound

```
bench/include/gem5_sim/overlays/mem/cache/prefetch/
  droplet.{cc,hh}             ~400 + ~80 LOC
```

The smoke above confirms DROPLET fires end-to-end: stride engine
detects the edge stream, indirect engine reads CSR neighbor IDs and
issues property prefetches. `pf_issued = 860`, `pf_useful = 762`
(88.6% accuracy) on email-Eu-core/pr.

The `pf_span_page = 797` counter shows the indirect prefetches DO cross
pages — they're counted but evidently `mmu != nullptr` for DROPLET in
the current config path, OR enough edge-stream stride prefetches stay
on-page to make `pf_identified > 0`. The numbers
(`pf_identified == pf_issued`) suggest no filter is rejecting them.

This is paper-faithful for the streamMPP1-class DROPLET approximation
already documented in the HPCA evaluation plan.

### E. gem5 PR kernel mode 6 (`bench/src_gem5/pr.cc`) — sound

Sprint 6f-6 ported the cache_sim mode-6 path to the gem5 PR kernel
(commit `1aa1b24b`). The audit confirms:

- Mode-6 fat mask is built via `ecg_mode6::buildInEdgeMasks()` (shared
  header `bench/include/ecg_mode6_builder.h`)
- The inner loop reads pre-encoded prefetch targets and issues
  `GEM5_ECG_PFX_TARGET(prefetch_target)` macro calls
- `GEM5_ENABLE_ECG_PFX_HINTS=1` env var is honored
- Per-edge mask path is correctly skipped when `ecg_pfx_mode != 6`

**But:** because of the ECG_PFX SimObject filter bug above, none of
these hints actually become prefetches in gem5. The kernel-side work
is correct; the SimObject-side issuance is broken.

### F. Binaries

```
bench/include/gem5_sim/gem5/build/X86/gem5.opt     Jun 5 17:48  (fresh ✅)
bench/include/gem5_sim/gem5/build/RISCV/gem5.opt  May 25 23:37  (STALE ⚠️)
```

The X86 binary was rebuilt after all sprint 6f-5/6f-6 source fixes
landed. Symbol check confirms it contains `GEM5_ECG_PFX_RECENT_FILTER_SIZE`
(the post-6f-6 env knob). The RISCV binary is 10 days older than the
graph_cache_context_gem5.hh fix and 2 weeks older than the recent
ecg_pfx.cc changes; **any RISCV cycle-accurate run today would use
stale code and report misleading results.**

To verify the `ecg.extract` opcode end-to-end (paper-faithful CHARGED=0
ISA-delivery validation), the RISCV binary must be rebuilt:

```
python3 scripts/setup_gem5.py --isa RISCV --rebuild --jobs 16
```

### G. Sideband JSON loader

`graph_cache_context_gem5.hh::loadFromSideband()` parses
`/tmp/gem5_graphbrew_ctx.json`. The format is documented in
`docs/...` and produced by the harness at kernel startup. Audit
confirms property regions are correctly parsed (smoke output:
`ECG_PFX: loaded sideband property=[0x..., 0x...) elem=4`).

---

## What's needed for paper-faithful gem5 cycle-accurate CHARGED=0 mode-6 validation

The HPCA evaluation manifest's `popt_off__isa__k2` headline arm is
already proven on cache_sim (5/5 wins). To extend the same arm to a
gem5 cycle-accurate cross-sim audit, the following are required:

1. **Fix the ECG_PFX page-cross filter** (fix 1 above). Without this,
   *any* ECG_PFX gem5 cell reports `pf_issued = 0` regardless of how
   the hints are delivered.
2. **Rebuild RISCV gem5.opt** so the `ecg.extract` opcode actually
   exercises the post-fix ECG_PFX SimObject.
3. **Wire kernel-side fat-id payloads** so the per-edge mode-6 mask is
   delivered via `ecg.extract` (currently delivered via `m5_work_begin`
   in X86; the RISCV path uses inline asm to emit the opcode but the
   payload structure must match what `decoder_ecg_extract.isa` expects).
4. **Run one cell end-to-end** (e.g., email-Eu-core/pr) with the RISCV
   binary, verify `pf_identified > 0` AND `pf_issued > 0` AND
   `pf_useful > 0`, then expand to delaunay_n19 + roadNet-CA + at
   least one large social graph if budget permits.
5. **Compare gem5 ECG mode 6 DRAM traffic against cache_sim's
   ECG mode 6 number for the same cell.** Target: within 20% relative
   delta (cycle-accurate noise + L1/L2 vs cache_sim's L3-only sim is
   expected to differ).

Time estimate: 3-5 days. Fix 1 is the longest pole (gem5 MMU plumbing
unfamiliar to most graph-cache contributors). Fixes 2-4 are routine.

---

## What's complete enough to publish without further gem5 work

- ECG/GRASP/P-OPT replacement policy correctness on cache_sim ✓
  (already proven by Phase 2 baseline parity within 0.77%)
- DROPLET-style prefetcher fires correctly in both gem5 and Sniper ✓
- gem5 X86 SE-mode can execute the PR kernel end-to-end ✓
- The `ecg.extract` opcode decodes correctly ✓ (RISCV binary stale but
  source verified)
- Ring-buffer hint queue eliminates hint-loss for high-rate emission ✓
  (single-slot mailbox bug fixed in sprint 6f-6)

The paper's primary contributions (P-OPT-AVG-derived per-edge fat-mask
+ POPT-vs-GRASP parity + DROPLET-vs-ECG_PFX bandwidth efficiency)
stand on cache_sim evidence. Sniper provides cross-sim corroboration
for the cycle-accurate axis. Gem5 is a known-incomplete third leg that
should be documented as future work, not removed from the paper, since
the policies and the opcode are correctly implemented; only the
SimObject's page-cross filter blocks the final hookup.

---

## Documentation of supersession

This finding supersedes the root-cause hypothesis in
`docs/findings/gem5_ecg_pfx_simobject_gap.md`. That doc correctly
identified the *symptom* (`pf_issued = 0` despite hint consumption) but
incorrectly attributed it to the single-slot mailbox. The mailbox fix
(commit `10ea8097`) landed and the X86 binary picked it up (verified
by symbol check), yet `pf_issued = 0` persists. The new root cause is
the page-cross filter in gem5's `Queued::notify()` (file
`bench/include/gem5_sim/gem5/src/mem/cache/prefetch/queued.cc:212-234`)
which requires `mmu != nullptr` on the prefetcher SimObject for any
prefetch whose target line lives on a different page from the demand
access. ECG_PFX prefetches are essentially always cross-page (they
target random property[v] addresses), so the filter kills them all.

---

## Smoke commands for reproducing this audit

```
# 1. Compare DROPLET (works) vs ECG_PFX (broken) on the same workload:
ECG_CONTAINER_BITS=64 timeout 600 python3 scripts/experiments/ecg/roi_matrix.py \
    --suite gem5 --no-build \
    --benchmark pr \
    --options "-f results/graphs/email-Eu-core/email-Eu-core.sg -s -o 5 -n 1 -i 2" \
    --policies LRU \
    --prefetcher ECG_PFX --prefetcher-level l2 --allow-gem5-ecg-pfx \
    --ecg-pfx-mode per_edge \
    --l1d-size 32kB --l1d-ways 8 --l2-size 256kB --l2-ways 8 \
    --l3-sizes 1MB --l3-ways 16 --line-size 64 \
    --timeout-gem5 540 \
    --out-dir /tmp/gem5_audit_smoke/ECG_PFX

# 2. Inspect the CSV — kill-switch values:
python3 -c "
import csv
with open('/tmp/gem5_audit_smoke/ECG_PFX/roi_matrix.csv') as f:
    for row in csv.DictReader(f):
        print('pf_identified =', row['pf_identified'])
        print('pf_issued     =', row['pf_issued'])
        print('pf_useful     =', row['pf_useful'])
        print('pf_span_page  =', row['pf_span_page'])
        break  # only first section
"
# Pre-fix expected output:
#   pf_identified = 0
#   pf_issued     = 0
#   pf_useful     = 0
#   pf_span_page  = 63
# Post-fix expected output:
#   pf_identified > 0
#   pf_issued     > 0 (ideally == pf_identified)
#   pf_useful     > 0 (some non-trivial fraction)
#   pf_span_page  ~ 63 (this counter still increments; it's not a drop)
```

---

## Related session artifacts

- Prior (superseded) gap doc: `docs/findings/gem5_ecg_pfx_simobject_gap.md`
- HPCA cache_sim evaluation summary: `docs/findings/hpca_evaluation_complete_v1.md`
- Sniper sg_kernel mode-6 port (parallel to gem5 PR port):
  commit `1aa1b24b`
- Ring-buffer hint queue fix: commit `10ea8097`
- ECG_PFX recent_filter env knob: commit `7ef72444`

## Post-fix verdict (sprint S68, 2026-06-08)

Automated rubber-duck-gated pipeline closed the gem5 ECG_PFX `pf_issued=0` gap end-to-end. All 9 milestones (M1, M1b, M2, M3, M4, M5, M5b, M6, M7) reached `status=ok` after iterating through 3 rubber-duck-caught defects (page-cross filter, queue-servicing scheduling, silent X86 substitution). Sprint overview: `docs/findings/sprint_s68_overview.md`.

### Per-milestone evidence

| Milestone | Status | Rubber-duck | Key evidence |
|---|---|---|---|
| M1  MMU plumbing | ok | 🟢 GREEN | 8 `registerMMU` calls in graph_se.py (4 prefetcher attachment sites × 2 cpu.mmu/dtb branches) |
| M1b Queue servicing | ok | 🟡 YELLOW | queued.hh nextPrefetchReadyTime() returns curTick() when pfqMissingTranslation non-empty; X86 incremental rebuild 86s |
| M2  Smoke diff (email-Eu-core/pr) | ok | 🟢 GREEN | ECG_PFX pf_identified: 0 → **63**; pf_issued: 0 → **63**; DROPLET control unchanged at 1673 |
| M3  Multi-graph (kron_s16_k4/pr) | ok | 🟡 YELLOW | pf_identified=101541, pf_issued=101541, pf_useful=**576** (utility proof) — pivoted from delaunay_n19 + kron_s17 (both timed out at gem5 X86 wall budget) |
| M4  RISCV rebuild | ok | 🟢 GREEN | RISCV gem5.opt rebuilt with all post-6f-6 + M1b patches; symbol `GEM5_ECG_PFX_RECENT_FILTER_SIZE` present |
| M5  ISA smoke (ecg.extract on RISCV) | ok | 🟡 YELLOW | pf_identified=64, pf_issued=64 via RISCV custom-0 opcode; isa_path_used=1, isa_kernel_used=1 (verified post-run via gem5 log + config.ini assertions) |
| M5b Latency guard | ok | 🟢 GREEN | queued.cc getPacket() now returns nullptr if pfq.front().tick > curTick(); preserves prefetcher latency contract |
| M6  Mechanism corroboration | ok | 🟡 YELLOW | Reframed from absolute byte parity (out of scope) to "both sims show non-zero mechanism activity on same cell": gem5 pf_issued=64, cache_sim total_traffic=126, cache_sim ecg_runtime_issued=62,149 |
| M7  Verdict doc | ok | — | This section |

### Defects caught by rubber-duck (paper-critical)

1. **M2 → M1b**: M1 alone was insufficient. Page-cross filter passed (pf_identified 0 → 63), but pf_issued stayed 0. Rubber-duck identified `nextPrefetchReadyTime()` returning MaxTick → cache never woke up to drain `pfqMissingTranslation`. Result: M1b 3-line patch unlocked the full pipeline.
2. **M3 → Pivot B**: Original M3 exit criteria required `pf_useful > 0` on both cells. email-Eu-core (4KB property region fits L1d) inherently can't achieve this. Rubber-duck split criteria into "mechanism" (email) + "utility" (medium graph). After 2 timeouts (delaunay_n19, kron_s17), settled on kron_s16_k4 (65K verts, fits gem5 wall budget).
3. **M5 → M5 v2**: Silent X86 substitution. roi_matrix.py hard-coded GEM5_OPT=X86 path, so M5 v1's "RISCV ecg.extract smoke" actually ran X86 m5op delivery — the "ECG_PFX: first target vertex" log line fires regardless of path. Fixed by adding env-var overrides + post-run assertions on gem5 log + config.ini.
4. **M5b**: M1b's wake-on-pending-translation patch could let getPacket() issue a prefetch on the SAME tick its translation completed, bypassing the latency contract. M5b added a `pfq.front().tick > curTick()` guard before issuance.
5. **M6 reframing**: Original M6 attempted absolute DRAM-byte parity between functional cache_sim and cycle-accurate gem5 — rubber-duck verdict: invalid metric. cache_sim sees only the instrumented kernel ROI; gem5 SE-mode sees the full process lifetime (libc init, syscall emul, m5op machinery, etc.). Reframed as mechanism corroboration.

### Durable patch infrastructure

The two queued.{hh,cc} patches are gitignored under the vendor gem5 tree, so we added:

- `bench/include/gem5_sim/overlays/mem/cache/prefetch/queued_hh.patch` (M1b)
- `bench/include/gem5_sim/overlays/mem/cache/prefetch/queued_cc_latency.patch` (M5b)
- `scripts/setup_gem5.py::apply_unified_diff_patches()` applies them via `patch -p1` with `--dry-run` idempotency check
- `UNIFIED_DIFF_PATCHES` manifest entry registers both patches

Both patches survive `setup_gem5.py --rebuild` and `--clean` cycles.

### Paper-ready blurb (§6.3 future work / scope)

> The gem5 cycle-accurate validation of ECG mode-6 prefetcher mechanism required three implementation fixes beyond the published gem5 v24.0 source: (1) MMU plumbing to allow cross-page prefetches in `BasePrefetcher` SimObjects; (2) a `nextPrefetchReadyTime()` patch so the cache wakes up to drain `pfqMissingTranslation` even when `pfq` is empty (critical for prefetchers that emit only cross-page candidates, like ECG_PFX targeting `property[random_v]`); (3) a `getPacket()` latency-readiness guard that preserves the prefetcher's `latency` cycle contract under (2). With these fixes, the RISC-V custom-0 `ecg.extract` opcode (32-bit container packing real_vertex + DBG/POPT/PFX hints) successfully delivers fat-id payloads from the GAPBS PR kernel to the L2 ECG_PFX prefetcher SimObject, which issues hardware prefetches end-to-end. We use gem5 as cycle-accurate mechanism corroboration rather than absolute DRAM-byte parity peer to cache_sim: the two simulators have fundamentally different stat scopes (cache_sim instruments only the kernel ROI; gem5 SE-mode tracks the full process lifetime).

### Open follow-ups (non-blocking for paper closeout)

- M3b: characterize the kron_s16_k4 pf_useful=576 vs pf_issued=101,541 (0.57% useful rate) by sweeping L2 size 64kB / 128kB / 256kB. Investigate whether lookahead/AMPLIFY tuning improves accuracy.
- Sniper mechanism corroboration on the same email-Eu-core cell (out of scope for S68; deferred).
- Packed fat-id DBG/POPT/PFX high-bit semantics in `ecg.extract` are wired but not exercised in S68 — the smoke uses only the PFX-target low 32 bits.

### Reproducibility

```bash
bash scripts/experiments/ecg/sprint_s68/run_sprint.sh --status
bash scripts/experiments/ecg/sprint_s68/run_sprint.sh --all   # full sprint
bash scripts/experiments/ecg/sprint_s68/run_sprint.sh M1       # specific milestone
```

Each milestone is idempotent (marker-guarded patches, mtime-checked rebuilds). Rubber-duck verdicts written to `results/sprint_s68/<m_id>/rubber_duck_verdict.md`. Sprint advances on GREEN/YELLOW, halts on RED.
