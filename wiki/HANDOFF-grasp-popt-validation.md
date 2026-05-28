# Handoff — GRASP / PIN / POPT faithfulness validation

Branch: `graphbrew_ecg`  •  Date: 2026-05-28

## What is already done

**Trace-replay parity (cache_sim layer):** 20/20 zero-delta vs upstream
`faldupriyank/grasp` for {LRU, PIN, GRASP, BELADY} × {BC, BellmanFordOpt,
PageRankOpt, PageRankDeltaOpt, Radii} on web-Google. See
`/tmp/graphbrew-upstream-policy-compare-all4/comparison.csv` (regenerable).

Recent commits on `graphbrew_ecg`:

- `e292903` Align cache_sim/gem5/Sniper with upstream GRASP semantics (f=50 default, `grasp_region` sideband flag, `EvictionPolicy::PIN`).
- `c1372e1` Fix POPT MSB polarity to match HPCA21 reference.
- `0d49c67` Add upstream GRASP trace-replay parity tooling (new `graphbrew_trace_replay.cc` + `upstream_policy_compare.py`).
- `3f8e81b` Extend validation gates and faithfulness pytest (11/11 green).
- `3e5b136` Wiki: GRASP/PIN/POPT faithfulness docs.

Faithfulness state recorded in: [/memories/repo/grasp_srrip_baseline.md](/memories/repo/grasp_srrip_baseline.md).

## What is NOT yet validated

The trace-replay parity only exercises `bench/include/cache_sim/cache_sim.h`.
The gem5 and Sniper integrations share the **same sideband** (registerGRASPTraceRegion,
`grasp_region` flag, `frontier_frac=50`) but use **different replacement-policy
implementations** living inside the simulator overlays. Those are not yet
proven faithful.

## Next session — three-tier validation plan

Run tiers in order; each builds confidence for the next.

### Tier A — Sideband registration sanity (10 min)

Goal: prove gem5 and Sniper actually receive 2 regions (propertyA, propertyB)
with `hot_pct=50` and `grasp_region=true` for a known DBG run.

1. Add a one-shot log line at region registration in:
   - `bench/include/gem5_sim/overlays/mem/cache/replacement_policies/graph_cache_context_gem5.hh`
   - `bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/graph_cache_context_sniper.cc`
   Format suggestion: `"[graphctx] register region base=0x%lx upper=0x%lx hot_pct=%u grasp_region=%d"`
2. Run PR on `email-Eu-core` (smallest graph) under both simulators with `-o 5` (DBG):
   ```bash
   make sim-pr && ./bench/bin_sim/pr -f results/graphs/email-Eu-core/email-Eu-core.sg -s -o 5 -n 1
   # then gem5 + sniper equivalents via roi_matrix.py --suite gem5/sniper
   ```
3. Assert log has **exactly 2** lines with `grasp_region=1, hot_pct=50`.
4. Lock in via pytest: `scripts/test/test_grasp_sideband_registration.py` (new).

### Tier B — POPT permutation equivalence (independent of simulators)

POPT is a reordering, not a replacement policy. Validation is fully
deterministic and doesn't need any simulator.

1. Build upstream POPT reference from `research/POPT_HPCA21_CameraReady`
   (header-only / source under `research/`).
2. Run on `web-Google.el` to produce `new_id[]` reference.
3. Run GraphBrew POPT (`bench/include/graphbrew/partition/cagra/popt.h`) on
   the same edge list.
4. Diff the permutation arrays — after the MSB-polarity fix (commit `c1372e1`)
   expect bit-exact equality or Kendall-τ ≈ 1.
5. Lock in via pytest with a tiny synthetic graph.

### Tier C — gem5 / Sniper GRASP-vs-LRU sign test

Bit-exact miss parity is impossible (MSHRs, prefetchers, replacement-state
init differ). Instead, validate **sign consistency** of GRASP-vs-LRU deltas
between cache_sim and the full simulators.

1. Use existing `scripts/experiments/ecg/roi_matrix.py --suite gem5` and
   `--suite sniper` on small graphs (`email-Eu-core`, `cit-Patents`) for
   PR and BC, L3 sizes {4kB, 32kB, 256kB, 2MB}.
2. Compare per-(graph, app, L3-size) GRASP-vs-LRU delta sign against the
   matching cache_sim row in `/tmp/graphbrew-grasp-cache-sweep/*/DBG/roi_matrix.csv`.
3. Disagreement at small L3 (4kB/32kB) = real bug to investigate.
   Disagreement only at 2MB is acceptable (working set fits).
4. Document deltas in `wiki/POPT-GRASP-Faithfulness-Audit.md`.

## Useful one-liners

```bash
# Regenerate trace-replay parity
rm -rf /tmp/graphbrew-upstream-policy-compare-all4 && \
python3 scripts/experiments/ecg/upstream_policy_compare.py \
  --policies lru pin grasp belady \
  --traces BC.web-Google.cvgr.dbg.lru.llc.trace \
           BellmanFordOpt.web-Google.cintgr.dbg.lru.llc.trace \
           PageRankOpt.web-Google.cvgr.dbg.lru.llc.trace \
           PageRankDeltaOpt.web-Google.cvgr.dbg.lru.llc.trace \
           Radii.web-Google.cvgr.dbg.lru.llc.trace \
  --out-dir /tmp/graphbrew-upstream-policy-compare-all4

# Faithfulness pytest
python3 -m pytest -q scripts/test/test_popt_grasp_faithfulness_sources.py \
                     scripts/test/test_ecg_validation_gates.py

# cache_sim sweep that produces the GRASP-vs-LRU reference for Tier C
python3 scripts/experiments/ecg/roi_matrix.py --suite cache-sim --benchmark pr \
  --options "-f results/graphs/email-Eu-core/email-Eu-core.sg -s -o 5 -n 1 -i 1" \
  --policies LRU SRRIP GRASP \
  --l1d-size 1kB --l2-size 2kB --l3-sizes 4kB 32kB 256kB 2MB --l3-ways 16 \
  --line-size 64 --out-dir /tmp/graphbrew-grasp-cache-sweep/email-pr/DBG
```

## Gotchas

- Upstream PIN/BELADY .bin sources live at `/tmp/graphbrew-faithfulness-upstream/grasp/trace-based-simulators/`
  — `upstream_policy_compare.py` builds them on demand. Trace files live in `datasets/`.
- Do **not** run multiple GraphBrew gem5 jobs on the same node unless sideband
  files are isolated (per `.github/copilot-instructions.md`).
- BellmanFordOpt uses `frontier_frac=100`, not 50 — confirm before asserting region params.
- BELADY in `graphbrew_trace_replay.cc` requires the uint64_t cast on `time=-1`
  so empty ways wrap to UINT64_MAX and get evicted first (regression-prone).
