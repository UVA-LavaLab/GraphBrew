# Checklists

---

## A) Correctness & Safety

- [ ] Output mapping is a bijection: all vertex IDs appear exactly once.
- [ ] No negative/overflow IDs; correct type width.
- [ ] Community-level permutations compose correctly into global permutation.
- [ ] Cross-community edges remain valid after relabeling.
- [ ] OOD fallback triggers are logged and deterministic.
- [ ] Margin fallback triggers are logged and deterministic.
- [ ] Seeds: Leiden and any RNG are controllable.
- [ ] Baseline correctness: BFS/CC/SSSP/BC/PR outputs match ORIGINAL.

## B) Performance Metrics

- [ ] Kernel time per workload (pre/post reorder)
- [ ] End-to-end time including reorder
- [ ] Breakdown timers for:
      Leiden, features, scoring, reorder build, relabel apply
- [ ] Hardware counters: LLC misses/MPKI, bandwidth, TLB (if available)
- [ ] Community stats: count, size dist, modularity, cross-edge ratio

## C) Model Quality

- [ ] Selection accuracy vs oracle (per-community and whole-graph)
- [ ] Regret distribution (mean, p95, worst)
- [ ] Confusion matrix across algorithms
- [ ] Calibration: score margin vs realized gain

## D) Robustness

- [ ] Unseen graph families
- [ ] Scale sweep
- [ ] Hardware sweep (if possible)
- [ ] Cross-workload stability checks

## E) Engineering Hygiene

- [ ] Logging includes: graph→type, community→chosen algo, margin, OOD distance
- [ ] Weights + centroids versioned by run_id + git commit
- [ ] "Skip reorder" early-exit for tiny graphs or high predicted cost

---

## F) Iteration Checklist (per code change)

Use this for every code change:

- [ ] Code change made to target file(s)
- [ ] `make clean && make -j$(nproc)` — **PASSES**
- [ ] Smoke test (4 variants × soc-Epinions1) — **ALL PASS**
- [ ] Test tier (3 graphs, 2 trials) — **no crashes, results generated**
- [ ] Medium tier (28 graphs, 5 trials) — **pass/fail criteria met**
- [ ] If FAIL → root cause identified → fix applied → **restart from top**
- [ ] If PASS → Large tier (37 graphs, 5 trials) — **final validation**
- [ ] Document improvement in commit message
