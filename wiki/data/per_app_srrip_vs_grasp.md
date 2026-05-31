# Per-app SRRIP-vs-GRASP slope ordering

**Verdict:** PASS  
**Source:** `per_app_capacity_slope.json`  
**Apps:** 5  
**Pinned deviating:** bfs  
**Allowed SRRIP-shallower-than-GRASP slack:** 1.0 pp/octave

## Per-app medians (pp/octave)

| app | GRASP median | SRRIP median | SRRIP-GRASP | deviates |
|---|---:|---:|---:|:---:|
| bc | -14.2610 | -14.4060 | -0.1450 | ✅ |
| bfs (pinned) | -6.4078 | -3.6533 | +2.7545 | 📌 |
| cc | -14.0757 | -14.7883 | -0.7126 | ✅ |
| pr | -16.1912 | -16.8056 | -0.6144 | ✅ |
| sssp | -19.2798 | -19.5955 | -0.3157 | ✅ |

## Verdict checks

| check | result |
|---|---|
| no_missing_apps | ✅ |
| no_new_deviating_apps | ✅ |
| every_app_has_both_grasp_and_srrip | ✅ |

## Interpretation

Gate 72 verified the SRRIP-vs-GRASP slope ordering holds at the GLOBAL median across all three tools. This gate ensures the ordering also holds per app on the cache-sim sweep — modulo documented kernel deviations. bfs is pinned because its frontier-driven access pattern produces near-flat miss curves (gate 65 flagged it as the most-saturated kernel), so both LRU (gate 68 pin) and SRRIP (this gate's pin) appear shallower than GRASP on bfs. This is a real corpus property of the kernel, not a measurement artefact.
