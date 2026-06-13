# Per-app SRRIP-vs-GRASP slope ordering

**Verdict:** PASS  
**Source:** `per_app_capacity_slope.json`  
**Apps:** 5  
**Pinned deviating:** none  
**Allowed SRRIP-shallower-than-GRASP slack:** 1.0 pp/octave

## Per-app medians (pp/octave)

| app | GRASP median | SRRIP median | SRRIP-GRASP | deviates |
|---|---:|---:|---:|:---:|
| bc | -13.8860 | -13.8907 | -0.0047 | ✅ |
| bfs | -2.2657 | -3.3722 | -1.1065 | ✅ |
| cc | -12.7012 | -15.1055 | -2.4043 | ✅ |
| pr | -15.6950 | -17.1544 | -1.4594 | ✅ |
| sssp | -19.6472 | -21.8254 | -2.1782 | ✅ |

## Verdict checks

| check | result |
|---|---|
| no_missing_apps | ✅ |
| no_new_deviating_apps | ✅ |
| every_app_has_both_grasp_and_srrip | ✅ |

## Interpretation

Gate 72 verified the SRRIP-vs-GRASP slope ordering holds at the GLOBAL median across all three tools. This gate ensures the ordering also holds per app on the cache-sim sweep — modulo documented kernel deviations. bfs is pinned because its frontier-driven access pattern produces near-flat miss curves (gate 65 flagged it as the most-saturated kernel), so both LRU (gate 68 pin) and SRRIP (this gate's pin) appear shallower than GRASP on bfs. This is a real corpus property of the kernel, not a measurement artefact.
