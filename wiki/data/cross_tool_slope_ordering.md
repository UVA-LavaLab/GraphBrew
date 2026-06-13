# Cross-tool SRRIP-vs-GRASP slope ordering

**Verdict:** PASS  
**Tools compared:** cache_sim, gem5, sniper  
**Gap floor (strict):** 0.05 pp/oct  
**Required strict tools:** 2

## Per-tool medians

| tool | GRASP | SRRIP | LRU | SRRIP-GRASP | LRU-GRASP (info) | SRRIP <= GRASP | strict |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| cache_sim | -13.0117 | -15.5996 | -15.6105 | -2.5879 | -2.5988 | ✅ | ✅ |
| gem5 | -5.9607 | -7.2113 | -5.1229 | -1.2506 | +0.8378 | ✅ | ✅ |
| sniper | -7.6445 | -7.9640 | -7.4015 | -0.3195 | +0.2430 | ✅ | ✅ |

## Verdict checks

| check | result |
|---|---|
| all_tools_present_and_valid | ✅ |
| all_tools_srrip_le_grasp | ✅ |
| enough_tools_strictly_steeper | ✅ |

## Interpretation

GRASP is oracle-aware; SRRIP is not. The claim under test is that SRRIP is at least as cache-hungry as GRASP — its miss-rate slope vs log2(L3) should fall at least as steeply, because at small caches SRRIP cannot anticipate the reuse-likely block identity and pays a larger penalty.

This gate confirms the claim is REPLICATED in all three tools of the GraphBrew pipeline (cache-sim sweep + gem5 anchor + sniper anchor). The LRU-vs-GRASP delta is reported per tool but explicitly NOT gated: gates 70/71 documented that sub-WSS anchor scales (4kB << email-Eu-core WSS ~4.5kB) can invert the LRU>GRASP ordering observed at 1-8MB scales — a regime-dependent physical effect, not a tool artifact.
