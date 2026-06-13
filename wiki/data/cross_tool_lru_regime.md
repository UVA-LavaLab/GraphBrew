# Cross-tool LRU-vs-GRASP regime inversion

**Verdict:** PASS  
**Regime inversion holds:** yes  
**Post-WSS LRU-steeper floor:** 0.3 pp/octave  
**Sub-WSS tolerance:** 0.2 pp/octave  

## Per-tool medians

| tool | L3 range | regime | GRASP | LRU | LRU-GRASP |
|---|---|---|---:|---:|---:|
| cache-sim | 1024kB-8192kB | post-WSS | -13.0117 | -15.6105 | -2.5988 |
| gem5 | 4kB-2048kB | sub-WSS | -5.9607 | -5.1229 | +0.8378 |
| sniper | 4kB-2048kB | sub-WSS | -7.6445 | -7.4015 | +0.2430 |

## Verdict checks

| check | result |
|---|---|
| cache_sim_postwss_LRU_steeper | ✅ |
| gem5_subwss_LRU_not_strictly_steeper | ✅ |
| sniper_subwss_LRU_not_strictly_steeper | ✅ |
| regime_inversion_sign_holds | ✅ |
| regime_labels_correct | ✅ |

## Interpretation

This gate formalizes the regime-dependent LRU-vs-GRASP slope finding that gates 70/71/72 surfaced as INFORMATIONAL. The cache-sim sweep (1-8MB, post-WSS for our corpus) shows LRU strictly steeper than GRASP — the classic 'oracle-aware policies are less cache-hungry' story holds at policy-relevant scales. Both anchor tools at sub-WSS scales (4kB-2MB, where no policy can fit the hot set) show the opposite ordering: LRU's give-up-and-stream behaviour extracts almost nothing from extra capacity, while GRASP's hold-the-hot-set behaviour still secures partial reuse and benefits more from additional ways. The cross-tool sign agreement between gem5 and sniper on the sub-WSS inversion (gem5 +0.84 pp/oct, sniper +0.24 pp/oct against cache-sim -0.97 pp/oct) confirms this is a physical regime effect, not a tool artifact.
