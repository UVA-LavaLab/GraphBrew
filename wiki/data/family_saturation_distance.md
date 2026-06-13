# Per-family saturation-distance replay

**Verdict:** PASS  
**Source:** `wiki/data/saturation_distance.json`  
**High-headroom families:** citation, social (floor 5.0 pp)  
**Pinned low-headroom families:** web (ceiling 5.0 pp)  
**Ordering slack:** 1.0 pp

## Per-family medians

| family | n_cells | min pp | median pp | p90 pp | max pp | graphs |
|---|---:|---:|---:|---:|---:|---|
| citation | 5 | +4.4125 | +14.1698 | +26.9590 | +26.9590 | cit-Patents |
| social | 15 | +0.4893 | +12.7801 | +23.5205 | +30.3964 | com-orkut, soc-LiveJournal1, soc-pokec |
| web | 5 | +0.0000 | +3.1518 | +28.7477 | +28.7477 | web-Google |

## Verdict checks

| check | result |
|---|---|
| all_family_medians_nonneg | ✅ |
| all_family_mins_nonneg | ✅ |
| high_headroom_families_meet_floor | ✅ |
| pinned_low_headroom_under_ceiling | ✅ |
| family_ordering_citation_social_web | ✅ |
| at_least_three_families_present | ✅ |

## Interpretation

Mirror of gate 67 for the saturation-distance metric. Hub-heavy families (citation, social) still have meaningful upper-octave headroom (median >= 5 pp 4MB->8MB drop); the single web graph (web-Google) is pinned as the low-headroom exemplar (median < 5 pp). Family ordering citation >= social >= web is locked within a 1 pp slack so small cell-count shifts don't break the gate but a real regime flip would.
