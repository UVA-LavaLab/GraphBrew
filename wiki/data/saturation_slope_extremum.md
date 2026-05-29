# Per-app saturation-vs-slope extremum corroboration

**Verdict:** PASS  
**Distance source:** `saturation_distance.json`  
**Slope source:** `per_app_capacity_slope.json`  
**Least cache-sensitive by distance:** `bfs`  
**Least cache-sensitive by slope:** `bfs`  
**Most cache-hungry by distance:** `bc` (INFO)  
**Most cache-hungry by slope:** `sssp` (INFO)  

## Per-app rankings

| app | distance pp | dist rank | slope pp/oct | slope rank |
|---|---:|:---:|---:|:---:|
| bc | +15.5517 | 5 | -14.2775 | 2 |
| bfs | +4.6274 | 1 | -5.2283 | 1 |
| cc | +12.1229 | 3 | -15.1524 | 3 |
| pr | +7.8327 | 2 | -16.5349 | 4 |
| sssp | +12.9274 | 4 | -19.4709 | 5 |

## Verdict checks

| check | result |
|---|---|
| bfs_is_argmin_distance | ✅ |
| bfs_is_shallowest_slope | ✅ |
| bfs_unique_extremum_on_both_metrics | ✅ |
| corpus_has_slope_steeper_than_3x_bfs | ✅ |
| corpus_has_distance_larger_than_2_5x_bfs | ✅ |

## INFORMATIONAL note

INFORMATIONAL: the most cache-hungry app DISAGREES across metrics. By distance (4MB->8MB upper-octave drop), the most cache-hungry app is 'bc'. By slope (OLS over the 1MB-8MB sweep), the most cache-hungry app is 'sssp'. This is not a fault — it is the regime-vs-aggregate distinction at play: distance captures upper-octave residual headroom while slope averages per-octave drop across the whole sweep. Apps with convex miss curves rank differently from apps with concave miss curves. This gate explicitly does NOT enforce agreement on the most-hungry extremum — only on the least-hungry extremum (bfs).
