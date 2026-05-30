# POLICY_COLORS perceptual distinguishability (gate 243)

**Status:** active  •  colors: 9  •  hatches: 2  •  pairs checked: 36  •  violations: 0

**Thresholds:** ΔE≥12.0, ΔL≥10.0 (or hatch), ΔE_white≥18.0

## Rules
- **C1** — every POLICY_LABELS key has a well-formed 7-char hex color
- **C2** — no two POLICY_COLORS values are exactly equal
- **C3** — every pair has CIE76 ΔE ≥ 12.0
- **C4** — pairs with lightness delta < 10.0 must use hatching (POLICY_HATCHES) for B&W printability, modulo the ACKNOWLEDGED_BW_PAIRS allowlist
- **C5** — every color has ΔE ≥ 18.0 from white
- **C6** — POLICY_HATCHES keys are a subset of POLICY_LABELS keys

## Palette

| policy_label | figure_label | color | L* | a* | b* | hatch |
|---|---|---|---:|---:|---:|---|
| ECG_DBG_ONLY | ECG-D | `#8CD17D` | 77.671 | -37.636 | 34.934 | — |
| ECG_DBG_PRIMARY | ECG-H | `#54A24B` | 60.098 | -41.699 | 37.594 | — |
| ECG_DBG_PRIMARY_CHARGED | ECG-H+C | `#B79A20` | 64.391 | -1.06 | 61.911 | /// |
| ECG_POPT_PRIMARY | ECG-P | `#B279A2` | 57.799 | 28.716 | -12.92 | — |
| GRASP | GRASP | `#4C78A8` | 49.247 | -0.842 | -30.254 | — |
| LRU | LRU | `#BDBDBD` | 76.611 | -0.0 | 0.0 | — |
| POPT | P-OPT | `#F58518` | 66.717 | 36.628 | 69.004 | — |
| POPT_CHARGED | P-OPT+C | `#F2B872` | 78.681 | 13.042 | 43.41 | /// |
| SRRIP | SRRIP | `#8E8E8E` | 59.02 | -0.0 | 0.0 | — |

## Pairwise distances (top closest pairs)

| a | b | ΔE | ΔL |
|---|---|---:|---:|
| LRU | SRRIP | 17.591 | 17.591 |
| ECG_DBG_ONLY | ECG_DBG_PRIMARY | 18.232 | 17.573 |
| ECG_DBG_PRIMARY_CHARGED | POPT_CHARGED | 27.301 | 14.29 |
| ECG_POPT_PRIMARY | SRRIP | 31.512 | 1.222 |
| GRASP | SRRIP | 31.804 | 9.774 |
| ECG_POPT_PRIMARY | GRASP | 35.316 | 8.552 |
| ECG_POPT_PRIMARY | LRU | 36.68 | 18.812 |
| POPT | POPT_CHARGED | 36.803 | 11.964 |
| ECG_DBG_PRIMARY_CHARGED | POPT | 38.421 | 2.327 |
| GRASP | LRU | 40.802 | 27.364 |

**0 violations** — palette is color- and greyscale-distinguishable.
