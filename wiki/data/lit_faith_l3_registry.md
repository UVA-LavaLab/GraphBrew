# L3 cache-size registry (gate 251)

**Status:** active  •  canonical: 11  •  files: 54  •  PAPER_L3-shaped tuples: 43  •  L3_MB dicts: 9  •  L3_BYTES dicts: 7  •  tokens seen: 11  •  violations: 0

**Anchor triplet:** `1MB`, `4MB`, `8MB`

## Rules
- **L1** — every harvested L3 token is in canonical registry
- **L2** — every PAPER_L3 / PAPER_L3_SIZES tuple == ANCHOR_TRIPLET
- **L3** — every L3_MB dict pairs token with canonical MB
- **L4** — every L3_BYTES dict pairs token with canonical bytes
- **L5** — every canonical entry has valid role + sub_tier
- **L6** — every canonical anchor token appears in some harvested PAPER_L3 tuple
- **L7** — harvested PAPER_L3-shaped constants agree across files (no two files disagree)

## Canonical registry

| token | bytes | mb | role | sub_tier |
|---|---:|---:|---|---|
| `4kB` | 4,096 | 0.003906 | `probe` | `small_l3_probe` |
| `16kB` | 16,384 | 0.01562 | `sweep_low` | `l_curve_lowend` |
| `32kB` | 32,768 | 0.03125 | `probe` | `small_l3_probe` |
| `64kB` | 65,536 | 0.0625 | `sweep_low` | `l_curve_lowend` |
| `256kB` | 262,144 | 0.25 | `sweep_low` | `knee` |
| `1MB` | 1,048,576 | 1 | `anchor` | `paper_anchor` |
| `2MB` | 2,097,152 | 2 | `sweep_high` | `l_curve_highend` |
| `4MB` | 4,194,304 | 4 | `anchor` | `paper_anchor` |
| `8MB` | 8,388,608 | 8 | `anchor` | `paper_anchor` |
| `16MB` | 16,777,216 | 16 | `sweep_high` | `l_curve_highend` |
| `32MB` | 33,554,432 | 32 | `sweep_high` | `l_curve_highend` |

## Harvested constants (per file)

| file | PAPER_L3 tuples | L3_MB dicts | L3_BYTES dicts | subset tuples |
|---|---:|---:|---:|---:|
| `scripts/experiments/ecg/cache_saturation_onset.py` | 1 | 1 | 0 | 0 |
| `scripts/experiments/ecg/cache_sensitivity_slope.py` | 0 | 1 | 0 | 0 |
| `scripts/experiments/ecg/corpus_balance.py` | 1 | 0 | 0 | 0 |
| `scripts/experiments/ecg/cross_generator_gap_parity.py` | 1 | 0 | 0 | 0 |
| `scripts/experiments/ecg/distribution_diagnostics.py` | 1 | 0 | 0 | 0 |
| `scripts/experiments/ecg/family_curvature_replay.py` | 1 | 0 | 0 | 0 |
| `scripts/experiments/ecg/family_geomean_improvement.py` | 1 | 0 | 0 | 0 |
| `scripts/experiments/ecg/family_margin_replay.py` | 0 | 0 | 1 | 0 |
| `scripts/experiments/ecg/family_policy_auc_clustering.py` | 1 | 0 | 0 | 0 |
| `scripts/experiments/ecg/family_slope_replay.py` | 1 | 0 | 0 | 0 |
| `scripts/experiments/ecg/gap_distribution_shape.py` | 1 | 0 | 0 | 0 |
| `scripts/experiments/ecg/l3_policy_stability.py` | 1 | 0 | 0 | 0 |
| `scripts/experiments/ecg/lofo_robustness.py` | 1 | 0 | 0 | 0 |
| `scripts/experiments/ecg/oracle_gap_auc.py` | 1 | 1 | 0 | 0 |
| `scripts/experiments/ecg/oracle_gap_curvature.py` | 1 | 0 | 0 | 0 |
| `scripts/experiments/ecg/per_graph_app_stability.py` | 1 | 0 | 0 | 0 |
| `scripts/experiments/ecg/per_graph_cache_slope.py` | 1 | 1 | 0 | 0 |
| `scripts/experiments/ecg/policy_rank_kendall.py` | 1 | 0 | 0 | 0 |
| `scripts/experiments/ecg/sign_consistency.py` | 0 | 0 | 0 | 1 |
| `scripts/experiments/ecg/slope_saturation_xcheck.py` | 1 | 0 | 0 | 0 |
| `scripts/experiments/ecg/winner_margin_by_regime.py` | 0 | 0 | 1 | 0 |
| `scripts/experiments/ecg/winner_margin_gradient.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_cache_saturation_onset_arithmetic.py` | 0 | 1 | 0 | 0 |
| `scripts/test/test_cache_saturation_onset_derivation_parity.py` | 1 | 1 | 0 | 0 |
| `scripts/test/test_cache_sensitivity_slope_derivation_parity.py` | 0 | 1 | 0 | 0 |
| `scripts/test/test_corpus_balance_arithmetic.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_corpus_balance_derivation_parity.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_cross_generator_gap_parity_arithmetic.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_cross_generator_gap_parity_derivation_parity.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_cross_tool_winners_derivation_parity.py` | 0 | 0 | 1 | 0 |
| `scripts/test/test_distribution_diagnostics_derivation_parity.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_family_curvature_replay_arithmetic.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_family_curvature_replay_derivation_parity.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_family_geomean_improvement_derivation_parity.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_family_margin_replay_derivation_parity.py` | 0 | 0 | 1 | 0 |
| `scripts/test/test_family_policy_auc_clustering_derivation_parity.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_family_slope_replay_arithmetic.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_gap_distribution_shape_derivation_parity.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_l3_policy_stability.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_l3_policy_stability_derivation_parity.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_lofo_robustness_derivation_parity.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_oracle_gap_auc_derivation_parity.py` | 1 | 1 | 0 | 0 |
| `scripts/test/test_oracle_gap_curvature_derivation_parity.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_per_graph_app_stability_arithmetic.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_per_graph_app_stability_derivation_parity.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_per_graph_cache_slope_derivation_parity.py` | 1 | 1 | 0 | 0 |
| `scripts/test/test_policy_rank_kendall_derivation_parity.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_policy_winner_table_derivation_parity.py` | 0 | 0 | 1 | 0 |
| `scripts/test/test_popt_vs_grasp_delta_derivation_parity.py` | 0 | 0 | 1 | 0 |
| `scripts/test/test_slope_saturation_xcheck_arithmetic.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_slope_saturation_xcheck_derivation_parity.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_winner_margin_by_regime_derivation_parity.py` | 0 | 0 | 1 | 0 |
| `scripts/test/test_winner_margin_gradient_arithmetic.py` | 1 | 0 | 0 | 0 |
| `scripts/test/test_winner_margin_gradient_derivation_parity.py` | 1 | 0 | 0 | 0 |

**0 violations** — every harvested L3 cache-size constant agrees with the canonical registry on tokens, byte counts, MB scaling, and anchor ordering.
