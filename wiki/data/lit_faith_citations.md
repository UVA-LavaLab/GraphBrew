# Literature-faithfulness citation locator audit

Bijection + grounding check between the live lit-faith corpus (`literature_faithfulness_postfix.json`) and the static source-of-truth (`literature_baselines.py`).

## Summary

- Unique citations in lit-faith: **15**
- Unique citations in baselines: **15**
- Intersection: **15** (faith-only: 0, baselines-only: 0)
- Baseline claims total: **38**; lit-faith per-cell claims: **330**
- Anchor papers expected: **3**; present in baselines: **3**; present in lit-faith: **3**; URL in docstring: **2**
- Citations well-formed (venue + year + locator + known anchor): **15** (ill-formed: 0)
- Baseline claims with short citation (< 20 chars): **0**

## Anchor papers

| key | venue | baseline claims | lit-faith claims | docstring URL/DOI |
|---|---|---:|---:|---|
| `faldu_hpca_2020` | HPCA 2020 | 22 | 177 | ✓ present |
| `balaji_hpca_2021` | HPCA 2021 | 16 | 252 | ✓ present |
| `jaleel_isca_2010` | ISCA 2010 | 5 | 75 | — not linked |

## Citation grounding

| citation | len | venue | year | locator | anchors | well_formed |
|---|---:|---|---|---|---|---|
| `Balaji & Lucia HPCA 2021 Fig 10` | 31 | ✓ | ✓ | ✓ | balaji_hpca_2021 | ✓ |
| `Balaji & Lucia HPCA 2021 Fig 10 (GRASP bar)` | 43 | ✓ | ✓ | ✓ | balaji_hpca_2021 | ✓ |
| `Balaji & Lucia HPCA 2021 Fig 9` | 30 | ✓ | ✓ | ✓ | balaji_hpca_2021 | ✓ |
| `Balaji & Lucia HPCA 2021 §6 (extrapolated to com-orkut from twitter)` | 68 | ✓ | ✓ | ✓ | balaji_hpca_2021 | ✓ |
| `Balaji & Lucia HPCA 2021 §6.3` | 29 | ✓ | ✓ | ✓ | balaji_hpca_2021 | ✓ |
| `Balaji & Lucia HPCA 2021 §6.3 (extended)` | 40 | ✓ | ✓ | ✓ | balaji_hpca_2021 | ✓ |
| `Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check` | 51 | ✓ | ✓ | ✓ | faldu_hpca_2020,balaji_hpca_2021 | ✓ |
| `Faldu et al. HPCA 2020 Fig 10` | 29 | ✓ | ✓ | ✓ | faldu_hpca_2020 | ✓ |
| `Faldu et al. HPCA 2020 Fig 11` | 29 | ✓ | ✓ | ✓ | faldu_hpca_2020 | ✓ |
| `Faldu et al. HPCA 2020 §6.1 (extrapolated from twitter Fig 10)` | 62 | ✓ | ✓ | ✓ | faldu_hpca_2020 | ✓ |
| `Faldu et al. HPCA 2020 §6.1 (extrapolated to com-orkut from twitter Fig 10)` | 75 | ✓ | ✓ | ✓ | faldu_hpca_2020 | ✓ |
| `Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC)` | 70 | ✓ | ✓ | ✓ | jaleel_isca_2010 | ✓ |
| `Jaleel et al. ISCA 2010 §5.2; Balaji & Lucia HPCA 2021 §6.3 (extended)` | 70 | ✓ | ✓ | ✓ | balaji_hpca_2021,jaleel_isca_2010 | ✓ |
| `Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1` | 57 | ✓ | ✓ | ✓ | faldu_hpca_2020,jaleel_isca_2010 | ✓ |
| `Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 (extended)` | 68 | ✓ | ✓ | ✓ | faldu_hpca_2020,jaleel_isca_2010 | ✓ |
