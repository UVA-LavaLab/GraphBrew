# Literature-faithfulness citation/date audit

Per_claim citation-field structural audit (gate 237 — LIT-CitDate).

## Rules
- **D1** — every per_claim row has non-empty citation
- **D2** — citation parses to at least one (author, venue, year) tuple
- **D3** — every parsed venue ∈ ['ASPLOS', 'HPCA', 'ISCA', 'MICRO', 'SC']
- **D4** — every parsed year ∈ [2005, 2026]
- **D5** — originator policy citations match the originating publication (GRASP→Faldu HPCA 2020, POPT[+derived]→Balaji HPCA 2021, SRRIP→Jaleel ISCA 2010), OR the citation is an explicit cross-attribution where the policy name appears in the citation string AND the parsed venue is in the whitelist
- **D6** — every citation contains a locator (§N | Fig N | Tab N | Table N | Section N)
- **D7** — corpus uses at least 10 distinct citation strings

## Constants
- venue whitelist: `ASPLOS, HPCA, ISCA, MICRO, SC`
- year range: `2005..2026`
- distinct-citation floor: `10`

## Totals
- per_claim rows: **330**
- rows parsed cleanly: **330**
- distinct citation strings: **15**
- venue tally: `{'HPCA': 429, 'ISCA': 75}`
- year tally: `{'2010': 75, '2020': 177, '2021': 252}`
- author tally: `{'Balaji': 252, 'Faldu': 177, 'Jaleel': 75}`

## Per policy

| policy | rows | parses ok | distinct citations |
| --- | ---: | ---: | ---: |
| GRASP | 19 | 19 | 5 |
| POPT | 8 | 8 | 3 |
| POPT_GE_GRASP | 114 | 114 | 2 |
| POPT_NEAR_GRASP_IF_BIG_GAP | 114 | 114 | 1 |
| SRRIP | 75 | 75 | 4 |

## Violations

_None._
