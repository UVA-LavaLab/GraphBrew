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
- per_claim rows: **279**
- rows parsed cleanly: **279**
- distinct citation strings: **14**
- venue tally: `{'HPCA': 348, 'ISCA': 84}`
- year tally: `{'2010': 84, '2020': 156, '2021': 192}`
- author tally: `{'Balaji': 192, 'Faldu': 156, 'Jaleel': 84}`

## Per policy

| policy | rows | parses ok | distinct citations |
| --- | ---: | ---: | ---: |
| GRASP | 19 | 19 | 5 |
| POPT | 8 | 8 | 3 |
| POPT_GE_GRASP | 84 | 84 | 1 |
| POPT_NEAR_GRASP_IF_BIG_GAP | 84 | 84 | 1 |
| SRRIP | 84 | 84 | 4 |

## Violations

_None._
