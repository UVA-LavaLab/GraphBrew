# Gate 258 — graph-name canonical map

Status: **active**

## Totals

- n_canonical: 26
- n_families: 8
- n_literal_sites: 588
- n_distinct_literals: 26
- n_family_dicts: 8
- n_eval_graphs: 6

## Rules

- **R1** — every harvested graph literal is in CANONICAL_GRAPHS
- **R2** — every canonical has non-empty family + paper_label
- **R3** — every canonical has source ∈ {SNAP,GAP,DIMACS,KONECT,WebGraph,synthetic,test}
- **R4** — every family-classifier dict has canonical keys with matching families
- **R5** — every harvested literal has a non-family-dict site (unless documented_future=True)
- **R6** — canonical names match ^[A-Za-z][A-Za-z0-9_-]*$
- **R7** — every EVAL_GRAPHS entry in config.py is canonical
- **R8** — every canonical family matches ^[a-z][a-z0-9_]*$

## Canonical graphs

| name | family | source | paper_label |
|---|---|---|---|
| `soc-pokec` | `social` | `SNAP` | soc-pokec |
| `soc-LiveJournal1` | `social` | `SNAP` | soc-LiveJournal1 |
| `com-orkut` | `social` | `SNAP` | com-orkut |
| `com-orkut-undir` | `social` | `SNAP` | com-orkut-undir |
| `email-Eu-core` | `social` | `SNAP` | email-Eu-core |
| `cit-Patents` | `citation` | `SNAP` | cit-Patents |
| `web-Google` | `web` | `SNAP` | web-Google |
| `web-BerkStan` | `web` | `SNAP` | web-BerkStan |
| `p2p-Gnutella31` | `p2p` | `SNAP` | p2p-Gnutella31 |
| `roadNet-CA` | `road` | `SNAP` | roadNet-CA |
| `roadNet-PA` | `road` | `SNAP` | roadNet-PA |
| `roadNet-TX` | `road` | `SNAP` | roadNet-TX |
| `road-CA` | `road` | `SNAP` | road-CA |
| `USA-Road` | `road` | `DIMACS` | USA-Road |
| `wikipedia_link_en` | `content` | `KONECT` | wikipedia_link_en |
| `delaunay_n18` | `mesh` | `synthetic` | delaunay_n18 |
| `delaunay_n19` | `mesh` | `synthetic` | delaunay_n19 |
| `delaunay_n20` | `mesh` | `synthetic` | delaunay_n20 |
| `kron21` | `kronecker` | `synthetic` | kron21 |
| `kron22` | `kronecker` | `synthetic` | kron22 |
| `kron23` | `kronecker` | `synthetic` | kron23 |
| `soc-LJ` | `social` | `SNAP` | soc-LJ |
| `twitter7` | `social` | `WebGraph` | twitter7 |
| `web-uk` | `web` | `WebGraph` | web-uk |
| `twitter-2010` | `social` | `WebGraph` | twitter-2010 |
| `uk-2005` | `web` | `WebGraph` | uk-2005 |

## Harvested tokens (in-tree literals)

- `USA-Road`
- `cit-Patents`
- `com-orkut`
- `com-orkut-undir`
- `delaunay_n18`
- `delaunay_n19`
- `delaunay_n20`
- `email-Eu-core`
- `kron21`
- `kron22`
- `kron23`
- `p2p-Gnutella31`
- `road-CA`
- `roadNet-CA`
- `roadNet-PA`
- `roadNet-TX`
- `soc-LJ`
- `soc-LiveJournal1`
- `soc-pokec`
- `twitter-2010`
- `twitter7`
- `uk-2005`
- `web-BerkStan`
- `web-Google`
- `web-uk`
- `wikipedia_link_en`

## Family-classifier dicts

- `scripts/experiments/ecg/lit_faith_margin.py` :: `GRAPH_FAMILY` (12 entries)
- `scripts/experiments/ecg/oracle_gap_report.py` :: `GRAPH_FAMILY` (8 entries)
- `scripts/experiments/ecg/winning_regime_taxonomy.py` :: `GRAPH_FAMILY` (8 entries)
- `scripts/test/test_corpus_diversity_floor.py` :: `GRAPH_FAMILY` (11 entries)
- `scripts/test/test_oracle_gap_derivation_parity.py` :: `GRAPH_FAMILY` (8 entries)
- `scripts/test/test_policy_winner_table_derivation_parity.py` :: `GRAPH_FAMILY` (11 entries)
- `scripts/test/test_popt_vs_grasp_delta_derivation_parity.py` :: `GRAPH_FAMILY` (8 entries)
- `scripts/test/test_winning_regime_taxonomy_derivation_parity.py` :: `GRAPH_FAMILY` (8 entries)

## Violations

None.
