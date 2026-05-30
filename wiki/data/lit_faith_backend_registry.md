# Gate 257 — backend/tool vocabulary registry

Status: **active**

## Totals

- n_canonical: 7
- n_families: 3
- n_literal_sites: 437
- n_distinct_literals: 7
- n_argparse_sites: 4

## Rules

- **R1** — every harvested backend/tool literal is in CANONICAL_BACKENDS
- **R2** — every canonical has non-empty family ∈ {cache_sim,gem5,sniper} + non-empty paper_label
- **R3** — no two canonicals share a paper_label unless they are declared punctuation variants
- **R4** — every canonical's punctuation_variants includes its own name
- **R5** — every --backend/--tool/--suite argparse choices+default ⊆ CANONICAL_BACKENDS (+'both')
- **R6** — canonical token names match ^[a-z][a-z0-9_-]*$
- **R7** — every canonical token is referenced by at least one in-tree literal

## Canonical backends

| name | family | paper_label | punctuation_variants |
|---|---|---|---|
| `cache_sim` | `cache_sim` | cache-sim | `cache_sim`, `cache-sim` |
| `cache-sim` | `cache_sim` | cache-sim | `cache_sim`, `cache-sim` |
| `gem5` | `gem5` | gem5 | `gem5` |
| `gem5-riscv` | `gem5` | gem5/RISC-V | `gem5-riscv` |
| `gem5-x86` | `gem5` | gem5/X86 | `gem5-x86` |
| `sniper` | `sniper` | Sniper | `sniper` |
| `sniper-sift` | `sniper` | Sniper/SIFT | `sniper-sift` |

## Harvested tokens (in-tree literals)

- `cache-sim`
- `cache_sim`
- `gem5`
- `gem5-riscv`
- `gem5-x86`
- `sniper`
- `sniper-sift`

## Violations

None.
