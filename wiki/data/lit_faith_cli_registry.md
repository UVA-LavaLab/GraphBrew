# Gate 260 — GAPBS kernel CLI vocabulary registry

Status: **active**

## Totals

- n_cl_classes: 6
- n_kernels: 11
- n_distinct_flags: 28
- n_kernel_source_checks: 27
- n_kernel_source_ok: 27

## Rules

- **R1** — every declared CL class has a non-empty getopt extension matching the live header
- **R2** — every canonical kernel source instantiates the canonical CL class
- **R3** — every canonical kernel's flag-set is the union of its inheritance chain
- **R4** — no within-chain getopt-letter conflict (same letter at two levels of the same chain)
- **R5** — every canonical flag matches ^[A-Za-z]$
- **R6** — every canonical flag has a documented FLAG_PURPOSE entry
- **R7** — every flag's arity (takes value / no value) is consistent across all classes that declare it

## CL classes

| name | parent | getopt_ext | chain | full_flags |
|---|---|---|---|---|
| `CLBase` | `None` | `f:g:hk:su:m:o:zj:SlD:` | CLBase | `D`, `S`, `f`, `g`, `h`, `j`, `k`, `l`, `m`, `o`, `s`, `u`, `z` |
| `CLApp` | `CLBase` | `an:r:v` | CLApp → CLBase | `D`, `S`, `a`, `f`, `g`, `h`, `j`, `k`, `l`, `m`, `n`, `o`, `r`, `s`, `u`, `v`, `z` |
| `CLIterApp` | `CLApp` | `i:` | CLIterApp → CLApp → CLBase | `D`, `S`, `a`, `f`, `g`, `h`, `i`, `j`, `k`, `l`, `m`, `n`, `o`, `r`, `s`, `u`, `v`, `z` |
| `CLPageRank` | `CLApp` | `i:t:` | CLPageRank → CLApp → CLBase | `D`, `S`, `a`, `f`, `g`, `h`, `i`, `j`, `k`, `l`, `m`, `n`, `o`, `r`, `s`, `t`, `u`, `v`, `z` |
| `CLDelta` | `CLApp` | `d:` | CLDelta → CLApp → CLBase | `D`, `S`, `a`, `d`, `f`, `g`, `h`, `j`, `k`, `l`, `m`, `n`, `o`, `r`, `s`, `u`, `v`, `z` |
| `CLConvert` | `CLBase` | `e:b:x:q:p:y:V:w` | CLConvert → CLBase | `D`, `S`, `V`, `b`, `e`, `f`, `g`, `h`, `j`, `k`, `l`, `m`, `o`, `p`, `q`, `s`, `u`, `w`, `x`, `y`, `z` |

## Kernels → CL class

| kernel | cl_class | n_flags | full_flags |
|---|---|---:|---|
| `bc` | `CLIterApp` | 18 | `D`, `S`, `a`, `f`, `g`, `h`, `i`, `j`, `k`, `l`, `m`, `n`, `o`, `r`, `s`, `u`, `v`, `z` |
| `bfs` | `CLApp` | 17 | `D`, `S`, `a`, `f`, `g`, `h`, `j`, `k`, `l`, `m`, `n`, `o`, `r`, `s`, `u`, `v`, `z` |
| `cc` | `CLApp` | 17 | `D`, `S`, `a`, `f`, `g`, `h`, `j`, `k`, `l`, `m`, `n`, `o`, `r`, `s`, `u`, `v`, `z` |
| `cc_sv` | `CLApp` | 17 | `D`, `S`, `a`, `f`, `g`, `h`, `j`, `k`, `l`, `m`, `n`, `o`, `r`, `s`, `u`, `v`, `z` |
| `converter` | `CLConvert` | 21 | `D`, `S`, `V`, `b`, `e`, `f`, `g`, `h`, `j`, `k`, `l`, `m`, `o`, `p`, `q`, `s`, `u`, `w`, `x`, `y`, `z` |
| `ecg_preprocess` | `CLApp` | 17 | `D`, `S`, `a`, `f`, `g`, `h`, `j`, `k`, `l`, `m`, `n`, `o`, `r`, `s`, `u`, `v`, `z` |
| `pr` | `CLPageRank` | 19 | `D`, `S`, `a`, `f`, `g`, `h`, `i`, `j`, `k`, `l`, `m`, `n`, `o`, `r`, `s`, `t`, `u`, `v`, `z` |
| `pr_spmv` | `CLPageRank` | 19 | `D`, `S`, `a`, `f`, `g`, `h`, `i`, `j`, `k`, `l`, `m`, `n`, `o`, `r`, `s`, `t`, `u`, `v`, `z` |
| `sssp` | `CLDelta` | 18 | `D`, `S`, `a`, `d`, `f`, `g`, `h`, `j`, `k`, `l`, `m`, `n`, `o`, `r`, `s`, `u`, `v`, `z` |
| `tc` | `CLApp` | 17 | `D`, `S`, `a`, `f`, `g`, `h`, `j`, `k`, `l`, `m`, `n`, `o`, `r`, `s`, `u`, `v`, `z` |
| `tc_p` | `CLApp` | 17 | `D`, `S`, `a`, `f`, `g`, `h`, `j`, `k`, `l`, `m`, `n`, `o`, `r`, `s`, `u`, `v`, `z` |

## Flag purpose

| flag | takes_value | purpose |
|---|---|---|
| `-D` | True | database directory for JSON output |
| `-S` | False | keep self loops |
| `-V` | True | output split CSR arrays (.out_degree/.out_neigh/.offset) |
| `-a` | False | output last-run analysis |
| `-b` | True | output serialized graph (.sg) |
| `-d` | True | delta-stepping parameter |
| `-e` | True | output edge list (.el) |
| `-f` | True | load graph from edge-list file |
| `-g` | True | generate 2^scale kronecker synthetic graph |
| `-h` | False | print help |
| `-i` | True | iteration count (bc) or max iters (pr) |
| `-j` | True | segmentation config (type:n:m) |
| `-k` | True | average degree for synthetic graph |
| `-l` | False | log per-trial performance |
| `-m` | True | in-place / memory-friendly loader |
| `-n` | True | number of trials |
| `-o` | True | apply reordering strategy (POPT / GRASP / etc.) |
| `-p` | True | output Matrix Market (.mtx) |
| `-q` | True | output reordered labels serialized (.lo) |
| `-r` | True | starting vertex (or 'rand') |
| `-s` | False | symmetrize input |
| `-t` | True | tolerance (pr / pr_spmv) |
| `-u` | True | generate 2^scale uniform-random synthetic graph |
| `-v` | False | verify output |
| `-w` | False | make output weighted (.wel/.wsg) |
| `-x` | True | output reordered labels (.so) |
| `-y` | True | output Ligra adjacency (.ligra) |
| `-z` | False | use indegree for degree-based orderings |

## Violations

None.
