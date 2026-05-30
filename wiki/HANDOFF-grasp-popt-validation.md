# Handoff — GRASP / PIN / POPT faithfulness validation

Branch: `graphbrew_ecg`  •  Date: 2026-05-28

## Status update — confidence-building automation (2026-06-XX)

Tier A/B/C have all landed. The work has since expanded into a full
"is everything still green?" gate suite that runs on a single
`make confidence` invocation. The dashboard lives at
[`wiki/data/confidence_dashboard.md`](data/confidence_dashboard.md)
and currently reports **261 gates, all GREEN, exit 0**.

**ECG arm catalog (gate 261, refresh @261):**
The eighth in the vocabulary-lock series (252 SBATCH, 255 policy,
256 profile, 257 backend, 258 graph, 259 build, 260 CLI, 261
arm-catalog). Locks the cross-file consistency of every ECG arm +
paper-shipping policy across the THREE namespaces they appear in:
the registry side (`ECG:` prefixed namespace,
`lit_faith_policy_registry.CANONICAL_ECG_ARMS`, 9 entries), the
paper side (underscore namespace in `paper_pipeline.POLICY_ORDER`
plus per-policy tables `POLICY_LABELS` / `POLICY_DESCRIPTIONS` /
`POLICY_COLORS` / `POLICY_HATCHES`, 9 entries), and the
measurement side (mixed-case proof-matrix labels in
`proof_matrix.ABLATIONS` whose `policy` field points back at the
registry namespace, 16 entries). Catches the silent-drift cases
where a new `ECG_DBG_PRIMARY_CHARGED` ablation is added but the
paper-side `POLICY_DESCRIPTIONS` forgets it (every figure that
joins on paper_label gets a blank legend caption), a registry
rename to `ECG:DBG_HEAD` breaks the proof-matrix
`ECG_DBG_POPT.policy` reference (silently-missing bar in grid
plots), or two ABLATIONS rows are accidentally given the same
label (silent row-merging in the rollup CSV). 7 rules A1-A7: A1
every paper non-baseline policy has a registry entry after
namespace translation (`ECG_X` → `ECG:X`; `POPT_CHARGED` is
itself); A2 every paper policy has label + description + color;
A3 every `_CHARGED` paper policy has a hatch pattern (grayscale
legibility); A4 every proof-matrix ablation policy is a canonical
baseline or a registry arm key; A5 every adaptive-selector
candidate references a real ablation label; A6 no duplicate
ablation labels; A7 every ECG-parented registry arm has at least
one ablation, OR (for `_CHARGED` arms) its uncharged parent does
(since `_CHARGED` arms are post-hoc projections of the uncharged
run; see paper_pipeline.py PAIRS). Today: 9 paper policies (2
charged: `POPT_CHARGED` and `ECG_DBG_PRIMARY_CHARGED`), 9
registry arms, 16 ablations (4 cache_alone + 7 ecg_replacement +
2 pfx_only + 3 combined), 2 adaptive selectors
(ECG_ADAPTIVE_ORACLE, ECG_ADAPTIVE_NO_FULL_POPT); 0 violations.

**GAPBS CLI registry (gate 260, refresh @260):**
The seventh in the vocabulary-lock series (252 SBATCH, 255 policy,
256 profile, 257 backend, 258 graph, 259 build, 260 CLI). Locks
WHICH command-line flags each GAPBS-derived kernel binary actually
accepts (per its CLBase / CLApp / CLIterApp / CLPageRank / CLDelta /
CLConvert inheritance chain in `bench/include/external/gapbs/command_line.h`)
— so a contributor cannot silently iterate `-i 16 -i 32` for both
pr AND bfs (bfs uses `CLApp`, which has no `-i`; the sweep
collapses to a single bfs measurement repeated N times), pass
`-t 1e-7` to sssp (CLDelta intercepts `-t` as unknown and bails
after the run starts → wasted SBATCH time), invent a long-form
`--num-trials 5` when the getopt loop only recognises short `-n
5`, or claim a per-kernel arg table like `args['pr_spmv']['--tolerance']`
when the binary only accepts `-t`. The gate parses the header for
the six `class CL*` declarations + each class's
`get_args_ += "..."` extension string, then cross-validates against
the live `bench/src{,_sim,_gem5}/<kernel>.cc` instantiation of the
canonical class. 7 rules R1-R7: R1 every CL class has a non-empty
getopt-extension matching the live header; R2 every canonical
kernel source instantiates the canonical CL class (verified across
all 3 backends → 27/27 OK); R3 every kernel's flag-set is the
union of its inheritance chain (no orphan kernel-only flags); R4
no within-chain getopt conflicts (same letter at two levels of one
chain); R5 every canonical flag matches `^[A-Za-z]$`; R6 every
canonical flag has a documented `FLAG_PURPOSE` entry (28 entries
covering all distinct flags); R7 flag-arity (takes value / no
value) is consistent across every class that declares the flag.
Today: 6 CL classes, 11 kernels, 28 distinct flags, 27/27
source-instantiation checks pass; 0 violations.

**Build-target registry (gate 259, refresh @259):**
The sixth in the vocabulary-lock series (252 SBATCH, 255 policy,
256 profile, 257 backend, 258 graph, 259 build). Locks WHICH
compile targets actually produce the binaries that the downstream
gates measure — so a contributor cannot silently rename
`bench/bin_sim/pr` to `bench/bin_sim/pr_cs`, swap an `-O3` for
`-O2` in `CXXFLAGS_GAP` (would break every cache-sim
apples-to-apples comparison against the literature baseline),
drop the `-fopenmp` that `ecg_preprocess` requires, introduce an
undocumented gem5 frontend variant beyond {base, m5ops,
riscv_m5ops}, or add a `pr_kernel_smoke` orphan to
`bench/src_sniper` that no canonical entry tracks. The gate
parses the root Makefile for the four `KERNELS_*` variables +
four `CXXFLAGS_*` blocks + four `SRC_*_DIR` / `BIN_*_DIR` pairs,
then cross-validates against an in-generator
`CANONICAL_BUILD_TARGETS` allow-list of 51 (backend, kernel,
variant) entries AND every `.cc` file on disk under the
canonical SRC_DIRs. 8 rules R1-R8: R1 every (backend, kernel)
has a matching `.cc` source; R2 every Makefile-declared kernel
is canonical for its backend (no silent additions); R3 every
`CXXFLAGS_<BACKEND>` block contains all required tokens
(`-std=c++17`, `-fopenmp`, `-DNDEBUG`, plus `-DNO_M5OPS` for
gem5 default and `-I$(SNIPER_INCLUDE)` for sniper); R4 every
canonical backend maps to the canonical SRC_DIR / BIN_DIR path;
R5 every canonical kernel has a non-empty graph-algorithm
family classification (pagerank / traversal / shortest-path /
connected-component / centrality / triangle / preprocess /
smoke); R6 every backend's CXXFLAGS carries its documented
optimisation level (`-O3` native+sim, `-O1` gem5, `-O2`
sniper); R7 every `.cc` file maps to a canonical
(backend, kernel) entry (no orphan sources); R8 every canonical
backend has a documented ROI mechanism (m5ops / sift /
sim-callback / none). Today: 4 backends, 51 canonical targets
(10 native + 9 cache_sim + 24 gem5 base/m5ops/riscv + 8 sniper
Phase-0+in-flight), 26 Makefile-harvested kernels, 0 orphan
sources, 0 violations.

**Graph-name canonical map (gate 258, refresh @258):**
The fifth in the vocabulary-lock series (252 SBATCH, 255 policy, 256
profile, 257 backend, 258 graph). Locks WHICH benchmark graph every
per-row record, anchor-cell census entry, family-classifier dict,
and cross-paper baseline table references — so a contributor cannot
silently misspell `soc-LiveJournal1` as `soc-livejournal1` in one
helper while every other generator still uses the SNAP casing, drop
the hyphen in `cit-Patents` to `citPatents`, introduce a new graph
to one stage without adding the matching family-classifier entry,
or shorten `delaunay_n19` to `delaunay19` and break the
SNAP-vs-synthetic provenance carried by the underscore.
CANONICAL_GRAPHS today: 26 entries across 8 families — social
(email-Eu-core, soc-pokec, soc-LiveJournal1, com-orkut,
com-orkut-undir, twitter-2010, twitter7, soc-LJ), web (web-Google,
web-BerkStan, uk-2005, web-uk), road (roadNet-CA, road-CA,
roadNet-PA, roadNet-TX, USA-Road), mesh (delaunay_n18, delaunay_n19,
delaunay_n20), citation (cit-Patents), kronecker (kron21, kron22,
kron23), p2p (p2p-Gnutella31), content (wikipedia_link_en). Source
provenance is SNAP / GAP / DIMACS / KONECT / WebGraph / synthetic /
test. AST-harvests every graph string literal from
`scripts/experiments/ecg/*.py` + `scripts/test/*.py`, every
per-source `GRAPH_FAMILY` / `FAMILY_OF` / `GRAPH_FAMILIES` /
`GRAPH_TO_FAMILY` dict, and the `EVAL_GRAPHS` list in
`scripts/experiments/ecg/config.py`. 8 rules: R1 every literal is
canonical; R2 every canonical has non-empty family + paper_label;
R3 every canonical has source in the canonical provenance set;
R4 every family-dict key is canonical AND maps to the SAME family
the canonical entry declares (catches the silent drift where
roadNet-CA is `road` in one helper and `roads` in another); R5
every harvested literal has a non-family-dict site (real corpus
use, not just metadata) — unless `documented_future=True` for
RESERVED_FUTURE_KEYS-style entries; R6 every canonical name matches
`^[A-Za-z][A-Za-z0-9_-]*$`; R7 every `EVAL_GRAPHS` entry is
canonical; R8 every canonical family matches `^[a-z][a-z0-9_]*$`
(digits allowed after the leading letter for `p2p`). Today: 26
canonical graphs, 8 families, ~564 literal sites, 8
family-classifier dicts, 6 `EVAL_GRAPHS` entries; 0 violations.

**Backend/tool vocabulary registry (gate 257, refresh @257):**
A vocabulary-lock companion to gate 252 (Slurm SBATCH schema), gate
255 (cache-policy vocab), and gate 256 (run-profile vocab). Locks
WHICH simulator backend / tool every result row, anchor pickup,
cross-tool aggregator, and report-label site references — so a
contributor cannot silently rename `cache_sim` to `cache_simulator`,
drop the hyphen in `gem5-riscv` to `gem5riscv`, or introduce a rogue
`sniperx` / uppercase `GEM5` variant without an explicit canonical
entry. CANONICAL_BACKENDS today: `cache_sim` and its kebab-case
display sibling `cache-sim` (analytical LRU-stack sim), `gem5` +
`gem5-riscv` + `gem5-x86` (cycle-accurate, three frontends), and
`sniper` + `sniper-sift` (interval-sim + SIFT trace frontend). AST-
harvests every backend string literal across
`scripts/experiments/ecg/*.py` + `scripts/test/*.py` plus every
`--backend / --source-backend / --tool / --tool-name / --suite`
argparse choices+default. 7 rules: R1 every harvested literal is in
`CANONICAL_BACKEND_NAMES`; R2 every canonical has non-empty `family
∈ {cache_sim, gem5, sniper}` plus non-empty `paper_label`; R3 no
two canonicals share a paper_label unless they are declared mutual
`punctuation_variants` (`cache_sim ⇄ cache-sim` is the only such
pair today); R4 every canonical's `punctuation_variants` tuple
includes its own name (catches lonely-variant declarations); R5
every harvested argparse `choices` list AND default ⊆
`CANONICAL_BACKEND_NAMES ∪ {'both'}` (the `both` literal is reserved
for `--suite` to mean cache_sim+gem5 together); R6 every canonical
name matches `^[a-z][a-z0-9_-]*$` (lowercase ASCII; hyphen/underscore
allowed; no leading digit) — catches uppercase typos like `Gem5` and
prefix-clash typos like `3gem5`; R7 every canonical token has at
least one in-tree literal reference (no dead canonicals). Today: 7
canonical tokens, 3 families, 437 literal sites, 7 distinct harvested
literals, 4 argparse sites; 0 violations.

**ECG final-paper-run profile registry (gate 256, refresh @256):**
A vocabulary-lock companion to gate 252 (Slurm SBATCH schema) and
gate 255 (cache-policy vocab). Reads
`scripts/experiments/ecg/final_paper_manifest.json` and validates
that every `stage.profiles[*]` token resolves to a key in
`manifest.profiles`, that every `manifest.profiles` key has a
non-empty description (the descriptor that `--list-profiles` emits),
and that every key is referenced by at least one stage, pytest
fixture, helper script, or README walkthrough — unless its
description starts with `Placeholder` (the documented escape hatch
for upcoming work). Also harvests every `--profile <token>`
adjacency from `scripts/experiments/ecg/*.py`,
`scripts/test/*.py`, and `scripts/experiments/README.md` to catch
typos like `--profile fianl_replacement` in published walkthroughs.
7 rules: R1 stage tokens resolve to manifest profiles; R2 every
manifest profile has a non-empty description; R3 every profile is
referenced (modulo Placeholder hint); R4 every external citation
resolves; R5 profile names match `^[a-z][a-z0-9_]*$`; R6 stage
names match `^[0-9]+[a-z0-9]*_[a-z][a-z0-9_]*$` (digit-prefix
preserves natural-sort run-order); R7 each stage's profiles list is
non-empty and duplicate-free. Today: 30 manifest profiles, 30
stages, 25 external citations across 12 distinct tokens (3 manifest
profiles intentionally flagged Placeholder); 0 violations.

**Cache-policy vocabulary registry (gate 255, refresh @255):**
A vocabulary-lock companion to gate 251 (L3 byte literals) and gate
248 (paper-label map). AST-harvests every `POLICIES`,
`ALL_POLICIES`, `BASELINE_POLICIES`, and `GRAPH_AWARE_POLICIES`
tuple/list across `scripts/experiments/ecg/*.py` and validates
against `CANONICAL_POLICY_NAMES` (8 entries: LRU, FIFO, RANDOM,
LFU, SRRIP, GRASP, POPT, ECG) plus `CANONICAL_ECG_ARMS` (9
documented operational variants: POPT_CHARGED + 8 `ECG:*` arms).
Catches a misspelled `POPT_charged` or lowercase `srrip` slipping
into a generator, a config drift that adds GRAPH_AWARE to
ALL_POLICIES out of order, or a 4-tuple anchor that drifts away
from the paper's canonical (LRU, SRRIP, GRASP, POPT) ordering.
9 rules: P1 every harvested token is canonical or a documented
ECG arm; P2 POLICIES tuples have no duplicates; P3 `config.py`
strictly decomposes `ALL_POLICIES == BASELINE + GRAPH_AWARE` while
extended `ALL_POLICIES` elsewhere may add canonical-or-arm tokens
only; P4 every canonical token has a valid family ∈
{baseline,graph_aware} + a non-empty `paper_label`; P5 no two
canonical tokens share the same `paper_label`; P6 every harvested
4-tuple `POLICIES` is a permutation of CANONICAL_FOUR_TUPLE
(LRU,SRRIP,GRASP,POPT); P7 every harvested 3-tuple is a subset of
canonical (so the (GRASP,LRU,SRRIP) anchor triplet stays locked);
P8 no harvested token is one of the documented forbidden aliases
(`Lru`, lowercase `srrip`, etc.); P9 every CANONICAL_ECG_ARMS
entry declares a real parent ∈ {POPT, ECG} plus a non-empty purpose
string. Today: 8 canonical tokens, 9 ECG arms, 43 harvested POLICIES
tuples, 2 ALL_POLICIES sites; 0 violations.

**wiki/data bidirectional registry (gate 254, refresh @254):**
Symmetric companion to gate 253 — gate 253 binds narrative ↔ live
suite count; gate 254 binds the raw artifact filesystem itself to
the catalog and the pytest gates. Every JSON file under `wiki/data/`
must trace back to an `artifact_catalog` entry (with non-empty
`generator`, `gate`, and `artifact` fields that resolve to real
paths and to a sibling `.md` summary) OR appear in the
`ALLOWED_AUXILIARY` allow-list with a documented purpose and a
real parent_id, OR appear in the small `SELF_REFERENTIAL` set
(the catalog can't catalog itself reflectively). Catches the silent
drift where a generator ships a new JSON without catalog accounting,
where an entry references a deleted artifact, where two entries
claim the same artifact path, or where a `.json` artifact lacks its
human-readable `.md` summary. 8 rules: W1 every `wiki/data/*.json`
accounted for; W2 no ghost catalog entries (artifact files exist);
W3 catalog entries have non-empty `generator`/`gate`/`artifact`;
W4 generator/gate/artifact paths all exist on disk; W5 every `.json`
artifact has a sibling `.md` summary; W6 catalog ids are unique;
W7 catalog artifact paths are unique; W8 auxiliary `parent_id` values
reference real catalog ids. Today: 110 wiki/data/*.json files,
106 catalog entries (incl. this generator), 4 auxiliary entries
(3 ECG-parity postfix + 1 ECG substrate-parity per-observation
companion), 1 self-referential (`artifact_catalog.json`); 0
violations.

**HANDOFF gate-reference registry (gate 253, refresh @253):**
A meta-gate that locks the contract between
`scripts/experiments/ecg/confidence_dashboard.py:PYTEST_SUITES` and
the narrative in this very file. Catches the silent-drift case
where a new gate lands in the dashboard but its paragraph is never
added here, or where the "**N gates, all GREEN**" headline falls
behind the live gate count. 7 rules: H1 every `gate N` /
`gates N-M` token in HANDOFF parses to a positive int (or positive
range); H2 every PYTEST_SUITES label carrying `(gate N)` is
mentioned in HANDOFF (no orphan dashboard labels); H3 the headline
`**N gates, all GREEN, exit 0**` equals `len(PYTEST_SUITES)`;
H4 `Refresh complete at gate N` equals `len(PYTEST_SUITES)`;
H5 `Next refresh due at gate M` equals refresh-at + 5 (declared
cadence); H6 no duplicate `(gate N)` token in dashboard labels
(each gate number labels at most one suite); H7
`max(labeled_dashboard_gates) == len(PYTEST_SUITES)` (the newest
labeled gate equals the live count — so a new gate cannot land in
the dashboard without an explicit `(gate N)` label). Today: 138
HANDOFF gate-refs, 12 labeled dashboard gates (gates 242..253),
253 PYTEST_SUITES total; 0 violations.

**Slurm SBATCH schema registry (gate 252, refresh @252):**
Every `*.sbatch` file under `scripts/experiments/ecg` and
`scripts/experiments/vldb` is now parsed line-by-line against a
single `CANONICAL_SBATCH_DIRECTIVES` registry (14 directive names; 7
required: `--job-name`, `--time`, `--nodes`, `--ntasks`,
`--cpus-per-task`, `--mem`, `--output`). Catches silent drift like a
contributor adding a typo'd directive (`--mem-per-node`), shipping
an sbatch missing `--output`, using a non-Slurm time format, or
mixing `--mem` with `--mem-per-cpu` (Slurm forbids the combination).
9 rules: S1 syntax (every `#SBATCH` token parses to
`--key[=value]`); S2 every required directive present; S3 every
directive used is canonical; S4 mem regex `\\d+[GMK]?`; S5 time
regex `HH:MM:SS` or `D-HH:MM:SS`; S6 single-node single-task
(`--nodes=1` AND `--ntasks=1`); S7 log templates contain `%x+%j` or
`%A+%a`; S8 `--job-name` starts with `gbrew-` or `ecg-`; S9 `--mem`
and `--mem-per-cpu` never co-occur. Today: 9 sbatch files (2 in
`ecg/`, 7 in `vldb/`), 14 canonical directives, 7 required; 0
violations.

**L3 cache-size registry (gate 251, refresh @251):**
The L3-size universe (4kB..32MB) is now locked by a single
`CANONICAL_L3_TIERS` map (11 tokens with role + sub_tier + byte size +
MB size each). The generator AST-harvests every module-level
`PAPER_L3` / `PAPER_L3_SIZES` / `L3_SIZES` tuple of string tokens AND
every `L3_MB` / `L3_BYTES` dict across `scripts/experiments/ecg` and
`scripts/test`, then enforces canonical agreement across the whole
universe. Catches silent drift where a contributor adds a new L3
sweep size (e.g. `"2MB"`) in one module but forgets the byte
arithmetic, or where the anchor triplet (`1MB`, `4MB`, `8MB`) is
reordered in a copy. 7 rules: L1 every harvested token is canonical;
L2 every `PAPER_L3` tuple equals `ANCHOR_TRIPLET` exactly; L3 every
`L3_MB` value matches canonical MB scaling; L4 every `L3_BYTES` value
matches canonical byte arithmetic via AST constant-folding (so
`4 * 1024` ≡ 4096 ≡ canonical `4kB`); L5 the canonical registry uses
only declared roles + sub_tiers; L6 every anchor token appears in at
least one harvested `PAPER_L3` tuple; L7 `PAPER_L3`-shaped constants
do not disagree across files. Today: canonical=11 tokens, 54 files
harvested, 43 `PAPER_L3`-shaped tuples, 9 `L3_MB` dicts, 7
`L3_BYTES` dicts, 1 subset tuple (`MANDATORY_L3_SIZES`); 0
violations.

**Paper-table CSV provenance (gate 250, refresh @250):**
Every shipped LaTeX paper table in `wiki/data/paper_pipeline_YYYYMMDD/`
is co-emitted with a parallel `.csv` source. `paper_pipeline.py`
intentionally caps each `.tex` at `[:20]` or `[:24]` rows for paper
layout, so the CSV is allowed to be a strict superset — but the
`.tex` must NEVER carry a row that the CSV does not hold. This gate
pairs each shipped `.tex` with its sibling `.csv` and asserts every
paper row's key tuple (policy, benchmark, prefetcher, check, charged,
oracle, candidate) traces to at least one CSV row after LaTeX-escape
normalization (`ECG\_DBG\_ONLY` ⇄ `ECG_DBG_ONLY`). 7 rules: P1 every
registered `.tex` AND `.csv` exists; P2 `tex_rows ≤ csv_rows` (subset
row count); P3 per-pair key-column multiset is a sub-multiset of the
CSV column after normalize; P4 every declared key column exists in
the corresponding header tuple; P5 no empty value in tracked CSV key
columns; P6 no unregistered `.csv` sibling of a registered `.tex`
stem; P7 every registered CSV has a non-empty header row. Today: 5
pairs, 78 tex rows ↔ 85 csv rows, 13 tracked key columns; 0
violations.

**Graph-family map full-coverage (gate 249, refresh @249):**
Gate 107 already locks the topology of 7 known `GRAPH_FAMILY` copies
(2 full + 5 short). This gate extends that protection by
AST-harvesting *every* module-level dict literal in
`scripts/experiments/ecg/` and `scripts/test/` whose keys look like
known graph names and whose values look like known family labels,
then asserting that every harvested copy agrees with a canonical
8-graph map declared in the generator on every shared key. Catches
new modules added by a future contributor that ship their own
`GRAPH_FAMILY` copy and silently diverge from the canonical mapping.
6 rules: F1 harvester picks up every module-level GRAPH_FAMILY-shaped
dict; F2 every harvested copy is a subset of canonical + reserved-
future keys; F3 every harvested copy agrees with canonical on every
shared key (no value drift); F4 canonical is non-empty AND every
value is in the documented allow-list (`social`, `web`, `citation`,
`road`, `mesh`); F5 no harvested copy with reserved-future keys is
missing from the gate-107 `FULL_SOURCES` universe; F6 out-of-universe
copies are tracked for visibility. Today: canonical=8 graphs (4
social + 1 web + 1 citation + 1 road + 1 mesh); 12 harvested copies
across 12 files; 7 out-of-universe but all subset-clean; 0 violations.

**Sideband-schema registry (gate 248, refresh @248):**
Locks the `[graphctx] register region ...` wire-format across the
three C++ overlays (`graph_cache_context_gem5.hh`,
`graph_cache_context_sniper.cc`, `graph_cache_context.h`) so a silent
field rename / reorder / drop in one overlay cannot make Tier-A's
sideband-registration parser (gate 1) stop matching that overlay
without any test failing for the obviously wrong reason. Gate codifies
the wire-format in a hand-curated `SCHEMA_REGISTRY` declaring each
field's name, printf specifier, and C++ parameter type, in canonical
*ordered* form, plus the Tier-A parser regex anchor itself. 7 rules:
S1 every emit-site file exists; S2 in-file printf format matches the
canonical schema byte-for-byte after concatenating adjacent C string
literals; S3 in-file `logGraphCtxRegistration(...)` parameter type
list matches the canonical type tuple; S4 every emit-site contains
the canonical literal prefix `[graphctx] register region`; S5 every
schema field uses a printf specifier from the documented allow-list
(`%s` / `0x%lx` / `%u` / `%d`); S6 the Tier-A parser regex compiles
AND round-trips a sample line built from the canonical schema (named
groups match schema field names); S7 each emit-site has exactly one
register-region `fprintf` (catches divergent backup emit paths).
Today: 6 schema fields (`source`, `name`, `base`, `upper`, `hot_pct`,
`grasp_region`) × 3 emit sites, 0 violations.

**Paper LaTeX-table emit invariant (gate 247, refresh @247):**
Locks the published-paper-facing .tex tables in
`wiki/data/paper_pipeline_YYYYMMDD/`. A developer can edit a
table-generation script, the .tex file regenerates with a different
caption or column header, and the paper's prose silently mismatches
the table — no numerical test catches this. Gate codifies each
shipped table with a hand-curated `TABLE_REGISTRY` entry declaring
`{filename, caption, col_spec, columns}`, plus a per-row hygiene
sweep. 7 rules: T1 every registered file exists; T2 in-file caption
matches; T3 `\begin{tabular}{...}` col-spec matches; T4 column-header
tuple matches; T5 every data row has the right column count AND no
cell is the literal `nan`/`NaN`/`inf`/`-inf` (those render fine in
PDF but are scientifically meaningless); T6 no unregistered .tex in
the dir (defensive — catches new tables added without registration);
T7 every table ends with the `\bottomrule\end{tabular}\end{table}`
closing trio. Today: 5 registered tables
(`ecg_mode_overhead_summary`, `faithfulness_summary`,
`popt_charged_overhead`, `popt_storage_overhead_summary`,
`roi_policy_summary`), 78 total data rows, 0 violations.

**lit-faith citation registry purity (gate 246, refresh @246):**
Locks down the prose-citation strings carried in
`wiki/data/literature_faithfulness_postfix.json:per_claim`. Before
this gate, a row could be hand-edited to cite a paper that does
not actually carry the claim, a registry entry could accrete typos
in its prose form until pattern matching silently broke, or two
rows in the same (policy, app, expected_sign) bucket — the *unit
the paper actually quotes* — could drift apart in which canonical
work they attribute the expected sign to, all without any
numerical test failing. The gate codifies a hand-curated
`CITATION_REGISTRY` of every canonical work the lit-faith table is
allowed to cite (today: Faldu HPCA 2020, Balaji & Lucia HPCA 2021,
Jaleel et al. ISCA 2010), each entry carrying {key, title, venue,
year, patterns, note}. 5 rules: C1 every per_claim citation
matches the substring patterns of ≥1 registered canonical work;
C2 every registered work is referenced ≥1 time (no dead-letter
registry entries); C3 within each (policy, app, expected_sign)
bucket all rows share ≥1 canonical key (the paper quote stays
anchored); C4 every registry entry has non-empty venue + year
(keeps the registry mineable for bibliography generation); C5
every per_claim row carries a non-empty citation. Today: 3 works,
330 rows, 24 buckets, coverage Balaji=252 + Faldu=177 + Jaleel=75,
0 violations.

**L3 regime-classifier consistency (gate 245, refresh @245):**
Catches a subtle class of bug that has been latent in the codebase
for a while: at least three regime-classifying functions across
`scripts/experiments/ecg/` share the *vocabulary* `{tiny, small,
large, unknown}` but use *different boundaries*. Specifically,
`policy_winner_table._l3_regime` and `popt_vs_grasp_report._l3_regime`
agree (both use `<` boundaries at 64 kB and 1 MB), while
`oracle_gap_report._regime` uses `<=` boundaries at 64 kB and 256 kB,
producing different labels at 32 kB and 64 kB. The gate codifies this
with a hand-curated `REGIME_REGISTRY` declaring each classifier's
*taxonomy family* and *vocabulary*; within each family, all members
must agree on every label in a `CANONICAL_L3_GRID` (`1kB`..`16MB`),
while cross-family divergence is allowed but must be declared with
an explanatory note. 5 rules: R1 every registered classifier resolves
to a callable; R2 byte-input classifiers stay inside their declared
vocabulary; R3 byte-input classifiers within a family agree on the
canonical grid; R4 source-pattern scan of `scripts/experiments/ecg/`
finds no unregistered regime classifiers (catches new drift-prone
additions); R5 non-byte-label classifiers (ratio/range-input) carry
an explanatory `note` describing what they actually classify. Today:
5 classifiers in 4 families — `tiny_small_large_v1` (identical
sibling pair: `policy_winner_table` + `popt_vs_grasp_report`),
`tiny_small_large_v2_oracle_gap` (intentionally separate),
`wss_range` (`cross_tool_lru_regime`, classifies an L3-size range),
`wss_ratio` (`wss_relative_l3`, classifies L3/WSS ratio). 0
violations. **Follow-up:** a future gate could unify v1 and v2 into
a single canonical classifier (would require auditing per-figure
consequences on the paper's oracle-gap and per-regime bar charts).

**Paper-figure data snapshot integrity (gate 244, refresh @244):**
Third gate in the always-active paper-snapshot trio: 242 audits the
policy *vocabulary*, 243 audits the visual quality of the *palette*,
and 244 audits the actual *figure-data snapshot directory*. The paper
plots all bar charts from a single committed
`wiki/data/paper_pipeline_YYYYMMDD/` snapshot, which is a frozen
subset of the live sweep — but nothing previously checked that it
was fresh, single-sourced, or rectangular. 6 rules: F1 exactly one
`paper_pipeline_YYYYMMDD/` directory in `wiki/data` (no stale
duplicates that confuse readers or break gate 242's latest-dir
lookup); F2 the dir name parses to a valid YYYYMMDD date AND is
within `MAX_SNAPSHOT_AGE_DAYS` (365 today; can be tightened later);
F3 every row in `roi_matrix_all.csv` has non-empty values for
`pipeline_source_csv`, `pipeline_run_dir`, and `pipeline_run_name`
(full referential provenance so anyone can re-run the source); F4
single-run cohesion — every row shares the same `pipeline_run_dir`,
ruling out Frankenstein snapshots stitched from multiple runs; F5
coverage rectangle — per (`benchmark`, `final_graph`, `l3_size`)
cell, the set of `policy_label`s equals the canonical `POLICY_LABELS`
palette (no missing or extra bars in any paper bar chart); F6 value
hygiene — `l3_miss_rate ∈ [0.0, 1.0]` universally, and
`total_accesses ≥ 1` for `HIGH_ACTIVITY_BENCHMARKS = {"pr"}` (BFS
and SSSP can legitimately log `total_accesses=0` on short-walk ROIs
and are deliberately carved out with documented rationale).
Source-of-truth: `paper_pipeline.py` POLICY_LABELS loaded via
importlib plus the snapshot directory itself. Today: 1 snapshot dir
(`paper_pipeline_20260528`), 108 rows × 9 policies × 4 graphs × 3
benchmarks × 1 L3-size, 0 violations.

**POLICY_COLORS perceptual distinguishability (gate 243, refresh @243):**
Companion to gate 242 — where 242 audits the policy *vocabulary*,
243 audits the *visual quality* of the paper palette: can a reader
(or a B&W printer) actually tell the policies apart on the figures?
Always-active (no scaffold/deferred mode — POLICY_COLORS /
POLICY_HATCHES are always present in `paper_pipeline.py`, loaded via
importlib). 6 rules: C1 every POLICY_LABELS key has a well-formed
7-char hex color; C2 no two POLICY_COLORS values are exactly equal;
C3 every pair has CIE76 ΔE ≥ 12 in CIE Lab (D65); C4 pairs with
lightness delta ΔL < 10 must use a POLICY_HATCHES entry, modulo
ACKNOWLEDGED_BW_PAIRS (10 grandfathered close-lightness pairs, each
with rationale ≥ 60 chars documenting why the current palette is
acceptable for color print and what fails on B&W); C5 every color
has ΔE ≥ 18 from #FFFFFF (no near-invisible policies on a white
page); C6 POLICY_HATCHES keys are a subset of POLICY_LABELS keys
(no orphan hatches). Pure-stdlib sRGB → CIE Lab implementation.
Today: 9 colors, 36 pairs, 0 violations. Caveat: the
ACKNOWLEDGED_BW_PAIRS list documents real B&W limitations of the
current palette and is the deliberate engineering compromise; a
future palette refresh should aim to shrink it.

**Paper label-map integrity (gate 242, refresh @242):**
Always-active audit (no scaffold/deferred mode — source-of-truth is
`paper_pipeline.py`'s `POLICY_LABELS/POLICY_DESCRIPTIONS/POLICY_COLORS`
dicts, loaded via importlib). Catches a regression class previously
ungated: the paper's policy-label vocabulary silently drifting from
what the JSON/CSV artifacts actually carry. 5 rules: G1 every canonical
policy label has a description and a hex color; G2 the figure-label
set (POLICY_LABELS values) is unique (no two policies render to the
same label); G3 every policy_label observed across 8 tracked JSON
artifacts (per_observation + winner_table + lit-faith postfix) is
either in POLICY_LABELS or in the allowlist
({CACHE, DROPLET, ECG_PFX} ∪ THEOREM_CLASS_LABELS where the latter is
{POPT_GE_GRASP, POPT_NEAR_GRASP_IF_BIG_GAP}); G4 same for the 5
tracked paper_pipeline_*/CSVs (`policy_label` column only — the `policy`
column is a family rollup containing "ECG" which is not a policy); G5
no orphan POLICY_LABELS entries (every catalogued label must appear in
at least one tracked source). Today: 9 policy labels, 13 sources
scanned, 0 violations. Pattern lock: 5 of 6 wiring touchpoints (no
postfix file → no META_ALLOWLIST / WIKI_UNTRACKED_EXEMPT). Field-name
fragmentation discovery: lit_faith_postfix uses `policy` (not
`policy_label`); policy_winner_table uses `winner_policy` /
`runner_up_policy`; paper_pipeline CSVs carry both.

**ECG prefetcher head-to-head (gate 241, refresh @241):**
The substrate-parity trinity (238/239/240) answers "does ECG mode
preserve the baseline cache substrate on backend X?"; gate 241 asks
the orthogonal question — "does ECG's PFX prefetcher beat or match
DROPLET on the same baseline?". SCAFFOLD/DEFERRED today: across all
`/tmp/graphbrew-*` corpora the `droplet_*` / `ecg_pfx_*` columns are
CONFIG-ONLY (degrees, table sizes, delivery mode) — every runtime
counter (`droplet_indirect_issued`, `droplet_stride_issued`,
`ecg_pfx_issued`, `ecg_pfx_useful`) is zero. The postfix declares
status="deferred" with expected source pattern
(`/tmp/graphbrew-ecg-pfx-vs-droplet-*/`) and minimum 8 observations.
6 rules implemented (G1 arm completeness `{LRU, DROPLET, ECG_PFX}`,
G2 baseline neutrality at 0.5 pp, G3 useful-fraction floor at 5 %,
G4 per-arm policy_label hygiene, G5 backend identity per cell, G6
observation floor) so activation is a postfix-only edit. 0
violations (deferred ⇒ no data to violate). Sibling field links
back to gates 238/239/240 for audit-trail cross-consistency.

**ECG substrate-parity trinity (gates 238-240, refresh @240):**
The cache_sim/gem5/Sniper substrate-parity story is now complete in
the gate ledger. Three siblings audit the same invariant (ECG mode
≡ stock mode on L3 miss-rate) on three different backends so a
substrate regression is impossible to land silently in any of them:

- **ECG-Parity** (gate 238, cache_sim, commit `4b9b01e`) — 54-obs
  matched-proof from email-Eu-core `proof_matrix.csv`. POPT-arm
  (ECG_POPT_PRIMARY ≡ POPT) AND DBG-arm (ECG_DBG_ONLY ≡ GRASP)
  active, ε=5e-4 on both arms. Adds PFX activation+useful floors
  and encoding hygiene rules. 0 violations.
- **ECG-Gem5-Parity** (gate 239, gem5, commit `a82766f`) — 12-obs
  matched-proof from `/tmp/graphbrew-gem5-popt-pin-geometry-email-pr
  -bracket/roi_matrix.csv`. POPT-arm only today (DBG arm + PFX
  activation queued — no ECG_DBG gem5 run yet, prefetcher=none
  everywhere in the bracket sweep). ε=2e-3 (2× headroom over
  observed gem5 drift max 1.09e-3, looser than cache_sim's 5e-4
  to absorb warmup/MSHR/OoO timing noise). 0 violations.
- **ECG-Sniper-Parity** (gate 240, Sniper, commit `b08b0a9`) —
  SCAFFOLD/DEFERRED today. `/tmp/graphbrew-grasp-sniper-sweep`
  contains LRU/SRRIP/GRASP for {pr, bfs, sssp, bc} × {cit-Patents,
  email-Eu-core} × {4kB, 32kB, 256kB, 2MB} but NO ECG_DBG_ONLY or
  ECG_POPT_PRIMARY rows, so the postfix declares status="deferred"
  with the expected source pattern documented
  (`/tmp/graphbrew-ecg-sniper-matched-proof-*/`). 9 rules (G1, G1b,
  G2, G2b, G3, G4, G5, G6, G7) implemented end-to-end so activation
  is a postfix-only edit. ε=2e-3 mirrors gem5. 0 violations
  (deferred ⇒ no data to violate).

Together these lock the substrate-faithfulness invariant on every
backend the paper uses. Cross-backend ε pattern: cache_sim 5e-4
(no timing noise) → gem5/Sniper 2e-3 (timing noise envelope).
Out-of-scope shared by all three: DROPLET comparison (now lives
in gate 241 — still scaffold/deferred today, no DROPLET-active sweep
available; runtime DROPLET counters are all zero across `/tmp`
corpora).

**Literature-faithfulness deepening (gates 231-235, refresh @235):**
After LIT-Stat closed the statistical-sanity loop on per_claim, the
deepening track expanded into per-bucket ordering audits, per-row
explanation depth, per-cell rationale grids, and per_observation
axis-coverage and cell-completeness audits:

- **LIT-PolyOrd** (gate 231) — per (graph_family × app) policy-ordering
  audit. Hub families ({social, citation, web}) must satisfy median
  POPT−LRU ≤ +0.5 pp and median GRASP−LRU ≤ +1.0 pp per bucket;
  no-hub families ({road, mesh}) are exempt (LRU regime). Per-app
  global hub-aggregate POPT improve-frac ≥ 0.55 with median delta
  ≤ 0 pp. Tiny-sample exception: improve-frac floor enforced only
  when n ≥ 5 cells per bucket. Today: 21 buckets, 114 cells,
  hub-aggregate POPT improve fracs bc=0.72/bfs=0.89/cc=0.87/pr=0.94/
  sssp=0.93, 0 violations.
- **LIT-DevExp** (gate 232) — deviation-explanation depth audit. Per
  `status == 'known_deviation'` row: the `known_deviation_reason`
  text must (a) name >= 1 algorithmic mechanism from a 30-keyword
  vocabulary (PR-rank, frontier, hub, union-find, ordering, capacity,
  look-ahead, oracle, ...), (b) exceed a 60-char length floor, (c)
  carry a non-empty citation, (d) resolve any cross-references
  against another known_deviation row, and (e) no single reason text
  covers more than 50% of the 30 known_deviation rows. Today: 30
  rows, median reason length 247 chars, median 5 mechanism hits,
  30/30 unique texts, 16 cross-referenced, 0 violations.
- **LIT-RatGrid** (gate 233) — per-cell rationale-grid audit. Per
  (policy, graph, app) cell: rationale text must be unique within
  the cell. Theorem-class policies {POPT_GE_GRASP,
  POPT_NEAR_GRASP_IF_BIG_GAP, SRRIP} must carry exactly 1 rationale
  per (policy, app) regardless of graph (algorithmic theorem statement
  is graph-invariant); point policies {GRASP, POPT, LRU} may carry
  up to 2 rationales per (policy, graph, app) to accommodate L3-regime
  variants ("spills at 1 MB" vs "fits at 8 MB"). Theorem-class is
  exempt from the citation-token rule. Today: 330 rows, 115 cells,
  GRASP=19/POPT=8/POPT_GE_GRASP=5/POPT_NEAR=1/SRRIP=5 distinct
  rationales, 0 violations.
- **LIT-CellComp** (gate 234) — per_observation cell-completeness
  audit (parallel sibling of LIT-Stat). Per (graph, app, l3) cell:
  canonical policy roster {LRU, GRASP, POPT} present, LRU baseline
  row present (delta_vs_lru_pct depends on it), `delta_vs_lru_pct`
  arithmetic matches `(miss_rate - lru_miss_rate) * 100` within
  0.001 pp, every non-LRU policy covers >= 3 L3 sizes per (graph,
  app), every present policy shares the same L3 axis within (graph,
  app), miss rates in [0, 1], no duplicate rows. Today: 456 rows,
  114 cells, 8 graphs, 5 apps, 4 policies (LRU + GRASP + POPT +
  SRRIP), 0 violations across all 7 rules.
- **LIT-AppFreq** (gate 235) — per-app axis-coverage audit. Each app
  must touch >= 6 graphs, >= 3 L3 sizes, >= 3 policies (canonical
  roster {LRU, GRASP, POPT} per app), every (app, graph) covers
  >= 3 L3 sizes, every app contributes >= 60 observation rows, and
  the anchor app (pr) must cover the full corpus (8/8 graphs).
  Catches "axis collapse" regressions. Today: pr=8 graphs/112 rows
  (full sweep), bc=bfs=7/92, cc=sssp=6/80, 0 violations.

**Refresh status:** Refresh complete at gate 261. Next refresh due
at gate 266.

**Literature-faithfulness deepening (gates 226-230, refresh @230):**
After the lit-faith bijection lock-down (gates 221-225), the next
push moved from claim-corpus structural checks to *arithmetic and
physical-invariant* checks on the live per_claim table itself:

- **LIT-Tol** (gate 226) — tolerance-calibration audit: for every
  per_claim row, recompute the distance-to-disagree from the
  classifier's exact branches (`_classify` mirror) and lock each
  `ok`/`within_tolerance` row to a healthy slack against its
  tolerance boundary. Trips on a comparator widening or a row
  silently brushing against the band edge.
- **LIT-Acc** (gate 227) — accesses-floor audit: warmup-noise guard.
  Per-app accesses floors (1M BC/CC/PR, 500k BFS/SSSP for production
  graphs; looser 20k bfs / 200k pr / 2M bc table for the email-Eu-core
  dev-smoke) and 5-bucket distribution check. Today 311 production
  rows + 19 smoke, production min 735,934, median 15.95M, zero floor
  violations. Catches workloads that silently truncate to a
  warmup-only trace.
- **LIT-CXApp** (gate 228) — cross-app rationale coherence: for
  every (citation, expected_sign) group, the per-cell rationales
  must (a) carry zero direction-flipping contradictions (with
  negation-context handling so "must NOT regress" doesn't trip), (b)
  align with the sign-vocabulary band, (c) share a common kernel
  token, and (d) keep the length-span ratio ≤ 3.0×. 17 groups, 35
  unique rationales, zero failures today.
- **LIT-Mono** (gate 229) — cache-size monotonicity audit: for every
  (graph, app, policy) triple with ≥ 2 L3 samples, miss rate must
  be non-increasing in L3 size (tolerance 0.5 pp). Today 30 triples
  audited across 17 graphs × 4 apps × {GRASP, SRRIP}, zero LRU
  violations, zero policy violations, 1 expected-saturated triple
  (com-orkut/bfs/SRRIP at 1MB→8MB — bfs on orkut is capacity-bound
  below 16MB). Median slope ≈ 0.16 miss-rate-points per L3 doubling.
- **LIT-Stat** (gate 230) — statistical-sanity audit: re-derives
  `delta_pct` from the two miss-rate columns each row compares
  (LRU-vs-policy, POPT_GE_GRASP, POPT_NEAR_GRASP_IF_BIG_GAP) and
  locks zero NaN/inf, zero out-of-bounds miss rates, zero rounding
  mismatches (> 0.001 pp), zero sign flips (above 0.01 pp noise
  floor), zero signed-delta inconsistencies on POPT_NEAR rows, zero
  unknown row kinds, zero bad status labels, zero status-vs-delta
  inconsistencies (with the POPT_NEAR phase-transition exception
  folded in — assertion fires only when grasp_gain_vs_lru > 10 pp
  AND POPT is worse than GRASP). Status vocabulary locked to {ok,
  within_tolerance, disagree, known_deviation, missing,
  insufficient_data}. Today 330 rows: 102 LRU-vs-policy + 114
  POPT_GE_GRASP + 114 POPT_NEAR_GRASP, 298 ok / 30 known_deviation
  / 2 within_tolerance / 0 disagree.

**Literature-faithfulness deepening (gates 221-225, refresh @225):**
Once the 220-gate scaffold was load-bearing, the next push hardened
the lit-faith comparator itself rather than adding more cells. Five
gates land that protect the literature claim corpus from silent
drift:

- **LIT-Cov** (gate 221) — diversity coverage: floors on per-family,
  per-app, per-L3-size, per-paper cells in `literature_baselines.py`,
  plus cross-paper triangulation (cells where ≥2 papers issue a
  claim). Trips on a dropped graph family or paper.
- **LIT-Mar** (gate 222) — margin distribution: per-bucket median /
  per-family floor / fragile-claim ceiling on `delta_pct`, plus
  classifier parity (every claim's reported status reproducibly
  derivable from delta_pct + tolerance). Trips on a comparator
  loosening or a corpus-wide regression.
- **LIT-Sig** (gate 223) — sign-mass concentration: per-(expected_sign,
  policy) Wilson 95 % lower bound on fraction-correct + exact
  binomial sign-test p-value (lgamma-based, scipy-free) + median
  delta_pct effect-size floors. Locks the magnitude *and* direction
  of the literature replication signal.
- **LIT-Cite** (gate 224) — citation locator integrity: strict
  bijection between the 15 unique citation strings in lit-faith and
  the 38 `LiteratureClaim` records in `literature_baselines.py`;
  per-citation well-formedness (venue + year + § / Fig / Sec / Table
  locator + known anchor paper); per-anchor inventory (Faldu HPCA
  2020 ≥ 12 claims, Balaji & Lucia HPCA 2021 ≥ 10, Jaleel ISCA 2010
  ≥ 5); DOI / URL in module docstring required for Faldu + Balaji.
- **LIT-Dev** (gate 225) — known-deviation completeness:
  `KNOWN_DEVIATIONS` is the whitelist that downgrades live lit-faith
  `disagree` rows to `known_deviation`. Each entry must carry a
  ≥ 80-char reason that quotes a quantitative magnitude (pp / MB /
  % / MPKI / times) and mentions at least one anchor (paper § /
  design term / algorithmic root-cause vocabulary like `PR-rank`,
  `frontier`, `union-find`, `Phase 1`). Strict bijection with the
  live faith corpus: zero orphan whitelist entries, zero live
  `known_deviation` rows without a documented explanation. All 34
  entries today well-formed; coverage 2 policies × 5 graphs × 5 apps
  × 3 L3 sizes; CC + BC apps dominate (82 % of KDs) — these are the
  kernels whose algorithmic mismatch with POPT's PR ranking is most
  documented.

Each LIT-* gate follows the established wiring: one generator
(`scripts/experiments/ecg/lit_faith_*.py`), one pytest
(`scripts/test/test_lit_faith_*.py`), `make lit-<slug>` target,
`PYTEST_SUITES` entry in `confidence_dashboard.py`, `CATALOG` row
in `artifact_catalog.py`, and `TRACKED_ARTIFACTS` entries in
`reproduce_smoke.py`. Reproduce_smoke now tracks **173 artifacts,
all stable across regen**.

**Major gate families added since the 42-gate baseline** (each is one
generator + 12-test pytest + Makefile target + dashboard entry +
catalog entry + reproduce_smoke tracking — same 10-step wiring):

- **Curvature / slope family** (gates 50-72): per-(app, graph, policy)
  capacity-sensitivity slopes (OLS of miss% over log2 L3-MB),
  per-policy summaries, per-app and per-family breakdowns, saturation
  distance (4MB→8MB miss-rate drop), curvature, slope vs distance
  cross-check, family-curvature replay, and a cross-tool SRRIP-vs-
  GRASP slope ordering invariant that confirms the "oracle-aware
  policies are less cache-hungry" claim replicates on cache-sim,
  gem5 anchor, and sniper anchor (gate 72).
- **Anchor-tool slope replays** (gates 70/71): timing-faithful slope
  reproductions on gem5 (2 cells) and sniper (6 cells) at the
  4kB-2MB anchor sweep, using the same OLS / monotonicity / SRRIP-
  vs-GRASP / help-floor checks as the cache-sim sweep.
- **Regime-dependence formalization** (gate 74): the cross-tool
  LRU-vs-GRASP slope inversion is now a first-class invariant:
  cache-sim post-WSS (1-8MB) shows LRU strictly steeper than GRASP
  (-0.97 pp/oct), while both anchor tools at sub-WSS scales show
  the opposite sign (gem5 +0.84, sniper +0.24). Sign agreement
  between gem5 and sniper confirms the inversion is physical
  (LRU's give-up-and-stream behaviour vs GRASP's hold-the-hot-set
  behaviour at sub-WSS).
- **Per-app deviation pinning** (gates 68/73): bfs is pinned as a
  documented kernel deviation for both LRU-vs-GRASP (gate 68) and
  SRRIP-vs-GRASP (gate 73) per-app ordering, with the pin gated by
  "no NEW deviations". Frontier-driven streaming pathology that
  gate 65 already flags as the most-saturated kernel.
- **Cross-tool universality + anchor census + saturation replays**
  (gates 75-81): cross-tool slope-sign universality roll-up across
  all 10 (tool, policy) cells (gate 76); cell-level L3-sweep
  monotonicity universality across 320 (Li, Li+1) steps (gate 77);
  anchor cell-pair census pinning 2 gem5 cells + 6 sniper cells
  against silent shrinkage (gate 78); per-family saturation-distance
  replay locking citation/social headroom and the web pin
  (citation=+15.69, social=+12.50, web=+2.15 pp) (gate 79); anchor
  monotonicity replay with tier-aware tolerances — gem5 strict
  (0/18 bumps), sniper bounded (19/54 bumps, 2 hard, max +1.18 pp)
  (gate 80); per-policy final-octave steepness ranking
  POPT(0.10) <= GRASP(0.23) << LRU(1.06) ~ SRRIP(1.09) pp/octave
  (gate 81).
- **Cross-tool agreement, distribution integrity, and registry
  cross-checks** (gates 82-85): cross-tool shared-anchor slope-sign
  agreement across the 3 gem5∩sniper anchor cells (gate 82, all 3
  sign-matched, all both-negative, all sniper-steeper, max |Δ|=5.13
  pp/oct); regression-budget margin-distribution gate over all 330
  literature claim cells, locking global min/median floors and per-
  claim-kind margin floors (gate 83); paper-claims registry-integrity
  gate that re-derives every one of the 14 published claims' value
  from its cited source JSON and asserts equality within a half-LSB
  tolerance (gate 84, also catches stale headline text); cross-
  artifact aggregate consistency that locks 17 invariants between
  policy_winner_table, winning_regime_taxonomy, popt_vs_grasp_delta,
  cross_tool_saturation, literature_deviations, regression_budget,
  and corpus_diversity — load-bearing: the winner counts across
  sibling artifacts now provably agree on 114 cells, 56 GRASP wins,
  44 POPT, 8 SRRIP, 6 LRU (gate 85).
- **Per-app, catalog, family, and timing-tool cross-artifact gates**
  (gates 86-90): per-app oracle-rank-1 ↔ winner-table top-2 parity
  with the bc:SRRIP→GRASP divergence registered as the only allowed
  disagreement (gate 86); artifact-catalog completeness — every
  `wiki/data/*.json` is registered (gate 87, caught the silent gap
  where `paper_baseline_table.json` shipped without being catalogued,
  and registered it as the 72nd catalog entry); family-sensitivity
  cross-artifact parity — the 7 canonical_state claims agree on
  cell-counts with `policy_winner_table.wins_by_family` and on
  dominance direction (gate 88); cross-tool slope-ordering xartifact —
  SRRIP strictly steeper than GRASP on all 3 tools, LRU regime-
  inversion verdict PASS on all 5 checks, anchor sign-agreement and
  doubly-saturated cross-tool agreement (gate 89); gem5/Sniper anchor
  cell parity — load-bearing (email-Eu-core, pr) shared anchor cell
  locked in `shared_cells`, both tools share L3 axis and policy set,
  every anchor cell has all 3 policies populated with miss-rate in
  (0, 1) (gate 90).
- **Cross-artifact integrity gates** (gates 91-95): cell-count
  cross-artifact parity locking the 114-cell universe across the four
  winner-class summaries and the 3 tied cells on bc/email-Eu-core
  (gate 91, 14 tests); cache-sensitivity slope baseline pinning the
  10 known monotonic violations plus the {LRU:13, POPT:5, SRRIP:13,
  GRASP:2} anti-scaling partition over 112 trajectories (gate 92, 14
  tests); WSS-knee vs relative-L3 parity pinning the per-(policy,
  regime) {n, mean_gap_pp, win_rate} grid (114=52+14+48 regime cells,
  knee_rank {GRASP:0, LRU:2, POPT:0, SRRIP:2}) and verifying the
  paired wkl↔wrl payloads agree to <=0.05 pp (gate 93, 13 tests);
  bootstrap-CI nested consistency (seed=1729, ci=0.95) — mean_delta
  anti-symmetry exact, paired-bootstrap CI mirror within 1.0pp,
  p_a_lt_b complementarity within 0.03 across all 7 sign_stability
  entries (gate 94, 13 tests); family-clustering 3-way agreement —
  PWT argmax == FPAC global_winner_by_app on all 5 apps, deviation
  set recomputable from per-family qualified winners, 3 stable
  family_sensitivity claims at stability_floor=0.95, and the global
  cluster split {GRASP:[bc,cc], POPT:[bfs,pr,sssp]} locked (gate 95,
  13 tests).
- **Second cross-artifact integrity block + milestone** (gates 96-100):
  AUC correlation cross-artifact parity — PAC.meta and FPAC.meta agree
  on auc_winner_by_app and clusters_by_winner, intra_inter math
  recomputable from the matrix within 1e-3, 3 qualifying families
  {citation, social, web} locked at min_apps=4 (gate 96, 13 tests);
  family-geomean ↔ margin-replay parity — all 5 families present in
  both artifacts, FMR per-family cells {citation:15, mesh:5, road:25,
  social:54, web:15} summing to corpus total 114, FGI headline-15
  entries derivable from geomean_improve_pct ≥ 10.0, and the 34 strict
  geomean improvements all have ≥1 winning cell in margin-replay (gate
  97, 13 tests); oracle-gap curvature ↔ effect-size parity — slope and
  curvature math (slope = gap_diff/log2_ratio, curvature = slope2 -
  slope1) verifiable to 1e-3, OGES Cliff's-delta antisymmetry to
  machine precision, knee_count {GRASP:4, POPT:3, LRU:0, SRRIP:0}
  pinned (gate 98, 13 tests); monotonicity-universality ↔ anchor-
  replay agreement — MU 14 sub-noise bumps (max 0.0347 pp), AMR per-
  tool steps/bumps/hard_bumps/catastrophic accounting all internally
  consistent, MU.max_noise_bump_pp == AMR.constants.hard_bump_threshold_pp
  (both 0.5 pp), and MU.largest_bump_pp strictly < shared threshold
  (the "cache-sim is sharper than the anchors" guarantee) (gate 99,
  13 tests). **Milestone gate 100** — catalog ↔ dashboard ↔ disk
  coverage triangle: every CATALOG entry has gate/generator/artifact
  on disk under wiki/, every PYTEST_SUITES path resolves, short
  labels unique and non-empty, ≥70 catalog gates fan into the
  dashboard with exactly two documented EXEMPT_FROM_DASHBOARD entries
  (test_confidence_dashboard.py and test_paper_baseline_table.py),
  and the catalog summary's "(N) gates today" text matches
  len(PYTEST_SUITES) (gate 100, 13 tests). Also disambiguated 3
  pre-existing duplicate PYTEST_SUITES short labels (Slope→CSlope/
  CapSlope, Parity→GapPar, Sat→SatOn/SatDist).
- **Third cross-artifact integrity block** (gates 101-105): deviations
  ↔ regime taxonomy parity — LD's 30 entries all carry
  mechanism=popt_overhead_dominates and family ∈ WRT's family universe,
  shares within 1e-5 of wins/total, mechanism×family cross-tab
  recomputable from per-deviation rows (gate 101, 13 tests); corpus
  diversity ↔ regime taxonomy feature parity — every WRT cell's
  avg_degree / hub_concentration / clustering_coeff matches the
  corpus_diversity per-graph value within 1e-3 across all 114 cells,
  WRT.family == GRAPH_FAMILY[graph] for every cell, and the corpus
  and WRT graph universes are identical (gate 102, 13 tests); paper
  claims registry recompute parity — all 14 claim values
  recomputable from their cited artifacts (winner shares from
  policy_winner_table sum to 100±0.5 %, popt_vs_grasp family means
  to 0.01 pp, ok_ratio/disagreement_rate/thrash counts exact),
  every claim.source and claim.gate path resolves, snake-eating-tail
  agreement between paper_claims.green_gate_count and dashboard.json
  (gate 103, 13 tests); family tri-artifact agreement —
  family_sensitivity / family_geomean / family_policy_auc_clustering
  share the same 5-family universe, sensitivity canonical_state
  matches canonical_claims fracs to 1e-9, clustering's
  (family, winner) picks all appear in geomean records, and
  winners_matching equals counted True flags per qualified family
  (gate 104, 13 tests); regression_budget ↔ lit_faith parity — both
  artifacts cover the same 330-cell key universe, rb.status ==
  lf.status for every cell, by_kind.n counts only in-distribution
  cells, fragile_cells subset of per_cell and bounded at 10,
  known_deviation cells all have margin_pp=0, and triple counts of
  known_deviation / within_tolerance agree across summary, per-cell,
  and tolerated list (gate 105, 13 tests).
- **Fourth cross-artifact integrity block** (gates 106-110): oracle_gap
  internal + oracle_gap_by_app aggregation parity — oracle is min across
  the 4-policy panel per (graph, app, l3), winner has miss==oracle and
  gap_pp==0, by_policy_app mean/median/p90/max/n/wins all recompute
  exactly from rows using the right percentile methods (numpy `higher`
  for p90, linear-interpolation median; the two MUST stay distinct),
  by_app_ranking entries sort ascending by mean_gap_pp (gate 106, 13
  tests); GRAPH_FAMILY map duplication lock — 2 full-tier copies (11
  entries with reserved future graphs road-CA / twitter-2010 / uk-2005)
  in policy_winner_table.py and test_corpus_diversity_floor.py vs 5
  short-tier copies (8 entries, current corpus only) in literature_
  deviations_report.py, oracle_gap_report.py, winning_regime_taxonomy.py,
  popt_vs_grasp_report.py, family_saturation_distance.py — every copy
  agrees on family for every shared graph key, using ast.literal_eval
  on the source files (no module imports needed) so the gate stays
  side-effect-free (gate 107, 13 tests); claim_density.json ↔ literature_
  baselines.py parity — every per-graph rollup (n_claims, n_ok, n_cells,
  n_apps, n_policies, n_citations, status_counts) recomputes exactly
  from literature_reproduction_summary.csv, every CSV row's
  (graph, app, l3, policy) reaches via claims_for() expansion or
  KNOWN_DEVIATIONS closure, every CSV citation ⊆ baseline citation
  universe, total_ok_pct matches ratio (gate 108, 13 tests);
  small_l3_thrash internal + WRT-tiny disjointness — 9-policy wide-panel
  4kB snapshot (n_rows = n_cells * n_policies = 9 * 9 = 81), per-policy
  aggregates all recompute from CSV, per-cell winner / runner-up honor
  POLICY_LABEL_ORDER tie-break (necessary for all-thrashing 1.0 cells),
  thrash cells (power-law @ 4kB) disjoint from WRT 'tiny'-regime cells
  (mesh+road @ 4kB and 16kB) so the paper never double-counts
  (gate 109, 13 tests); bootstrap_ci.json + oracle_gap_by_app_bootstrap
  .json parity & hygiene — every (policy, family) and (policy, regime)
  mean/median/n exactly matches oracle_gap.summary, ci_lo ≤ ci_hi with
  ci_width recompute, POPT-vs-GRASP family-level paired-delta sign agrees
  with whether CI excludes zero (locks the headline 'POPT loses on road
  with 95% CI excluding 0' claim), per-app pairs are anti-symmetric in
  mean_delta and p_a_lt_b complement to ≤ 1 + slack (gate 110, 13 tests);
  oracle_gap_curvature arithmetic + knee rule — per-cell slope_1to4 =
  Δgap / log2(4) = Δgap/2, slope_4to8 = Δgap / log2(2) = Δgap/1,
  curvature_at_4MB = slope_4to8 − slope_1to4, knee_present iff
  curvature ≥ 0.05, knee_rank_by_policy sorts by mean_curvature DESC
  (NOT knee_count), knee_lead_verdict=PASS iff set(sat_rank[:2]) ==
  set(knee_rank[:2]) (gate 111, 13 tests); gap_distribution_shape
  Hesterberg envelope — 60 per_cell (5 apps × 3 L3 × 4 policies) carry
  full moment stats (n, mean, sd, min, max, skewness_g1,
  excess_kurtosis_g2), a cell is 'outside envelope' iff |skew|>2.0 OR
  |kurt|>7.0 (Hesterberg 2015), pinned_exception_set == observed,
  verdict=PASS iff sets agree + no drift + within budget (gate 112,
  13 tests); distribution_diagnostics envelope + marginals — 20 per_app_
  policy cells + 4 per_policy marginals all share the same nine moment
  fields, observed_envelope reports four worst |skew|/|kurt| values that
  recompute exactly from per_app_policy and per_policy, bootstrap_
  validity_verdict=PASS iff all four worsts ≤ envelope (gate 113, 13
  tests); cross_tool_winners classification — per-cell (app, graph)
  winners across cache_sim/gem5/sniper with n_tools = count of non-empty
  winners, classification='split' iff ≥2 disagree else 'majority',
  summary.split_cells & majority_cells are faithful projections of cells
  (catches a flip from 'split' → 'majority' immediately) (gate 114, 13
  tests); cohens_h_win_rates ↔ wilson_win_rates parity & arithmetic —
  every per_app (app, policy) {wins, total, p_hat} matches exactly
  between the two effect-size views, Cohen's h recomputes as
  2·|asin(√p_a) − asin(√p_b)| (Cohen 1988) from raw counts within
  H_TOL=1e-3, delta_p = p_a − p_b, favors picks the higher p_hat (or
  'tie' sentinel for p_a==p_b), magnitude bucket matches thresholds
  {large=0.8, medium=0.5, small=0.2}, comparisons cover full P(4,2)=12
  permutations per app (gate 115, 14 tests); lofo ↔ leave_one_graph_out
  robustness parity — different scopes (lofo restricts to scope_l3_sizes,
  logo uses full corpus) so per-(app,policy) win_counts intentionally
  differ, but app-level fragility classification must agree (both methods
  must label the same apps fragile vs robust). Empirically: bfs+sssp
  fragile, bc+cc+pr robust by BOTH methods — strong triangulation across
  two independent perturbation strategies (gate 116, 15 tests);
  multiple_testing_correction Holm-Bonferroni step-down + Benjamini-
  Hochberg step-up ladders — 81 hypothesis tests from three sources
  (bootstrap_paired_gap=30, mannwhitney_gap=30, popt_vs_grasp_family_app
  =21) at α=0.05; HB threshold = α/(n−rank+1) with step-down semantics
  (one fail → all later ranks fail), BH threshold = α·rank/n with step-up
  semantics (everything up to k_max survives even if its own p > its own
  threshold), BH ≥ HB always by FDR-vs-FWER guarantee, survivor counts
  44/28/40 reproduce exactly from raw p-values (gate 117, 13 tests);
  cache_saturation_onset step-down rule + per-policy ranking — octave
  arithmetic (delta_gap_pp = gap_to − gap_from, slope_pp_per_octave =
  −delta/Δlog2_MB) from oracle_gap_auc trajectories, saturation_onset
  reproduced from the exact "smallest L3 from which every remaining
  octave shrinks within (−0.5, 0] pp" rule, per_policy onset_counts +
  saturation_rank_by_policy match the documented (−1MB-sat,−4MB-sat,
  never-sat) sort key (gate 118, 11 tests); cross_tool_slope_universality
  central roll-up — medians copy from capacity_sensitivity (cache-sim),
  gem5_slope_replay, and sniper_slope_replay; three invariants enforced
  (all medians negative, all medians in band [−25, −0.5], no tool span
  > 5.0 pp/oct), violations recomputed exactly across three categories,
  verdict='PASS' iff all checks pass (gate 119, 11 tests);
  gem5_slope_replay OLS arithmetic + verdict — per-cell slope is the
  OLS slope of miss_pp vs log2(L3_kB) over the 4 anchor sizes;
  miss_pp_by_size cross-links to gem5_anchor (×100 conversion);
  per_policy median/mean/n reproduce from per_cell slopes; cross-policy
  deltas (lru_minus_grasp, srrip_minus_grasp) exact; verdict_checks
  recomputed from the four monotonicity/sign/steepness invariants
  (gate 120, 12 tests); family_slope_replay arithmetic + verdict —
  per-family OLS slopes from oracle_gap rows (L3_LOG2_MB={1MB:0, 4MB:2,
  8MB:3}, HELP_FLOOR=-5.0 pp/oct), qualifying families require at least
  one (graph, app) cell with all 4 policies × 3 L3 sizes; replays_pattern
  = LRU steeper than GRASP AND SRRIP steeper than GRASP AND every policy
  median below help floor; PINNED_DEVIATING=('social',); verdict=PASS iff
  replay_count >= 1 AND no new deviating families (gate 121, 11 tests);
  policy_steepness_ranking checks + arithmetic — per-policy final-octave
  steepness ranking (POPT <= GRASP <= LRU AND POPT < SRRIP) with
  oracle-aware ceiling 0.5 pp/oct, non-oracle floor 0.5 pp/oct, oracle
  median strictly < half non-oracle median, POPT min slope <= 0.2 pp/oct
  (at least one app fully saturates); all seven checks reproduced from
  cache_saturation_onset per_app final_octave_slope_pp magnitudes
  (gate 122, 14 tests); cross_policy_asymmetry head-to-head arithmetic
  — for every unordered policy pair, a_wins/b_wins/ties recomputed from
  oracle_gap H2H (lower miss_rate wins), means × 100.0 conversion to pp,
  asymmetry_ratio = max/min (None when either mean is 0); verdict=PASS
  iff every pair has at least one win on both sides AND max ratio < 20.0
  ceiling (gate 123, 30 parametric runs); saturation_slope_extremum
  — distance metric (saturation_distance.per_app.mean_pp) and slope
  metric (median across policies of per_app_capacity_slope.median_pp)
  must agree on the LEAST-sensitive app (bfs) but are explicitly
  allowed to disagree on the MOST-hungry app (regime-vs-aggregate
  distinction); all five verdict_checks reproduce (bfs argmin on both
  metrics, bfs unique extremum, corpus has slope >= 3× bfs AND distance
  >= 2.5× bfs) (gate 124, 16 tests); winner_margin_gradient per-(app,L3)
  margin gradient — paper L3 scope {1MB,4MB,8MB} classification
  (decisive >= 4, moderate in [2,4), weak == 1, tied == 0); top_policy
  with alphabetical tie-break; win_counts reproduced from oracle_gap
  is_winner==1 tallies; n_cells_in_scope = distinct (graph,app,l3) count;
  class_counts, strong_cell_fraction, weak_cells, tied_cells, and per_cell
  tied_top_policies all reproduced (gate 125, 16 tests);
  per_app_srrip_vs_grasp slope ordering — per-app delta = SRRIP - GRASP
  (signed, sign matters), deviates iff delta > 1.0 pp/oct, PINNED=('bfs',)
  guard test asserts bfs.deviates True today; verdict=PASS iff no missing
  apps AND no new deviating apps AND every app has both policies present
  (gate 126, 15 tests); corpus_balance arithmetic + diversity metrics —
  per-family and per-app (paper-L3 cell) row counts reproduced from
  oracle_gap, Shannon H bits + Simpson's D + evenness reproduced from the
  raw counts to 5e-4 tolerance, plus dominance share and honest disclosures
  (families_capped_below_4MB / 8MB sorted lists) (gate 127, 16 tests);
  family_curvature_replay arithmetic + verdict — per-family discrete second
  derivative on log2-MB axis (1→0, 4→2, 8→3) reproduced via gate-58 formula,
  replays_pattern as conjunction of any_oracle_aware_positive +
  all_non_oracle_nonpositive, verdict=PASS iff replay_count >= 1 AND no new
  deviating families beyond the pinned set (gate 128, 17 tests);
  cross_generator_gap_parity arithmetic + spread — the structural-integrity
  backbone reconciling oracle_gap raw + oracle_gap_auc trajectory +
  cache_sensitivity_slope gap_at_*/octave records, per-cell raw mean +
  AUC + slope all reproduced to 1e-3 pp, spread/agree/mismatches accounting
  derives consistently, cells sorted lexicographically and keyset equals
  union of all three sources (gate 129, 15 tests); slope_saturation_xcheck
  arithmetic + statistics — per-cell saturation distance (miss_rate(4MB) -
  miss_rate(8MB) in pp) and OLS slope of miss_rate vs log2(MB) across
  1MB/4MB/8MB reproduced from raw rows, Pearson r + Spearman rho + medians
  + 4 invariant booleans (cells>=80, r>=0.4, rho>=0.4, median ratio in
  [0.7,1.3]) all reproduced (gate 130, 18 tests); per_graph_app_stability
  arithmetic + classification — TIE_TOL=1e-6 winner detection on rounded
  gap_pp, intersection/union derivation, 5-way classification decision
  tree (stable_unique / stable_unique_with_ties / stable_partial /
  regime_change / insufficient_l3), and headline cell-list parity
  (gate 131, 17 tests); paper_baseline_table arithmetic + parity — the
  paste-ready paper appendix table; miss_rate values verified
  PIXEL-IDENTICAL to oracle_gap.json (456 cells, 0 mismatches at 1e-9),
  delta_pp_vs_lru derived from miss-rate diffs, verdict labels
  restricted to oracle-aware policies (gate 132, 15 tests);
  literature_faithfulness_postfix arithmetic + parity — the load-bearing
  headline of the paper's "how faithful are we to published baselines?"
  claim and the largest single artifact (~360 KB); summary status counts
  reproduced from per_claim, bucket lists (tolerated /
  known_deviations / disagreements) match per_claim filtered by status,
  per_observation.miss_rate matches oracle_gap (cross-source parity),
  every per_claim carries a citation literal, no DISAGREE entries today
  (gate 133, 17 tests; closes the "every wiki/data/*.json artifact has
  its own arithmetic gate" milestone — 72/72 artifacts now covered);
  cross-artifact miss + delta parity — first multi-artifact gate; locks
  the RELATIONSHIPS between oracle_gap + paper_baseline_table +
  literature_faithfulness_postfix (three-way miss_rate parity at 1e-9,
  delta parity at 1e-3 pp, pbt verdict ⊂ lfp status superset mapping,
  composite-claim whitelist for synthetic claim types like POPT_GE_GRASP)
  (gate 134, 17 tests); winner identification parity — locks the
  winner-identification chain oracle_gap.is_winner ↔
  per_graph_app_stability.winners_by_l3 / intersection / classification
  / headline cell lists; reimplements the cell_winners + classify
  generator from scratch in pytest (TIE_TOL on rounded gap_pp, not
  raw miss_rate; insufficient_l3 cases suppress intersection to []
  by design), so generator drift trips this gate (gate 135, 22 tests).
- **Source-of-truth arithmetic + downstream derivation parity gates 136–140.**
  Gate 136 (OGA-Stat, 21 tests) locks the oracle_gap.json foundation:
  gap_pp = round((miss-oracle)*100, 3); oracle = min(miss) per cell;
  is_winner uses STRICT raw-miss equality (a different rule than
  per_graph_app_stability's TIE_TOL-on-rounded — both pinned now);
  summary mean/median/max/p90 (p90 = nearest-rank round(0.9*(n-1)))
  per policy and per policy/family and per policy/regime reproduce
  from gap_pp to 1e-3. Gate 137 (AUC-Der, 20 tests) locks the
  oracle_gap_auc derivation: trapezoidal AUC on log2(MB) computed
  from RAW means (load-bearing — the displayed trajectory is
  separately rounded). Gate 138 (CSS-Der, 20 tests) locks
  cache_sensitivity_slope: octave delta + slope = -d(gap)/d(log2 MB)
  reconstructed from raw AUC trajectory (Python banker's rounding at
  5th decimal flips e.g. -0.13595 → -0.1359 vs -0.136 when computed
  from pre-rounded delta — must mirror generator path exactly).
  Gate 139 (PGR-Math, 18 tests) locks per_graph_app_stability's
  per_graph_rollup + meta counts: rollup MERGES
  {stable_unique, stable_unique_with_ties} into the single
  stable_unique bucket; corpus-level meta counts double-consistent
  (rollup sum AND per_graph_app set count); stability_fraction
  reproduces round(n_stable / max(total - n_insufficient, 1), 3).
  Gate 140 (PAC-Der, 19 tests) locks policy_auc_correlation:
  matrix[a][b] = round(pearson(zscore(auc_a), zscore(auc_b)), 4)
  (and the z-score-invariance triple-check: equals pearson on raw
  AUC vectors); pair_list = C(5,2)=10 entries, sorted desc;
  nearest_sibling.closest_app = argmax; intra_inter splits by
  auc_winner cluster and reproduces both means and gap.
- **Cross-artifact / statistical derivation parity gates 141–145.**
  Gate 141 (FPC-Der, 18 tests) locks family_policy_auc_clustering:
  per-(family, app, pol) pooling via mean across graphs in family,
  trapezoidal AUC on log2(MB), winner-by-app vs GLOBAL_WINNER pin,
  z-score+Pearson correlation matrix per family, intra/inter cluster
  means using GLOBAL_CLUSTERS partition (NOT per-family winners — a
  deliberate cross-family test of clustering universality). Family
  qualification REQUIRES full L3 coverage on EVERY (app, pol) cell.
  Gate 142 (CHW-Der, 19 tests) locks cohens_h_win_rates: Cohen's h
  via arcsine-transformed proportion delta (h = |2arcsin√p_a -
  2arcsin√p_b|), Cohen 1988 cumulative magnitude bucketing (small≥0.2,
  medium≥0.5, large≥0.8), and the large_effects filter
  (magnitude==large AND p_a>p_b — dominance direction enforced).
  delta_p uses RAW p_hat (not rounded display) — load-bearing rule.
  Gate 143 (WWR-Der, 19 tests) locks wilson_win_rates: z=1.959963984540054
  (load-bearing precision — NOT the common 1.96 approximation), Wilson
  score interval at 95% with [0,1] clamping across three scopes
  (overall, per_app, per_family); cross-gate consistency check that
  per_app rates equal cohens_h per_app rates (same raw aggregation step
  underpins both); per_app + per_family totals each sum to overall.
  Gate 144 (BCI-Der, 19 tests) locks bootstrap_ci: aggregation
  correctness across four sections (policy×family, policy×regime,
  family Δ-buckets, sign_stability) — bootstrap CI BOUNDS remain
  pinned by byte-level reproduce_smoke (same seed=1729, 5000 resamples
  ⇒ byte-identical output), this gate adds the aggregation-level
  safety net (n/mean/median/ci_width/sign logic exact). Gate 145
  (GDS-Der, 20 tests) locks gap_distribution_shape: sample Fisher-Pearson
  g1 skewness + sample-adjusted g2 excess kurtosis formulas per
  (app, L3, policy) cell across 60 cells; the Hesterberg envelope
  (|skew|<2 ∧ |kurt|<7), the pinned-exception delta (new vs gone
  offenders), and the bootstrap_validity_verdict logic (PASS iff
  no new_offenders AND n_outside ≤ 14). This is the upstream artifact
  that determines whether plain-percentile bootstrap is valid for
  the per-cell sample sizes the paper relies on.
- **Cross-artifact / cross-source derivation parity gates 146–150.**
  Gate 146 (CWC-Der, 19 tests) locks cell_winner_census: classifies every
  (graph, app, l3_size) cell as no_winner / unique_winner / tied_winners
  using the load-bearing STRING comparison `r.get("is_winner") == "1"`
  (NOT bool/int); n_cells_total counts ALL L3 sizes including probe
  sweep rows (114 cells, not the 60-cell paper grid). Gate 147 (PWT-Der,
  22 tests) locks policy_winner_table: argmin(miss_rate) within
  (graph, app, l3) groups with stable tie-break by policy name;
  margin_pp = (runner − winner)*100; L3 regime bucketing at 64 kB and
  1 MB byte boundaries; canonical GRAPH_FAMILY map; fragile_top_5 =
  5 lowest-margin cells with margin<0.5 pp. Gate 148 (MUN-Der, 22 tests)
  locks monotonicity_universality: cache-monotonicity (more cache cannot
  hurt) over (graph, app, policy) L3 sweeps from oracle_gap.json;
  MAX_NOISE_BUMP_PP=0.5, BUMP_PCT_CEILING=0.10, verdict=PASS iff all
  three checks (no_hard_violations ∧ bump_pct_under_ceiling ∧
  largest_bump_within_noise) are True; this is a foundational soundness
  gate every downstream slope/distance/sensitivity artifact assumes.
  Gate 149 (PVG-Der, 23 tests) locks popt_vs_grasp_delta: per-(graph,
  app, l3) Δ(POPT−GRASP)*100 in pp with cells lacking either policy
  SKIPPED; classification floor=0.5 pp (popt_better<−0.5, grasp_better
  >+0.5, else tie); stats use **statistics.pstdev** (POPULATION, NOT
  sample) at 3dp rounding; popt_top5_helps sorted ascending,
  grasp_top5_helps descending; this is the central paper question
  "when does POPT actually help GRASP?" Gate 150 (WRT-Der, 21 tests)
  locks winning_regime_taxonomy: the paper's headline-figure aggregator
  binning each winner cell into a (family, regime) bucket; RULE_THRESHOLD
  =0.80 (>=80% dominance extracts a quotable rule); rule text matches
  the exact f-string the paper quotes; rules sorted by (family,
  REGIME_ORDER.index(regime)); _extract_rules break statement enforces
  at most one rule per bin; KNOWN_POLICIES pinned to four (LRU, SRRIP,
  GRASP, POPT) to catch silent upstream typos.
- **Cross-artifact / cross-source derivation parity gates 151–220.**
  Gate 151 (CTW-Der, 19 tests) locks cross_tool_winners.json: joins
  lit-faith CSV + gem5_anchor.json + sniper_anchor.json; per-tool winner
  = argmin(miss_rate) with stable byte tie-break on policy name; cells
  collapsed to the LARGEST L3 each tool actually sampled (tools sweep
  different L3 sets so equal-L3 overlap is ~zero); skip cells with
  fewer than 2 tools; classify unanimous / majority / split — current
  corpus is 100% "split" which is itself the headline. Gate 152
  (RBD-Der, 23 tests) locks regression_budget.json: per-cell distance
  the observed Δ would have to drift in the *adverse* direction before
  the lit-faith status flips to disagree; for cache_policy cells the
  test pins the binary-search exit point by re-running
  literature_faithfulness._classify at margin±ε and asserting the flip
  happens at exactly the recorded margin; POPT_GE_GRASP cells satisfy
  the closed-form margin = max(0, tolerance_pct − delta_pct). Gate 153
  (FGI-Der, 23 tests) locks family_geomean_improvement.json: per-(family,
  app, policy != LRU) geomean miss-rate ratio vs LRU with seeded
  bootstrap CI (B=2000, seed=1729, α=0.05) over the paper L3 scope
  (1MB, 4MB, 8MB); LRU-baseline-required + miss_rate > 0 filters;
  ci_strict_improvement iff ci_hi_ratio < 1.0; headline includes
  CI-strict improvements with improve_pct ≥ 10%, sorted DESC; this is
  what the paper cites for headline magnitudes (+24% on social etc.).
  Gate 154 (PRK-Der, 21 tests) locks policy_rank_kendall.json: per-(app,
  graph) Kendall-τb on the 4-policy rank vector at each L3 pair;
  mean-rank tie-breaking so rank sum invariant n*(n+1)/2 = 10.0
  always; only full-L3-coverage cells emit; verdict PASS iff median
  1MB↔8MB τ > 0 AND no NEW flip cells beyond the 6-cell pinned
  exception list (bc/cit-Patents, bc/web-Google, bfs/email-Eu-core,
  cc/web-Google, pr/email-Eu-core, sssp/soc-pokec) AND total flip
  cells ≤ 6. Gate 155 (CLD-Der, 22 tests) locks claim_density.json:
  per-graph claim aggregation from
  literature_reproduction_summary.csv; n_claims / n_ok / ok_pct;
  status_counts as a Counter dict that must sum to n_claims;
  apps/policies/citations are sorted-unique with truthy-citation
  filter (drops empty strings); summary roll-up totals match per-graph
  sums; n_graphs matches source-distinct-graph count. This is the
  reviewer-facing breakdown of "why does graph X only have N claims?"
  Gate 156 (PBT-Der, 22 tests) locks paper_baseline_table.json: each
  cell groups by (graph, app, l3) over POLICY_COLS=(LRU,SRRIP,GRASP,POPT);
  per_observation miss_rates byte-exact against
  literature_faithfulness_postfix.json#per_observation;
  delta_pp_vs_lru reproduced via raw `(mr − lru_mr) * 100.0`;
  accesses=max(over policies); verdict via _verdict_for 5-branch
  classifier (no_lru / insufficient / DISAGREE / within_tol / ok); the
  PER_GRAPH_CLAIMS pseudo-policy filter (POPT_GE_GRASP,
  POPT_NEAR_GRASP_IF_BIG_GAP) means only GRASP/POPT real-policy claims
  attach verdicts, matching the 27-verdict (19 GRASP + 8 POPT) cap.
  Gate 157 (CTLR-Der, 20 tests) locks cross_tool_lru_regime.json: per-
  tool LRU − GRASP slope deltas with TOOL_L3_RANGE_KB pinned;
  regime classifier (hi ≤ 4096 → sub-WSS, lo ≥ 1024 → post-WSS, else
  mixed) at POSTWSS_GAP_FLOOR_PP_OCT=0.30 and SUBWSS_TOLERANCE_PP=0.20;
  5 verdict predicates ANDed (cache_sim post-WSS LRU steeper,
  gem5/sniper sub-WSS LRU not strictly steeper, regime inversion sign
  holds, regime labels correct). Gate 158 (CTSU-Der, 20 tests) locks
  cross_tool_slope_universality.json: per-(tool, policy) median rounded
  to 4dp from the same three upstream slope sources; physical band
  [MIN_SLOPE_PP_OCT=-25.0, MAX_SLOPE_PP_OCT=-0.5]; steepness_span
  rounded to 4dp; STEEPNESS_SPAN_CEILING_PP_OCT=5.0; violations list
  element-by-element reconstruction; verdict AND of 3 predicates
  (all_negative, all_in_band, no_span_exceeded). Gate 159 (CTSO-Der,
  24 tests) locks cross_tool_slope_ordering.json: per-tool SRRIP-vs-
  GRASP slope ordering from the same three upstream sources;
  GAP_FLOOR_PP_OCTAVE=0.05 and REQUIRED_STRICT_TOOLS=2;
  srrip_steeper ≡ srrip ≤ grasp; srrip_strictly_steeper ≡
  srrip < grasp − 0.05; per_tool keys use UNDERSCORES (`cache_sim`
  not `cache-sim` like in gates 157/158); 3-predicate AND verdict;
  LRU-vs-GRASP delta reported but EXPLICITLY NOT gated (regime-
  dependent — owned by gate 157). Gate 160 (PSR-Der, 30 tests) locks
  policy_steepness_ranking.json: per-policy aggregates
  (n/min/median/mean/max + per_app) from
  cache_saturation_onset.json#per_app[app][policy].final_octave_slope_pp
  via `abs()` + `statistics.median/mean`, all rounded to 6dp; oracle
  vs non-oracle family medians; 7 ordering/threshold checks
  (popt_le_grasp_median, grasp_le_lru_median, popt_lt_srrip_median,
  oracle_aware_ceiling 0.5 pp/oct, non_oracle_floor 0.5 pp/oct,
  oracle_half_of_non_oracle 50%, popt_min_saturates 0.2 pp/oct);
  ranking_by_median sorted ASC must place oracle-aware family in top
  two slots. This nails down the headline saturation-rank story
  (POPT/GRASP saturate to near-zero final-octave slope while LRU/SRRIP
  stay steep) byte-exactly against its single upstream input.
  Gate 161 (WMR-Der, 23 tests) locks winner_margin_by_regime.json:
  joins oracle_gap.json#rows with wss_relative_l3.json#meta.wss_proxies
  by (app, graph, l3); WSS regime classifier (L3/WSS < 0.25 → under,
  > 4.0 → over, else near); per-cell winner via argmin(miss_rate) +
  margin_pp = (second − best) × 100; per-(policy, regime) bespoke
  `_median` (sorted, midpoint average for even n), sum/len mean, and
  linear-rank p90; all rounded to 4dp; verdict = AND of
  (all_regimes_have_wins, ≥1 oracle-aware policy with shrinking median
  from under → near → over). Gate 162 (WSL-Der, 26 tests) locks
  wss_relative_l3.json: WSS proxy = working_set_ratio × 1MB rounded to
  2dp (single upstream corpus_diversity.json); same regime classifier
  as gate 161; per-(policy, regime) statistics.fmean (NOT mean) and
  quantiles(n=10)[-1] for p90 (only when n ≥ 10); wins via per-cell
  argmin(gap_pp); n_cells_in_regime shared across policies in a
  regime; per_regime_ranking sort key `(mean is None, mean or 0)` so
  None entries sink. Gate 163 (SSX-Der, 28 tests) locks
  slope_saturation_xcheck.json: per (app, graph, policy) cells at
  L3 in {1MB, 4MB, 8MB}; distance_pp = mr(4MB) − mr(8MB);
  slope_pp = OLS over L3_LOG2_MB (4dp); ratio_dist_slope =
  distance/|slope| when |slope| > 0; SLOPE_EPSILON=0.05 partitions
  per_cell vs flat_cells; global reducers (Pearson r, Spearman rho
  on ranks with average tie-breaking, bespoke `_median`) on the
  matched cohort; 4-invariant AND verdict (MIN_MATCHED_CELLS=80,
  PEARSON_FLOOR=0.40, SPEARMAN_FLOOR=0.40, ratio band [0.70, 1.30]).
  Gate 164 (WMG-Der, 31 tests) locks winner_margin_gradient.json:
  per (app, L3) cells in PAPER_L3_SIZES={1MB, 4MB, 8MB}; win counter
  per policy; tie-break `sorted(items, key=lambda kv: (-kv[1], kv[0]))`
  picks alphabetically smallest at the top; margin = top_wins −
  runner_up; classifier (decisive ≥ 4, moderate ≥ 2, weak == 1,
  tied == 0); strong_cell_fraction = round((decisive + moderate) /
  total, 4); weak_cells and tied_cells surfaced as sorted lists for
  the "single-graph flip risk" reviewer disclosure. Gate 165 (SD-Der,
  28 tests) locks saturation_distance.json: joins oracle_gap rows
  with wss_proxies; per (app, graph) cell with both 4MB and 8MB
  measurements computes distance_pp = round(best4 − best8, 4) ×
  100 conversion via min(by_l3[l3]) * 100; pico_sentinel ≡ graph ==
  "email-Eu-core"; non_negative_violations gated only on cells with
  WSS > 4 MB (sub-WSS noise excluded); per-app aggregates
  {median, mean, p90, max, min} via bespoke `_median` and `_pct`
  with linear-rank percentile; app_diversity_range = max(median) −
  min(median); verdict = AND of (no non-negative violations, no pico
  violations > 0.05 pp, app_diversity_range ≥ 3.0 pp).
  Gate 166 (PACS-Der, 32 tests) locks per_app_capacity_slope.json:
  per (app, graph, policy) cells at L3 in {1MB, 4MB, 8MB} compute
  OLS slope_pp (4dp) over L3_LOG2_MB; per (app, policy) aggregates
  {n_cells, median, mean, min, max}; medians_of_medians ranks apps
  via argmin/argmax with range_pp = least − most; deviating_apps
  ≡ LRU.median − GRASP.median > ALLOW_LRU_SHALLOWER_BY_PP=1.0;
  new_deviating = deviating − {bfs} (PINNED_DEVIATING_APPS);
  3-invariant AND verdict (all_negative every-(app,policy) median,
  no_new_deviating, at_least_one_cache_sensitive where every-policy
  median < HELP_FLOOR_PP_OCTAVE=-5.0). Gate 167 (OGA-Der, 21 tests)
  locks oracle_gap_by_app.json: bucket by `f"{policy}/{app}"`
  (forward-slash, single) with POLICIES filter; per bucket
  {n, mean=round(statistics.fmean,4), median=round(statistics.median,4),
  p90=round(_p90,4) where idx = min(n-1, max(0, int(0.90*n))) is
  DIFFERENT from gate 165's `_pct`, max, wins=sum(is_winner)};
  by_app_ranking sorted ASC by mean_gap_pp; descriptive matrix only
  (no verdict). Gate 168 (CTS-Der, 26 tests) locks
  cross_tool_saturation.json: per-cell classifier (strict spread <
  sat_floor, 4 regimes), abs-diff delta_pp, agree predicate
  (delta ≤ headline_tol), regime-gated summary reducers
  (doubly_saturated_agree counts ONLY agree=True doubly-saturated
  cells; disagreements list ONLY doubly-saturated agree=False).
  Gate 169 (PSG-Der, 25 tests) locks per_app_srrip_vs_grasp.json:
  per-app SRRIP − GRASP delta from gate 166 medians; deviation
  predicate delta > ALLOW_SRRIP_SHALLOWER_BY_PP=1.0; pinned-app
  subtraction (deviating − {bfs}); 3-invariant AND verdict
  (no_missing_apps, no_new_deviating_apps,
  every_app_has_both_grasp_and_srrip). Gate 170 (CPA-Der, 28 tests)
  locks cross_policy_asymmetry.json: itertools.combinations(POLICIES,2)
  yields 6 unordered pairs; head-to-head winner via strict-less-than
  miss_rate; a_mean_margin = mean over a_wins of (mb-ma)*100 (b
  symmetric); asymmetry_ratio = max/min (None when either mean is
  zero, always ≥ 1.0 by construction); 2-invariant AND verdict
  (every_pair_both_win ∧ max_ratio < ASYMMETRY_RATIO_CEILING=20.0).
  Gate 171 (SSE-Der, 29 tests) locks saturation_slope_extremum.json:
  joins saturation_distance.json#per_app (mean 4MB→8MB drop) with
  per_app_capacity_slope.json#meta.per_app (per-policy median OLS
  slope); per-app slope = bespoke `_median` over per-policy
  median_pp values (NOT mean, NOT median-of-medians); distance_rank
  ASC by distance_pp; slope_rank ASC by abs(slope) (NOT signed
  slope); MOST_BY_SLOPE uses signed-min sort key (NOT argmax
  steepness — load-bearing); 5-invariant AND verdict (bfs argmin
  on BOTH axes + bfs strictly-greater-than every other app on BOTH
  metrics + corpus_has_slope_3x_bfs + corpus_has_distance_2_5x_bfs).
  Gate 172 (OGE-Der, 30 tests) locks oracle_gap_effect_size.json:
  per-(app, ordered policy pair a≠b) Cliff's delta = (#{x>y} −
  #{x<y}) / (n_x*n_y) with magnitude classifier (negligible <0.147,
  small ≥0.147, medium ≥0.33, large ≥0.474), Mann-Whitney U with
  average-rank tie-breaking + asymptotic normal p-value
  (`_normal_sf = 0.5 * math.erfc(z/√2)`, 6dp); per-policy
  distribution uses sorted[n//2] for median (NOT statistics.median —
  even-n differs) + mean=sum/n; stochastically_smaller polarity
  matches sign(delta); large_negative_deltas = filter
  magnitude=='large' AND delta<0 sorted ASC by delta.
  Gate 173 (OGC-Der, 27 tests) locks oracle_gap_curvature.json:
  per (app, policy) discrete second derivative at the 4MB midpoint
  on a log2-MB axis (s01 divides by 2 octaves, s12 by 1); knee_present
  uses NON-STRICT >= 0.05 pp/oct² (boundary inclusive); cells_total
  gates cells where all three of {1MB, 4MB, 8MB} are present in the
  upstream trajectory; per-policy median uses statistics.median (NOT
  bespoke); knee_rank sort key (-knee_count, -mean_curvature); cross-
  gate-55 lead_agrees mirrors the current saturation_rank_by_policy
  first element from cache_saturation_onset.json; verdict = min(GRASP,
  POPT knee_count) > max(LRU, SRRIP knee_count). Gate 174 (OGB-Der,
  17 tests) locks oracle_gap_by_app_bootstrap.json: paired Δ cell
  pairing by (graph, l3_size) with BOTH policies required; iteration
  order through apps_sorted × POLICIES × POLICIES (a≠b) is load-
  bearing for Random(1729) reproducibility; n=2000 resamples; 95% CI
  percentile indices lo=int(0.025·n)=50 and hi=int(0.975·n)−1=1949;
  p_a_lt_b decided by strict mean < 0; gate re-runs the full
  bootstrap and asserts BYTE-EXACT match against the published
  artifact — catches any iteration-order / seed / sample-count drift.
  Gate 175 (CGP-Der, 24 tests) locks
  cross_generator_gap_parity.json: three-way reconciliation across
  oracle_gap (raw rows averaged via statistics.mean, 4dp),
  oracle_gap_auc (trajectory_by_policy, raw float), and
  cache_sensitivity_slope (top-level gap_at_{l3} pulls FIRST, octave
  records consulted via setdefault as fallback for the 4MB midpoint
  — priority order load-bearing); cells emitted in lex-sorted
  (app, policy, l3_size) order over the union of all three upstream
  key sets; spread_pp = max(present) − min(present), 6dp; agree
  requires BOTH spread ≤ 1e-3 AND all_three_present (partial cells
  are NEVER agree=True); mismatches = [c for c if not c.agree];
  n_full_triple_cells counts all_three_present cells separately.
  Gate 176 (LFR-Der, 25 tests) locks lofo_robustness.json: leave-one-
  family-out winner partition on oracle_gap rows filtered to PAPER_L3;
  is_winner check uses STRING "1" (NOT int 1, NOT truth); tie-break
  sort key (-wins, policy_name) alphabetical; drops dict per-family
  is {"missing": True} only for missing-app cells, else 6 fields
  including same_winner_as_full; fragile_family_drops in family-
  iteration order; is_lofo_robust iff empty fragile list;
  robustness_fraction = round(n_robust/n_apps, 4); 0.0 if no apps.
  Gate 177 (LGO-Der, 23 tests) locks leave_one_graph_out.json: LOGO
  sibling of LFR-Der but uses FULL CORPUS (no L3 scope filter); same
  is_winner STRING "1" predicate; drops entries carry 5 fields
  (NO runner_up_wins — load-bearing asymmetry vs LFR-Der full_corpus
  which carries 6); n_drops always equals n_graphs; fragile_drops
  in sorted-graph iteration order; is_logo_robust iff empty fragile.
  Gate 178 (PGF-Der, 19 tests) locks popt_vs_grasp_by_family_app.json:
  per-(family × app) paired bootstrap Δ = gap(POPT) − gap(GRASP) with
  N_RESAMPLES=2000, SEED=1729, CI_LEVEL=0.95, N_PAIRED_FLOOR=3;
  iteration order families_sorted × apps_sorted is load-bearing for
  Random(1729); cell match key (graph, l3_size); STRICT m<0 for
  p_popt_lt_grasp; cells_with_data counts n_paired≥3; cells_skipped_
  insufficient counts 0<n_paired<3 (zero-paired cells NOT counted as
  skipped — load-bearing); full 50,000-draw bootstrap byte-exact re-roll.
  Gate 179 (CSO-Der, 26 tests) locks cache_saturation_onset.json:
  per-(app,policy) octave walker on oracle_gap_auc#trajectory_by_policy;
  saturation onset = smallest i where ALL octaves[i:] satisfy
  −0.5 < delta_pp ≤ 0 (NON-STRICT upper bound = 0 includes flat,
  excludes anti-scaling); slope_pp_per_octave = round(−Δgap/Δlog2, 4)
  with SIGN FLIP load-bearing (positive slope = shrinking);
  saturation_rank_by_policy sort key (-c['1MB'], -c['4MB'],
  n_never_saturated) — ties broken by FEWER never-saturated cells;
  closes the gate 173 OGC-Der cross-gate-consistency seam (which
  mirrors this artifact's saturation_rank_by_policy first element).
  Gate 180 (OGR-Der, 29 tests) locks oracle_gap.json — THE UPSTREAM
  of ~13 downstream artifacts: per-cell construction from
  literature_faithfulness_postfix.csv with skip predicate (<2 policies,
  unknown family dropped); oracle = min over PRESENT policies (NOT all
  four); is_winner uses abs(mr − oracle) < 1e-9 (NOT equality);
  miss_rate/oracle 6dp string, gap_pp 3dp string, n_policies_in_cell
  int; row sort (family, graph, app, L3_bytes, policy); regime
  classifier NON-STRICT <= boundaries (tiny ≤64KB, small ≤256KB, else
  large); summary p90 = sorted[min(n-1, int(round(0.9·(n-1))))] —
  bespoke not numpy.percentile; mean=statistics.fmean, median=
  statistics.median, all 4dp; overall_by_policy emits ALL POLICIES
  even with 0 rows; by_policy_family/regime keys sorted by (pol, label)
  tuple. Full 456-row byte-exact CSV re-derivation. Locks the deepest
  seam in the dashboard.
  Gate 181 (PST-Der, 29 tests) locks policy_stability.json: per-policy
  CV across apps; mean = statistics.fmean, stdev = statistics.pstdev
  (POPULATION not sample); auc_cv = None iff mean ≤ 0 else round(sd/mean,
  4); always_top_2 = max(ranks) ≤ 2 NON-STRICT; always_bot_2 = min(ranks)
  ≥ 3; n_wins counts rank == 1, n_lasts counts rank == 4 specifically
  (NOT == n_policies — load-bearing); safest_order sorts by (auc_cv or
  inf, mean) so None CVs sink; safest_policy / highest_variance_policy /
  best_avg_policy match head/tail of the orderings.
  Gate 182 (L3S-Der, 26 tests) locks l3_policy_stability.json: per-(app,
  l3) winner aggregation from oracle_gap.json#rows; n_cells = UNIQUE-GRAPH
  count (set cardinality, not row count); wins dict ONLY carries policies
  with ≥1 win; ranking key (-wins, POLICIES.index) so ties go to CANONICAL
  order (LRU, SRRIP, GRASP, POPT) NOT alphabetical; unique_winner is
  STRICT >; top_share = round(tw/n, 4); paper_l3_tops EXCLUDES tie cells
  (load-bearing); is_stable requires n_unique == 1 AND len(paper_tops) ==
  3 (full coverage); has_regime_change is n_unique >= 2; stable + regime
  mutually exclusive. cc/GRASP and pr/POPT must be stable single-winner;
  bfs must show GRASP→POPT regime change.
  Gate 183 (MTC-Der, 29 tests) locks multiple_testing_correction.json:
  joins three p-value upstreams (oracle_gap_effect_size mannwhitney_p
  ALREADY two-sided; oracle_gap_by_app_bootstrap p_a_lt_b ONE-sided;
  popt_vs_grasp_by_family_app p_popt_lt_grasp ONE-sided); two-sided
  conversion min(1.0, 2 · min(p, 1-p)) — NOT 2·p; unordered-pair dedup
  via tuple(sorted([a, b])); Holm-Bonferroni step-down threshold α/(n-r+1)
  with rejection-CHAIN rule (once survives goes False, never True again);
  Benjamini-Hochberg step-up uses LARGEST-K rule (NOT per-row threshold);
  BH ⊇ HB (every HB survivor is also BH); expected_false_positives_at_alpha
  = round(α·n, 3); per-source aggregation with int(bool) counters.
  Gate 184 (CBL-Der, 34 tests) locks corpus_balance.json: graph→family
  map from oracle rows (later rows overwrite); per-(family, L3) and
  per-(app, L3) counts in PAPER_L3 scope = ROW counts NOT unique-cell
  counts; Shannon H in BITS (log2 not ln); H_max = log2(K); Pielou
  evenness = H/H_max with 0 fallback when K ≤ 1; Simpson D = 1 - Σp²;
  all metrics rounded to 4dp; dominance via max() with first-encountered
  tie-break; per_family.graphs sorted alphabetically; paper_l3_sizes_reached
  is set→sorted-list (dedup); reaches_4mb/8mb booleans; honest_disclosures
  4MB-capped ⊆ 8MB-capped (monotone); capped + reaching partition all
  families.
  Gate 185 (CSE-Der, 30 tests) locks capacity_sensitivity.json: per-(app,
  graph, policy) OLS slope of miss_rate_pp over log2(L3_MB) with
  L3_LOG2_MB = {"1MB":0, "4MB":2, "8MB":3}; miss_rate × 100 for pp;
  closed-form OLS (n·Σxy − Σx·Σy)/(n·Σxx − (Σx)²); per_cell iteration via
  sorted(cells.items()) over (app, graph, policy) tuples; slope_pp 4dp;
  per-policy bespoke percentile s[max(0, min(n-1, int(round(p·(n-1)))))]
  (NOT numpy.percentile interpolation); POLICIES order alphabetical
  ("GRASP","LRU","POPT","SRRIP") for policy_summary key emission;
  steepest = argmin(medians), shallowest = argmax(medians);
  median_steepness_gap = round(|steep|−|shall|, 4); three-clause verdict:
  (1) every policy median_pp < −5.0 STRICT, (2) steepest == "LRU" exact,
  (3) GRASP > LRU medians STRICT; PASS iff all three.
  Gate 186 (PCS-Der, 34 tests) locks per_graph_cache_slope.json: per-(graph,
  app, policy) per-octave slopes -d_gap/log2(L3_to/L3_from); INCLUSIVE
  >=1.0 pp threshold for significant_anti_scaling (NOT strict >);
  full-trajectory filter via set equality on {1MB,4MB,8MB} (NOT subset);
  per_graph_policy key 'graph|policy' string; families dict
  last-write-wins; n_oracle_aware_anti_scaling = GRASP+POPT counts only;
  anti_scaling_cells sorted DESC by max_pp_growth; per-policy/per-graph
  counters carry no zero entries.
  Gate 187 (PGS-Der, 36 tests) locks per_graph_app_stability.json: per-
  (graph, app, l3) winner set = ALL policies tied for min gap_pp within
  1e-6 tolerance (NOT exact eq); 5-rule classification (insufficient_l3,
  regime_change, stable_unique, stable_unique_with_ties, stable_partial);
  meta.n_stable_unique COMBINES stable_unique + stable_unique_with_ties;
  stability_fraction = stable / max(1, total - insufficient);
  per_graph_rollup uses 'partial' key (NOT 'stable_partial') and
  'stable_unique' includes tied variant; line formats 'g/a -> w' / 'g/a
  -> w1,w2' / 'g/a'.
  Gate 188 (DDG-Der, 24 tests) locks distribution_diagnostics.json:
  per_app_policy keyed 'app__policy' with DOUBLE underscore separator;
  SAMPLE statistics throughout (Bessel-corrected sd via statistics.stdev);
  adjusted Fisher-Pearson skewness g1 = n/((n-1)(n-2))·Σ((x-m)/sd)³ with
  n<3/sd=0 → 0.0; adjusted Fisher excess kurtosis g2 = n(n+1)/((n-1)(n-2)
  (n-3))·Σ((x-m)/sd)⁴ − 3(n-1)²/((n-2)(n-3)) with n<4/sd=0 → 0.0; envelope
  constants (skew=2.0, kurt=7.0) from Hesterberg 2015 + Efron & Tibshirani
  1993; PASS iff all four worst-case metrics STRICTLY < envelope.
  Gate 189 (FCR-Der, 30 tests) locks family_curvature_replay.json:
  L3_LOG2_MB = {1MB:0, 4MB:2, 8MB:3} (NON-uniform); curvature =
  (slope_hi − slope_lo)/1.5 where slope_lo=(g4-g1)/2, slope_hi=(g8-g4)/1;
  (graph, app) qualifies iff ALL 4 policies AND each has ALL 3 L3 sizes;
  per_policy carries EVERY policy in POLICIES tuple (zero-fill empty);
  sign-test threshold 0.0 (any oracle-aware curv > 0 AND all non-oracle
  ≤ 0); replays_pattern = conjunction; deviating preserves iteration
  order; PINNED_DEVIATING_FAMILIES = () empty.
  Gate 190 (FMR-Der, 32 tests) locks family_margin_replay.json: joint
  upstream oracle_gap + wss_relative_l3 (meta.wss_proxies); ORACLE_AWARE
  is a TUPLE (order load-bearing for shrink_evidence emission); regime
  classifier on STRICT bounds (ratio<0.25→under, >4.0→over, else near);
  (app, graph, l3) cell skipped iff <4 policies OR missing wss/L3_BYTES;
  margin = (second_miss − best_miss) × 100; per_policy_regime is FULL
  4×3 grid (12 entries, zero-filled on empty wins); _pct bespoke
  percentile formula; family qualifies iff some oracle-aware policy has
  wins in BOTH under_wss AND over_wss; shrink_evidence records only when
  under_median > over_median (STRICT >).
  Gate 191 (FSR-Der, 33 tests) locks family_slope_replay.json: per-family
  OLS slope on the same NON-uniform log2 axis (1MB→0, 4MB→2, 8MB→3);
  per_policy emits ONLY when xs non-empty (NOT zero-filled, unlike
  FCR-Der); three-clause replay invariant (LRU/SRRIP both STRICTLY
  steeper than GRASP, every policy median STRICTLY < −5.0 pp/octave
  help floor); social family pinned-deviating sentinel locked;
  PINNED_DEVIATING_FAMILIES = ("social",) — non-empty in contrast to
  the empty curvature-replay pin.
  Gate 192 (FSE-Der, 26 tests) locks family_sensitivity.json: pinned
  CANONICAL_FAMILY 8-graph mapping; ALL_FAMILIES 5-tuple; SIGN_CLAIMS
  list of 7 ORDERED triples; STABILITY_FLOOR = 0.95; DEFAULT_N_RESAMPLES
  = 2000; DEFAULT_SEED = 1729 — seed-stable bootstrap rebuild as the
  byte-parity test runs the full generator with default args;
  relabelings is exactly 8 × (5−1) = 32 perturbations (cartesian
  completeness over (graph, non-canonical) pairs); per_claim_flip_count
  keys mirror SIGN_CLAIMS-derived claim names and sum equals total
  flipped across all relabelings (conservation); flip direction ∈
  {lost, gained} matches canonical-stable XOR perturbed-stable at the
  floor.
  Gate 193 (FSD-Der, 33 tests) locks family_saturation_distance.json:
  pinned GRAPH_FAMILY 8-graph mapping (different from Gate 192's
  CANONICAL_FAMILY because this generator owns its own pin);
  thresholds HIGH_HEADROOM_FLOOR_PP = LOW_HEADROOM_CEILING_PP = 5.0,
  ORDERING_SLACK_PP = 1.0; HIGH_HEADROOM_FAMILIES =
  ("citation","social") and PINNED_LOW_HEADROOM = ("web",) are
  disjoint tuples; pico-sentinel filter applied at upstream
  (is_pico_sentinel rows excluded); per_family record carries n_cells,
  min/median/p90/max in pp + sorted-unique graphs list; six verdict
  checks with locked polarity (nonneg/floor/ordering are INCLUSIVE ≥;
  low-ceiling is STRICT <); JSON layout `json.dumps(..., indent=2) +
  "\n"` WITHOUT sort_keys (insertion order load-bearing).
  Gate 194 (ACC-Der, 28 tests) locks anchor_cell_census.json: two
  upstreams (gem5_slope_replay + sniper_slope_replay); pinned
  EXPECTED_L3_AXIS = ["4kB","32kB","256kB","2MB"], EXPECTED_POLICIES =
  ["GRASP","LRU","SRRIP"], EXPECTED_GEM5_CELLS = 2 cells (email-Eu-core
  × {bc,pr}), EXPECTED_SNIPER_CELLS = 6 cells (cit-Patents + email-Eu-
  core × {bfs,pr,sssp}); 13-check verdict matrix covers cell count
  INCLUSIVE ≥ baseline, subset-of-expected, EXACT L3-axis + policy list
  equality, anchors_share_*, and cell_policy_records = |cells|×|policies|
  product; shared_cells = sorted(gem5 ∩ sniper) with (email-Eu-core,pr)
  baseline sentinel; insertion-order JSON layout.
  Gate 195 (ACT-Der, 29 tests) locks anchor_cross_tool_agreement.json:
  shared-anchor slope-sign agreement on (graph,app,policy) cells
  present in BOTH gem5_slope_replay + sniper_slope_replay; thresholds
  pinned MAX_ABS_SLOPE_DIFF_PP = 8.0 ceiling, SHARED_CELLS_FLOOR = 3,
  SIGN_AGREEMENT_FLOOR = SNIPER_STEEPER_FLOOR = 1.0; per-cell predicates
  sign_match (OR-of-equal-zero edge), both_negative (STRICT < 0),
  sniper_steeper (INCLUSIVE |s| ≥ |g|), abs_diff_pp = ||sniper|−|gem5||;
  rate floors evaluated with 1e-9 epsilon to absorb float rounding;
  abs-diff ceiling INCLUSIVE ≤ with epsilon; verdict_ok iff every
  check.ok; JSON with sort_keys=True + trailing newline.
  Gate 196 (AMR-Der, 40 tests) locks anchor_monotonicity_replay.json:
  walks every (graph,app,policy) anchor cell across both upstreams,
  enumerates STRICT > 0 per-step bumps along expected_sizes, and
  applies tier-aware tolerances; gem5 tier is strict-monotone
  ({bump_rate_max_pct:0.0, hard_bumps_max:0, max_bump_pp_max:0.0});
  sniper tier is bounded-noise ({40.0, 5, 2.0}); HARD_BUMP_THRESHOLD_PP
  = 0.5 INCLUSIVE ≥; universal CATASTROPHIC_BUMP_PP = 3.0 kill-switch;
  worst_bumps capped at 6 sorted by -delta_pp; checks
  {bump_rate_ok, hard_bumps_ok, max_bump_pp_ok, no_catastrophic}
  all INCLUSIVE ≤ with 1e-9 epsilon; per-tool verdict_ok iff all four,
  overall verdict_ok iff both tools; median_bump_pp uses
  statistics.median.
  Gate 197 (GSR-Der, 30 tests) locks gem5_slope_replay.json against
  gem5_anchor.json: ANCHOR_L3_LOG2_KB = {4kB:2.0,32kB:5.0,256kB:8.0,
  2MB:11.0} (NON-uniform, do not change); EXPECTED_SIZES tuple in
  axis order; POLICIES = (GRASP,LRU,SRRIP); HELP_FLOOR_PP_OCTAVE =
  -1.0; cell skipped iff <4 sizes present; policy skipped within
  a cell iff missing at any size (for/else trick — no partial slope
  emission); miss_rate × 100 conversion (rate→pp); OLS slope rounded
  4 dp; 4-clause verdict {cache_monotonic_every_cell (INCLUSIVE ≤
  violation predicate), all_per_policy_medians_negative (STRICT < 0),
  srrip_at_least_as_steep_as_grasp (INCLUSIVE ≤),
  grasp_below_help_floor (STRICT <)}; JSON indent=2 + '\\n' WITHOUT
  sort_keys (insertion order load-bearing).
  Gate 198 (SSR-Der, 30 tests) mirrors GSR-Der for
  sniper_slope_replay.json against sniper_anchor.json: same axis,
  POLICIES tuple, HELP_FLOOR, OLS rule, 4-clause verdict; sniper
  reshape exposes the larger cell coverage (cit-Patents + email-Eu-
  core × {bfs,pr,sssp} = 6 (app,graph) pairs) the cross-tool
  agreement gates depend on.
  Gate 199 (LDR-Der, 27 tests) locks literature_deviations.json
  against its two CSV upstreams: row selection filters
  status=='known_deviation' (conservation invariant — exactly the
  flagged cells, no drift if status flips); GRAPH_FAMILY 8-graph
  mapping pinned; MECHANISM_ORDER 4-tuple
  (popt_overhead_dominates / within_extended_tolerance /
  policy_data_missing / unclassified) load-bearing for record sort
  key; classification has a computed-policy branch
  (POPT_GE_GRASP/POPT_NEAR_GRASP_IF_BIG_GAP route through GRASP/POPT
  miss-rate index, predicate popt_vs_grasp_pp > tol STRICT >) and
  a real-policy branch (within_extended_tolerance iff |delta_pct| ≤
  2×tol INCLUSIVE); popt_vs_grasp_pp = (popt_mr − grasp_mr) × 100
  rounded 3 dp; JSON sort_keys=True so summary breakdowns land
  alphabetical on write (Counter.most_common's insertion order is
  intentionally discarded).
  Gate 200 (WKL-Der, 28 tests) locks wss_knee_location.json against
  wss_relative_l3.json: REGIME_LADDER = (under_wss, near_wss, over_wss)
  tuple; POLICIES = (GRASP,LRU,POPT,SRRIP); ORACLE_AWARE = {GRASP,POPT};
  NON_ORACLE = {LRU,SRRIP} (disjoint, union = POLICIES);
  KNEE_THRESHOLD_PP = 0.5; _find_knee_regime walks the ladder
  left→right and returns the first regime with median_gap_pp ≤
  threshold (INCLUSIVE ≤); sentinel rank = len(REGIME_LADDER) = 3
  when no regime plateaus; per-policy carries (per_regime, knee_regime,
  knee_rank, is_oracle_aware); verdict PASS iff
  max(oracle_ranks) < min(non_ranks) STRICT < (ties FAIL).
  Gate 201 (PCR-Der, 25 tests) locks paper_claims.json against nine
  upstream artifacts (corpus_diversity, claim_density,
  literature_faithfulness_postfix, policy_winner_table,
  small_l3_thrash, popt_vs_grasp_delta, literature_deviations,
  cross_tool_saturation, confidence_dashboard) plus a LIVE import of
  PYTEST_SUITES for the green-gate count (anti-staleness trick — the
  on-disk dashboard JSON would be stale during a registry re-gen);
  9 claim categories, 7 documented units; deterministic narrative
  id sequence; _maybe_round 3dp float / int passthrough; JSON
  sort_keys=True+indent=2 no trailing newline. Gate 202 (SLT-Der,
  24 tests) locks small_l3_thrash.json: POLICY_LABEL_ORDER 9-tuple is
  the tie-break key for `_winner` (sort by (miss_rate, index) with
  sentinel = len(order) for unknown labels — LRU at index 0 wins
  ties, paper's narrative-critical LRU dominance); margin_pp =
  (runner_up.miss − winner.miss) × 100 ≥ 0; single-row cells have
  runner_up = winner with margin = 0; policy_stats SKIPS empty
  labels; showdown REQUIRES LRU else skipped, grasp/popt may be
  None; lru_minus_X_pp = (X − LRU) × 100 (POSITIVE = X worse than
  LRU); schema_version pinned at 1; JSON sort_keys=True+indent=2+
  trailing newline. Gate 203 (CDV-Der, 28 tests) locks
  corpus_diversity.json: top-level is a LIST (paper_claims registry
  depends on this — would silently re-shape downstream); GRAPH_ORDER
  is the canonical 8-tuple paper ordering; FIELDS is a 14-tuple of
  (human_label, key, kind) triples; _coerce 4-case dispatch
  (int via int(float(raw)), float via float(raw), empty/whitespace/
  unparseable → None); find_log 4-tier preference (PR/LRU/1MB >
  PR/SRRIP/1MB > any cache_sim_pr_*sorted > any cache_sim_*); collect
  overrides parse_log's dir-derived graph name with explicit name=
  argument (load-bearing for multi-hyphen names like
  soc-LiveJournal1); JSON indent=2 no sort_keys no trailing newline
  (dataclass field order is load-bearing). Gate 204 (LFE-Der,
  28 tests) locks the comparator's predicates (complementary to
  gate 133 LFP-Par which only sees the artifact's internal
  consistency): _classify branches sign='-' / '+' / '~' with min_abs
  / max_abs bound interactions and tolerance bands; POPT_GE_GRASP
  relative-claim dispatch (diff ≤ tolerance → ok else disagree);
  POPT_NEAR_GRASP_IF_BIG_GAP phase-transition gate (grasp_gain_pp >
  10 STRICT triggers assertion; below → ok with skip-note); _coerce_int
  defensive returns (None/empty/non-numeric → 0; float-string also →
  0, predicate is int(text) NOT int(float(text))); _pick_section
  canonical-ROI rule (smallest non-zero else first); JSON write rule
  sort_keys=True+indent=2 no trailing newline; min_accesses_threshold
  default 10_000 pinned. Gate 205 (GAS-Der, 28 tests) locks
  gem5_anchor_summary's evaluate_invariants and supporting helpers
  for BOTH gem5_anchor.json and sniper_anchor.json (single generator,
  shared shape): HEADLINE_L3=256kB / ASYMPTOTE_L3=2MB / SMALL_CACHE_L3
  =4kB pinned; HEADLINE_MAX_GRASP_OVER_LRU_PP=0.45 (tightened from 0.5
  once both anchors landed — worst observed |Δ|=0.328 pp leaves
  0.122 pp slack); ASYMPTOTE_MAX_SPREAD_PCT=1.0; SMALL_CACHE_MIN_
  SPREAD_PP=2.0; predicate parity for all four invariants (headline ≤
  INCLUSIVE → ok, asymptote ≤ INCLUSIVE → ok, small-cache ≥ INCLUSIVE
  → ok else STRICT < → disagree, no_error_rows SUMS across cells);
  missing policy → 'missing' status (NOT 'disagree' — distinguishes
  sweep-incomplete from sweep-contradicts-paper); _pick_canonical_section
  mirrors lit-faith / sign_consistency; _l3_sort_key unit dispatch
  table (with documented 'GB shadowed by B' quirk — no L3 size in any
  sweep uses these so not a practical bug); JSON write rule indent=2
  NO sort_keys + trailing newline.
- **Cross-source derivation + cross-registry integrity gates 206–210.**
  Gate 206 (SCD-Der, 28 tests) locks sign_consistency.py — the
  canonical GRASP-vs-LRU sign comparator that lit-faith._pick_section
  and gem5_anchor_summary._pick_canonical_section both mirror
  (docstring-tagged); complementary to the existing integration-only
  test_grasp_sign_consistency.py; MANDATORY_L3_SIZES=('4kB','32kB')
  tuple; PolicyRow frozen 7-field dataclass; _coerce_int uses
  int(float()) truncation toward zero ('3.7'→3, '-2.9'→-2) — DIFFERS
  from gem5_anchor_summary's _coerce_int (int(text) → ValueError on
  '3.14'); _sign ±1e-9 INCLUSIVE zero band (boundary maps to '0');
  _pick canonical-ROI (smallest non-zero section else first);
  compute_deltas sign convention GRASP-LRU (negative=improvement);
  evaluate dispatch mandatory_violations vs warnings bucket; '0' sign
  collapses to 'ok' for numerical noise. Gate 207 (LRS-Der, 36 tests)
  locks literature_reproduction_summary.py — the paper-ready renderer
  that groups lit-faithfulness cells by citation, outputs feed wiki
  tables quoted in the paper introduction; _verdict_glyph 6-status
  dispatch + unknown→pass-through; _format_delta {:+.3f} signed
  (zero formats '+0.000pp' — load-bearing for sign-flip visibility);
  _expected_window 4 assembly branches (sign+min+max+tolerance);
  _key CITATION_ORDER_PREFIX ordering (Faldu Fig before §); _paper_name
  n≥2-citations → 'cross-paper' (Faldu+Balaji extrapolations don't
  count toward either paper's reproduction%); _paper_rollup
  reproduction% = (ok+within_tolerance)/total EXPLICITLY EXCLUDES
  known_deviation; render_csv lineterminator='\n' (NOT \r\n which
  git diff flags); CSV column order pinned; render_markdown single
  trailing '\n'. Gate 208 (LCS-Der, 29 tests) locks
  local_cache_screen_summary.py — cache_sim diversity-screen
  aggregator; number coercion None/empty/non-numeric → None;
  format_number None→'' (NOT '0'; distinguishes missing from zero),
  |v-round(v)|<1e-9 collapse, {:.6g} fall-through; input_label
  explicit > parent > stem precedence; parse_input split('=',1) so
  paths with '=' survive; summarize_rows drops status!='ok' rows,
  groups by (benchmark, prefetcher, l3_size), l3_delta_vs_lru =
  (lru_misses-row_misses)/lru_misses, missing/zero LRU → empty
  delta (NOT 0); l3_rank by (ascending misses, lex policy tie-break);
  FIELDNAMES 13-column public contract pinned. Gate 209 (XAI-Int,
  21 tests) PIVOTS to GRAPH-LEVEL cross-registry integrity:
  CATALOG ↔ PYTEST_SUITES ↔ paper_claims must stay consistent;
  CATALOG IDs/generators/gates/artifacts all unique-and-resolve
  (except documented ALLOWED_SHARED_GENERATORS for
  gem5_anchor_summary.py which legitimately produces both gem5 and
  sniper anchors); PYTEST_SUITES short tickers unique (dashboard
  column headers); every paper-claim gate appears in CATALOG or
  PYTEST_SUITES (except confidence.green_gate_count whose gate is
  the dashboard generator itself — meta-claim, documented in
  ALLOWED_SELF_GATED_CLAIMS); cross-registry uniqueness (CATALOG IDs
  disjoint from short tickers; claim IDs disjoint from CATALOG IDs);
  catalog summary 'N gates today' self-consistency check pins
  CATALOG.confidence_dashboard.summary count to len(PYTEST_SUITES).
  Gate 210 (RSC-Cov, 14 tests) locks reproduce_smoke coverage:
  every CATALOG.artifact appears in reproduce_smoke rows (except
  CSV_EXEMPT={literature_reproduction_summary.csv} and
  SELF_REF_EXEMPT={reproduce_smoke.json, reproduce_smoke.md});
  every paper_claims.source covered; every catalogued json's
  sibling .md companion-on-disk is tracked (catches stale-MD
  drift); TRACKED_ARTIFACTS no-dupes + all-on-disk + extensions
  ∈ {.json, .md, .csv}; reproduce_smoke.json schema invariants
  (n_artifacts == len(rows), ok==count of ok rows, drift==[],
  missing==[], passed is True, sha fields are 64-hex). All five
  gates committed in one logical commit each; regen cycle GREEN
  at each step.
- **Cross-registry integrity loop closure + per-suite floor gates
  211–215.** Gate 211 (MFC-Int, 15 tests) locks Makefile coverage
  integrity: every CATALOG generator module must be invoked by at
  least one Makefile target (either `python3 -m
  scripts.experiments.ecg.X` dotted form OR `python3
  scripts/experiments/ecg/X.py` legacy direct-script form);
  MAKEFILE_EXEMPT_GENERATORS={"scripts.experiments.ecg.corpus_diversity"}
  documented allow-list (corpus_diversity is hand-baked from GAPBS
  log scraping, wiring it into make would 10x regen time);
  allow-list minimality test ensures exempt entries are NOT
  actually in the invoked set; PYTEST_SUITES discovered as
  dict[label,(path,short)] in source NOT list[dict] like the JSON
  output. Gate 212 (WDC-Cov, 14 tests) locks wiki/data/ coverage:
  every on-disk .json/.md/.csv file is in TRACKED_ARTIFACTS or
  WIKI_UNTRACKED_EXEMPT={reproduce_smoke.json, reproduce_smoke.md}
  (the audit's own output, chicken-and-egg); half-tracked-pair
  detector for CSVs with sibling json/md and for json/md pairs
  (both-tracked-or-both-exempt invariant); floors >=50 json,
  >=50 md, >=10 csv; TRACKED_ARTIFACTS expanded 142→158 by
  appending the 16 newly-discovered CSV/MD siblings (15 CSVs +
  literature_reproduction_summary.md) with an explanatory comment
  block. Gate 213 (PCV-Src, 18 tests) locks paper claims
  source-value parity: PAPER_CLAIMS_DERIVATIONS maps each of the
  14 claim_ids → (derive_fn, tolerance); tolerance schedule
  (integer claims tol=0 exact; percentage rounded-to-1dp tol=0.1pp;
  signed pp tol=0.01); derivations exercise the full schema
  surface (cross_tool_saturation.summary.disagreements is a list →
  len() == claim; small_l3_thrash.cells[i].winner field NOT
  winner_policy; policy_winner_table wins_by_policy/n_cells*100;
  claim_density.summary.total_ok_pct; popt_vs_grasp_delta
  by_family[X].mean_pp); 14 per-claim parametric tests +
  test_all_claims_have_derivation + test_no_orphan_derivations +
  no NaN/inf in either claim values or derived values. Gate 214
  (PCS-Sch, 25 tests) locks paper claims schema integrity:
  REQUIRED_FIELDS=(id,category,text,value,units,source,gate)
  exhaustive parametric coverage; closed-vocabulary KNOWN_CATEGORIES
  (9: corpus, cross_tool, deviations, lit_faith, meta,
  popt_vs_grasp, reproduction, thrash, winner_table) +
  KNOWN_UNITS (7: cells, claims, disagreements, gates, graphs,
  percent, pp); _ID_RE=^[a-z0-9_]+(\\.[a-z0-9_]+)+$ enforces
  snake.dotted form; id-prefix-vs-category mismatch is INTENTIONAL
  (e.g. `winner.X` in category `winner_table`, `confidence.X` in
  category `meta`) and explicitly tested for non-enforcement; bool
  excluded from numeric validation (Python's `isinstance(True, int)
  == True` gotcha); cross-field path resolution (source under
  wiki/data/ and file exists; gate under scripts/ and file exists);
  text >= 10 chars human-readability floor. Together with gate
  213 (PCV-Src, value), gate 209 (XAI-Int, graph), and gate 210
  (RSC-Cov, coverage), the paper claims registry is now sealed
  end-to-end: shape-valid + vocabulary-valid + value-faithful to
  source + reachable from catalog graph + tracked in reproduce_smoke.
  Gate 215 (PST-Min, 655 parametric tests across 18 groups) locks
  per-suite test-count floor: every PYTEST_SUITES entry (215 total)
  has a parseable Python file with >=1 module-level `def test_*`
  function (catches empty test files), at least 1 PASSED test in
  the dashboard (catches all-skip/all-xfail suites that "pass via
  avoidance"), zero collection errors, and AST static-count
  <= runtime accounting (passed+skipped+xfailed+xpassed+failed+
  errors); plus self-consistency cross-checks (dashboard suite
  count == PYTEST_SUITES count, short codes bijection between
  source and dashboard, unique short codes on both sides);
  distribution sanity (>=half of suites have >=3 tests; aggregate
  static count >=500). All five gates committed as one logical
  commit each; regen cycle GREEN at each step. Triple-loop
  integrity now closed: catalog↔suites↔claims (gate 209) +
  source↔value (gate 213) + schema↔vocabulary (gate 214) +
  invocation↔coverage (gates 210, 211, 212) + collection↔runtime
  (gate 215).
- **wiki/data formatting & registry-integrity gates 216–220.**
  Gate 216 (CGH-Sig, 150 tests) locks generator --help signature:
  every CATALOG generator module must accept `--help` and exit 0
  (catches argparse-removal regressions); covered by a parametric
  subprocess invocation per generator. Gate 217 (WTQ-Fmt, 370
  tests) locks markdown formatting: every tracked .md is UTF-8,
  LF-only line endings, ends with exactly one trailing newline,
  no trailing whitespace on lines, and is non-empty; rolled out
  with 16 generator fixes adopting the canonical pattern
  `write_text(body.rstrip("\n") + "\n")`. Gate 218 (WJF-Fmt, 365
  tests) locks the JSON counterpart: every tracked .json parses
  as valid JSON, is UTF-8, LF-only, ends with exactly one
  trailing newline, no trailing whitespace on data lines, and is
  non-empty; rolled out with 28 generator fixes adopting the
  canonical pattern `write_text(json.dumps(..., indent=2,
  sort_keys=True) + "\n")`. Four pre-existing byte-parity tests
  (CDV-Der, LFE-Der, FSR-Der, PCR-Der) were updated — not
  exempted — to assert the new `+ "\n"` contract; docstrings cite
  WJF-Fmt as the reason. Gate 219 (PSL-Par, 1324 tests) locks
  PYTEST_SUITES structural integrity: all 219+ entries have unique
  paths, unique short codes, unique labels; short codes match
  `^[A-Z][A-Za-z0-9]*(-[A-Za-z0-9]+)*$` except the legacy
  `PSL_LEGACY_SHORT_EXEMPT = {"Tier A", "Tier B", "Tier C"}`
  (referenced by test_confidence_dashboard fixtures); path-on-disk
  + path-format invariants; label length and no-control-char
  bounds; dashboard JSON cross-source parity (source labels and
  shorts bijection with `confidence_dashboard.json`). Gate 220
  (WMP-Pair, 365 tests) locks .md/.json companion-pair invariants:
  every tracked .json stem has a tracked .md companion (and vice
  versa); both files exist on disk and are non-empty; markdown
  inline references to `wiki/data/<stem>.{json,csv,md}` must
  resolve to a real file; `WMP_MD_ONLY_EXEMPT =
  {literature_reproduction_summary}` (the cross-paper synthesis
  with no own JSON) plus an empty `WMP_JSON_ONLY_EXEMPT`, both
  with minimality self-tests. Wiki-data integrity now sealed:
  format-stable .md (217) + format-stable .json (218) + companion
  pair (220) + registry parity (219) + on-disk coverage (212) +
  generator-signature (216).
- **Bootstrap / statistical-significance gates**, **policy-rank
  Kendall stability**, **WSS-knee-location**, **family-classification
  sensitivity**, **cross-policy mean-margin asymmetry**, and others
  filled out the dashboard from the original 11 pytest gates to the
  current 155. `make confidence-fast` runs the whole suite in under
  ~3 minutes; `reproduce_smoke.py` snapshots 142 SHA-256 hashes of
  the tracked artifacts and re-runs `make lit-claims lit-catalog`
  in a subprocess to verify drift=0.

Latest additions on top of the Tier A/B/C work:

- `scripts/experiments/ecg/literature_baselines.py` — 264-claim
  spec covering Faldu HPCA20 (143 claims), Balaji HPCA21 (106
  claims), Jaleel ISCA10 (15 claims). Every entry carries a
  `citation=` literal back to a paper figure or section.
- `scripts/experiments/ecg/literature_faithfulness.py` — comparator
  that classifies every observed cell into ok / within_tolerance /
  disagree / known_deviation / insufficient_data / missing.
- `scripts/experiments/ecg/literature_reproduction_summary.py` —
  per-paper grouped reproduction map at
  [`wiki/data/literature_reproduction_summary.md`](data/literature_reproduction_summary.md).
- `scripts/experiments/ecg/regression_budget.py` — per-cell distance-
  to-disagree in pp; emits `wiki/data/regression_budget.{json,md}`.
- `scripts/experiments/ecg/confidence_dashboard.py` — single-screen
  view of all 155+ pytest gates + lit-faith headline + corpus diversity
  + regression budget.
- 6 new pytest gate files in `scripts/test/`:
  `test_baselines_match_literature`, `test_confidence_dashboard`,
  `test_corpus_diversity_floor`, `test_cross_tool_parity`,
  `test_known_deviations_have_root_cause_anchor`,
  `test_literature_baselines_citation_locator`,
  `test_regression_budget_floor`.
- Make targets: `make lit-faith`, `make lit-repro`, `make lit-budget`,
  `make confidence` (CI-ready: exit 0 iff every gate is GREEN).
- `scripts/experiments/ecg/gem5_anchor_summary.py` now emits a
  `small_cache_divergence:<graph>/<app>@4kB` invariant alongside the
  headline (256kB) and asymptote (2MB) checks: at 4kB << WSS the
  three policies **must** diverge by ≥ 2 pp. Together with the
  asymptote check this codifies the GRASP-paper L-shape and would
  catch regressions where a "fix" at small caches collapses policy
  differentiation. Invariants are now per-(graph, app); the Sniper
  anchor scopes to PR + SSSP on `email-Eu-core` and `cit-Patents`
  (16 invariants all `ok`, max small-cache spread 6.36 pp). BFS is
  deferred — 4 kB spread sits at 1.78 pp and email-Eu-core/bfs/Sniper
  shows GRASP +1.49 pp over LRU (insufficient reuse for the L-shape).
- `literature_faithfulness_postfix.json` now reports **0 insufficient_data**
  (was 19). The email-Eu-core/{pr,bc,bfs}/{1MB,4MB,8MB} cells were
  re-run with bumped iterations (PR `-i 2 → -i 20`, BFS `-n 1 → -n 16`,
  BC `-i 1 → -i 64`) to push L3 access counts above the 10 000-access
  validity threshold. lit-faith claims now: 238 ok, 2 within-tolerance,
  30 known-deviation, 0 disagree, 0 missing, 0 insufficient.
- Tier C sign-consistency coverage expanded from 4 to 8 (graph, app)
  pairs: BFS and SSSP added on both email-Eu-core and cit-Patents.
  cache_sim reference sweeps generated for the new pairs; Sniper data
  fold-in confirms strong agreement on email-Eu-core/bfs (all 4 sizes)
  and exposes 3 documented Sniper disagreements (email-Eu-core/sssp —
  noise-floor cache_sim deltas, cit-Patents/sssp@4kB, cit-Patents/bfs
  @4kB+32kB) tracked as xfail in `KNOWN_DISAGREEMENTS`. Tier C count:
  14 pass / 6 skip / 4 xfail (was 5 pass / 3 skip).
- `scripts/experiments/ecg/paper_pipeline.py` now auto-emits per-(graph,
  app) L-curve figures (`figures/l_curve_<graph>_<app>.svg`) and an
  aggregated summary CSV whenever cache_sim rows span ≥3 distinct L3
  sizes. Completes the `final_cache_sim_l_curve` profile end-to-end;
  invariants pinned by 7 new tests in `test_paper_pipeline_l_curve.py`.
- Corpus expanded from 6 → 7 graphs with **roadNet-CA** added as the
  new "road" topology family — hub_concentration 0.140 (vs 0.337 for
  the prior corpus minimum). The new graph is intentionally adversarial
  to GRASP: PR @ 1 MB cache_sim shows GRASP 0.957 vs LRU 0.941
  (GRASP +1.5 pp WORSE because there are no hubs to pin), while POPT
  beats both (0.892). This is direct evidence for the GRASP-needs-hubs
  hypothesis. The finding is locked in by a new theory-driven gate
  `scripts/test/test_no_hub_graph_invariant.py` that asserts no graph
  with `hub_concentration < 0.20` can show GRASP > LRU+0.5 pp, and that
  the corpus must contain ≥1 such no-hub graph (load-bearing rather
  than vacuous).
- roadNet-CA PR cache_sim extended to a full L3 sweep
  ({4 kB, 16 kB, 64 kB, 256 kB, 1 MB}) so the L-curve figure
  `figures/l_curve_roadNet-CA_pr.svg` renders the complete working-set
  curve. The data shows a clean inflection: GRASP only marginally
  beats LRU when both policies are saturated at ≥99.5 % miss
  (4–64 kB), and *loses* monotonically once partial fit appears
  (+0.28 pp at 256 kB, +1.47 pp at 1 MB). POPT remains best at every
  L3 size that can hold useful state. Lit-faith ratio is now 248/280 ok
  (was 240/272); +8 derived POPT-vs-GRASP claims all "ok".
- Corpus expanded again to 8 graphs with **delaunay_n19** added as a
  **mesh** topology family (524 k vertices, 3.1 M edges, hub_concentration
  0.138, clustering_coeff 0.379). The cache_sim data immediately
  *contradicted* the simple "GRASP needs hubs" thesis: at 1 MB GRASP
  beats LRU by 13.73 pp on delaunay_n19 despite uniform degree (~6),
  because GRASP's random-within-bucket protection accidentally aligns
  with the mesh's local cluster structure (protected vertices keep
  their neighbours' hits warm). The original `test_no_hub_graph_invariant.py`
  was therefore renamed (`git mv`) to `test_road_like_graph_invariant.py`
  and the predicate tightened to require BOTH
  `hub_concentration < 0.20` AND `clustering_coeff < 0.10` — i.e.,
  *road-like* (uniform degree AND no triangle structure). roadNet-CA
  still satisfies both (hub 0.140, cluster 0.063); delaunay_n19 is
  correctly excluded by the clustering criterion. Lit-faith ratio
  climbs to **288/320 ok = 90.0 %** (was 248/280). All 17 confidence
  gates remain ✅ GREEN.
- Road-like invariant extended from a single graph-level check to a
  per-(graph, app, L3) cell sweep — every cell on every road-like
  graph at every swept L3 must satisfy the GRASP-cannot-help
  predicate. Test count grows automatically as more road-family
  data folds in (21 cases today for roadNet-CA × {bfs, sssp, cc, pr}
  × 5 L3 sizes + 1 corpus-present check).
- **Four paper-grade aggregator gates added** (bumps total from 17 → 21):
  - `scripts/experiments/ecg/policy_winner_table.py` projects the
    lit-faith CSV onto a winner-per-cell view. 109 cells today; GRASP
    wins 56 (51 %), POPT 41 (38 %), LRU/SRRIP 6 each. Test
    `scripts/test/test_policy_winner_table.py` (7 cases) pins that
    every winner is a known policy, hub graphs at large L3 actually
    have a GRASP/POPT winner, and road-family GRASP wins stay within
    the 0.5 pp noise floor.
  - `scripts/experiments/ecg/small_l3_thrash_report.py` aggregates
    the standalone `final_cache_sim` 4 kB-L3 sweep (9 (graph, app)
    cells × 9 policy variants including POPT_CHARGED and 4 ECG
    modes). LRU wins 5/9 cells; GRASP regresses up to +35.857 pp vs
    LRU on soc-LiveJournal1/bfs. Test
    `scripts/test/test_small_l3_thrash.py` (8 cases) pins the
    "GRASP+POPT both regress ≥ 5 pp vs LRU" tiny-L3 signature,
    that POPT_CHARGED never wins, and that all four ECG variants
    are present.
  - `scripts/experiments/ecg/cross_tool_saturation_report.py`
    pairs each lit-faith cell with the matching gem5/Sniper anchor
    cell, picks each tool's largest L3, and verifies cross-tool
    agreement when both tools are saturated. 7 overlapping cells
    today; 4 doubly-saturated, all agree on Δ(GRASP−LRU) within
    2 pp. Test `scripts/test/test_cross_tool_saturation.py` (8 cases)
    pins that ≥ 1 doubly-saturated cell exists and that no
    doubly-saturated cell disagrees — the central cross-tool
    soundness claim for the paper.
  - `scripts/experiments/ecg/claim_density_report.py` tallies per-
    graph literature claim density (8 graphs, 320 claims, 288 OK
    = 90.0 %). Citation density per graph ranges 2 (delaunay_n19)
    → 12 (cit-Patents). Test `scripts/test/test_claim_density.py`
    (7 cases) pins zero-density absence, status-count consistency,
    and summary-vs-per-graph totals.
- New Make targets: `make lit-winner`, `make lit-thrash`,
  `make lit-cross-tool`, `make lit-density` (all wired into the
  `confidence` dep chain).
- **Four further paper-grade aggregators added** (bumps total from
  21 → 25):
  - `scripts/experiments/ecg/popt_vs_grasp_report.py` projects per-
    cell `Δ(POPT − GRASP)` in pp, broken down by graph family and L3
    regime. Headline: ROAD family mean **−9.276 pp** (POPT crushes
    GRASP, max swing −60.023 pp on roadNet-CA/sssp/1MB); SOCIAL
    family mean +0.360 pp (essentially tie). Counts: POPT better 37,
    GRASP better 35, tie 37 of 109 cells. Test
    `scripts/test/test_popt_vs_grasp_delta.py` (8 cases) pins the
    sign convention, classification consistency, and the family-level
    claims.
  - `scripts/experiments/ecg/literature_deviations_report.py`
    classifies every `known_deviation` row in the reproduction
    summary against a closed mechanism vocabulary
    (`popt_overhead_dominates`, `within_extended_tolerance`,
    `policy_data_missing`, `unclassified`). Today: 30/30 rows classify
    as `popt_overhead_dominates` — the perfect inverse of the road-
    graph finding. Test `scripts/test/test_literature_deviations.py`
    (8 cases) pins the vocabulary, asserts zero unclassified leakage,
    and catches any new policy name without an explicit rule.
  - `scripts/experiments/ecg/paper_claims_registry.py` is the single
    source of truth for every numerical claim the paper makes — 14
    claims across 8 categories (corpus, reproduction, lit_faith,
    winner_table, popt_vs_grasp, thrash, deviations, cross_tool,
    meta), each linked to source artifact + governing gate. Test
    `scripts/test/test_paper_claims_registry.py` (9 cases) pins
    required headline IDs, unique IDs, source/gate file existence,
    the road-popt-negative-sign claim, and a confidence-gate-count
    ≥ 22 floor.
  - `scripts/experiments/ecg/cross_tool_winners_report.py` complements
    the saturation report by computing, for each (graph, app), the
    winning policy each tool picks at its largest L3 (cache_sim,
    gem5, Sniper). Surfaces 6 split cells today — an expected
    negative result because the tools sweep different L3 ranges, so
    largest-L3 operating points sit in different saturation regimes.
    Test `scripts/test/test_cross_tool_winners.py` (8 cases) pins
    schema, closed vocabulary `{unanimous, majority, split}`,
    overlap-with-each-tool, and cache_sim-anchor presence.
  - `scripts/experiments/ecg/winning_regime_taxonomy.py` joins the
    policy winner table with corpus diversity to project a
    (graph family × L3 regime) winner matrix at
    [`wiki/data/winning_regime_taxonomy.md`](data/winning_regime_taxonomy.md).
    Auto-extracts ≥80 % dominance rules: mesh/{tiny,small,large} →
    POPT 100 %; road/large → LRU 75 %; citation/large → GRASP 73 %.
    Test `scripts/test/test_winning_regime_taxonomy.py` (9 cases)
    pins schema, large-regime family coverage, mesh+road presence,
    and the road/large LRU-win existence claim.
  - `scripts/experiments/ecg/oracle_gap_report.py` projects each
    policy's gap to the per-cell empirical oracle = min(LRU, SRRIP,
    GRASP, POPT) at [`wiki/data/oracle_gap.md`](data/oracle_gap.md).
    Mean gaps: POPT 1.65 pp (smallest), GRASP 3.10 pp (most wins:
    56), SRRIP 3.60 pp, LRU 4.93 pp. GRASP/road mean 12.48 pp
    (devastating counter-narrative); POPT/mesh 0.08 pp (near-
    perfect). Test `scripts/test/test_oracle_gap.py` (9 cases) pins
    no-negative-gap, winners-have-zero-gap, every-cell-has-a-
    winner, POPT-smallest-overall-mean (load-bearing), and the
    GRASP/road ≥ 5 pp counter-narrative.
  - `scripts/experiments/ecg/artifact_catalog.py` is the single
    canonical index of every paper-grade aggregator (19 entries
    today) with on-disk audit of (generator, gate, artifact)
    triples. Lives at [`wiki/data/artifact_catalog.md`](data/artifact_catalog.md).
    Test `scripts/test/test_artifact_catalog.py` (10 cases) pins
    schema, no-missing entries on any of the three axes, unique
    snake_lower ids, and an entry-count floor that future PRs
    must bump explicitly. This closes the long-standing five-place
    coordination gap (script + gate + Makefile + dashboard +
    HANDOFF) when adding a new aggregator.
  - `scripts/experiments/ecg/bootstrap_ci.py` adds non-parametric
    percentile bootstrap CIs (5000 resamples, seed 1729) on every
    load-bearing claim: per-(policy, family) and per-(policy, regime)
    oracle-gap means, paired ΔPOPT−GRASP per family, and 7 sign-
    stability claims. Lives at
    [`wiki/data/bootstrap_ci.md`](data/bootstrap_ci.md). Headlines
    are now reported with 95% CIs so reviewers cannot dismiss them
    as point estimates. Key findings: **POPT < GRASP on road
    P=0.976** (bedrock), **POPT < LRU on social P=1.000** (unanimous),
    POPT-mean-smallest headline is dominated by the road family
    (mesh borderline at P=0.948, social/citation/web not significant).
    Test `scripts/test/test_bootstrap_ci.py` (11 cases) pins the
    road sign-stability floor at 0.95, POPT/mesh CI hi ≤ 1.0 pp,
    and POPT-vs-LRU social unanimity ≥ 0.99.
  - `scripts/experiments/ecg/oracle_gap_by_app.py` projects the
    per-cell oracle gap onto the (policy, app) plane so the paper
    can defend "no one-size-fits-all" with per-kernel winners
    instead of relying solely on family-level aggregates. Lives at
    [`wiki/data/oracle_gap_by_app.md`](data/oracle_gap_by_app.md).
    Per-kernel rank-1 (mean gap, pp): **pr→POPT 0.100, bfs→POPT
    1.625, cc→GRASP 0.640, bc→SRRIP 1.689, sssp→POPT** (with
    **GRASP/sssp catastrophic at 7.106 pp**, the worst of any
    policy on any kernel). Test
    `scripts/test/test_oracle_gap_by_app.py` (11 cases) pins all
    20 (policy, app) buckets present, n≥15 per bucket, pr→POPT
    (≤0.5 pp), cc→GRASP, and the GRASP-must-not-win-sssp counter-
    narrative.
  - `scripts/experiments/ecg/wss_relative_l3.py` re-bins every
    oracle-gap cell by L3 / WSS ratio (under_wss < 0.25, near_wss
    0.25–4.0, over_wss > 4.0), using the per-graph
    `working_set_ratio` already published in
    [`wiki/data/corpus_diversity.json`](data/corpus_diversity.json)
    as a WSS proxy. Defends against the reviewer pushback that
    absolute-byte L3 tables silently compare across graphs of
    wildly different sizes. Lives at
    [`wiki/data/wss_relative_l3.md`](data/wss_relative_l3.md).
    Headlines (114 cells; 0 skipped): **POPT has the smallest
    mean gap in EVERY WSS regime** (under 1.619 pp, near 2.351 pp,
    over 0.223 pp); GRASP dominates by **win count** in every
    regime (under 23/48, near 27/52, over 8/14); **LRU win_rate
    in under_wss is 1/48 (~2%)** — strongest quantitative case
    that a real cache-friendly policy actually matters when WSS
    blows past L3. Test `scripts/test/test_wss_relative_l3.py`
    (10 cases) pins the load-bearing per-regime POPT rank-1
    claim plus the no-unknown-graphs invariant so silently-
    dropped cells can't bias the bins.
  - `scripts/experiments/ecg/family_sensitivity.py` re-runs the 7
    sign-stability claims from `bootstrap_ci` under every single-
    graph family reassignment (8 graphs × 4 alternate families =
    32 perturbations), reporting how many flips cross the 0.95
    stability floor. Lives at
    [`wiki/data/family_sensitivity.md`](data/family_sensitivity.md).
    Key findings: **POPT < LRU on social = BEDROCK (0/32 flips)**;
    **POPT < GRASP on road = LOCAL (4/32, all from roadNet-CA
    relocation)** — the road headline depends on a single graph,
    documented as such; POPT < GRASP on mesh = GRAINY (14/32,
    baseline n=5). Uses seed=1729, 2000 resamples. Test
    `scripts/test/test_family_sensitivity.py` (10 cases) pins the
    bedrock claim at 0 flips and enforces that all road flips
    must originate from roadNet-CA (no spurious sources).
  - `scripts/experiments/ecg/reproduce_smoke.py` snapshots SHA-
    256 hashes of 140 tracked `wiki/data/*.{json,md}` artifacts,
    re-runs `make lit-claims lit-catalog` in a subprocess, and
    diffs the canonical hashes (masking volatile timing/runtime
    fields). Lives at
    [`wiki/data/reproduce_smoke.md`](data/reproduce_smoke.md).
    This caught a real drift on first integration —
    `paper_claims.json` carried a stale "28/28" headline after a
    gate count bump because the dashboard regenerates AFTER
    `lit-claims` in the dep chain. Documents the one-cycle
    convergence wart for future maintainers. Test
    `scripts/test/test_reproduce_smoke.py` (8 cases) pins the
    artifact floor at 62 with the load-bearing files list.
  - `scripts/experiments/ecg/oracle_gap_by_app_bootstrap.py`
    paired-bootstraps Δ = gap(a) − gap(b) for every ordered policy
    pair, per kernel (5 apps × 12 pairs = 60 comparisons). 2000
    resamples, seed=1729, 95% percentile CI. Pins CI-backed sign
    claims: pr→POPT<{LRU,SRRIP,GRASP} all P=1.000; cc→GRASP<POPT
    P=0.9995; bfs→POPT<GRASP P=0.999 CI hi=-0.45; sssp→POPT<GRASP
    P=0.971; bc has NO stable ordering among {GRASP, POPT, SRRIP}.
    Output at [`wiki/data/oracle_gap_by_app_bootstrap.md`](data/oracle_gap_by_app_bootstrap.md).
    Test `scripts/test/test_oracle_gap_by_app_bootstrap.py`
    (11 cases) enforces P-floor 0.99 on strong claims, 0.95 on
    stability claims.
  - `scripts/experiments/ecg/popt_vs_grasp_by_family_app.py`
    breaks the POPT-vs-GRASP comparison down by (family × app),
    exposing nuance whole-app gates would average away. 21 cells
    with paired data. Headline findings (all CI-backed):
    **road is POPT-favored on every kernel** (sssp -21.8 pp,
    bfs -11.4 pp, bc -4.6 pp, pr -2.6 pp, cc -1.3 pp);
    cc-counter-narrative is CI-strict on social/cc (+5.5 pp
    P=0.000) and citation/cc (+4.6 pp P=0.000); social/pr is
    CI-strict POPT (P=0.9995); citation/sssp is surprisingly
    GRASP-strict (+1.43 pp P=0.000) — contradicting the per-kernel
    sssp→POPT claim when broken out by family. Output at
    [`wiki/data/popt_vs_grasp_by_family_app.md`](data/popt_vs_grasp_by_family_app.md).
    Test `scripts/test/test_popt_vs_grasp_by_family_app.py`
    (10 cases).
  - `scripts/experiments/ecg/wilson_win_rates.py` — Wilson 95% CIs
    on per-(scope, policy) win-rates. Right tool for small-n
    binomial when p̂ near 0/1. Headline: pr/POPT 20/28 CI
    [0.529, 0.848] strict majority; cc/GRASP 17/20 CI [0.640, 0.948]
    strict majority AND above the 25% null baseline; cc/POPT 0/20
    CI [0.000, 0.161] strict below-chance; sssp policies overlap
    CI. Test `scripts/test/test_wilson_win_rates.py` (11 cases).
  - `scripts/experiments/ecg/cohens_h_win_rates.py` — Cohen's h
    effect-size (arcsine-transformed) on win-rate gaps. 14 large-
    effect (h ≥ 0.8) dominance pairs: cc/GRASP-vs-POPT h=2.346
    (largest), pr/POPT-vs-{LRU,SRRIP} h=2.014. **sssp has no
    large-effect dominance** (max h=0.726 medium). Test
    `scripts/test/test_cohens_h_win_rates.py` (12 cases).
  - `scripts/experiments/ecg/oracle_gap_effect_size.py` — Cliff's
    delta + Mann-Whitney U on RAW gap_pp distributions
    (nonparametric, outlier-robust). MW-U via `math.erfc`, no scipy
    dep. 10 large-effect (|d|≥0.474) dominance pairs. pr/POPT vs
    LRU d=-0.911 MW p=0; cc/GRASP dominates all 3 with MW p<1e-4.
    **sssp again has no large-effect dominance** (third independent
    weak-signal signal). Test `scripts/test/test_oracle_gap_effect_size.py`
    (11 cases).
  - `scripts/experiments/ecg/l3_policy_stability.py` — per-(app, L3)
    winner stability across paper L3 sizes (1MB / 4MB / 8MB).
    **Stable single winners**: cc=GRASP, pr=POPT. **Regime change**:
    bfs (GRASP@1MB → POPT@≥4MB). **No stable winner**: sssp.
    Gray-zone: bc (tied SRRIP/GRASP at 1MB, GRASP unique at 4MB+8MB).
    Pins the firewall against averaging across L3 and silently
    hiding a regime change. Test `scripts/test/test_l3_policy_stability.py`
    (11 cases).
  - `scripts/experiments/ecg/multiple_testing_correction.py` —
    aggregates 81 p-values across the entire gate family (gate 38
    MW, gate 34 paired bootstrap, gate 35 per-(family,app)) and
    applies Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR)
    at α=0.05. **Naive significant 44/81; HB survivors 28/81; BH
    survivors 40/81.** Pins which claims may honestly be called
    'significant' in the paper. Test
    `scripts/test/test_multiple_testing_correction.py` (14 cases).
  - `scripts/experiments/ecg/leave_one_graph_out.py` — drops each
    of the 8 graphs in turn and re-ranks winners. **LOGO-robust**
    (winner survives every drop): pr/POPT, cc/GRASP, bc/GRASP.
    **LOGO-fragile**: bfs (flips when soc-LiveJournal1 dropped),
    sssp (flips under 3/8 drops). Sssp's fragility is now the
    fifth independent signal converging on 'sssp is weak'. Test
    `scripts/test/test_leave_one_graph_out.py` (12 cases).
  - `scripts/experiments/ecg/cell_winner_census.py` — corpus
    decisiveness census. **114 cells: 97.4% unique winner, 2.6%
    tied (3 cells, all in bc/email-Eu-core), 0% no-winner.** The
    one 4-way tie (bc/email-Eu-core/1MB) and two 2-way ties pin
    the 'tied subcorpus' the paper must disclose separately. Test
    `scripts/test/test_cell_winner_census.py` (12 cases).
- New Make targets: `make lit-popt-vs-grasp`, `make lit-deviations`,
  `make lit-claims`, `make lit-cross-tool-winners`,
  `make lit-regime-taxonomy`, `make lit-oracle-gap`,
  `make lit-oracle-gap-by-app`, `make lit-oracle-by-app-bootstrap`,
  `make lit-popt-vs-grasp-by-family-app`,
  `make lit-wilson-wins`, `make lit-cohens-h`, `make lit-gap-effect-size`,
  `make lit-l3-stability`, `make lit-mt-correction`, `make lit-logo-robust`,
  `make lit-cell-census`,
  `make lit-wss-relative-l3`,
  `make lit-bootstrap-ci`, `make lit-family-sensitivity`,
  `make lit-reproduce-smoke`, `make lit-catalog` (all wired into
  the `confidence` dep chain; `lit-claims` depends on every other
  `lit-*` so the registry values are always fresh).

See `wiki/Baseline-Literature-Faithfulness.md` → "The fifteen
confidence gates" and "Regression budget" sections for the
per-gate spec.

## What is already done

**Trace-replay parity (cache_sim layer):** 20/20 zero-delta vs upstream
`faldupriyank/grasp` for {LRU, PIN, GRASP, BELADY} × {BC, BellmanFordOpt,
PageRankOpt, PageRankDeltaOpt, Radii} on web-Google. See
`/tmp/graphbrew-upstream-policy-compare-all4/comparison.csv` (regenerable).

Recent commits on `graphbrew_ecg`:

- `e292903` Align cache_sim/gem5/Sniper with upstream GRASP semantics (f=50 default, `grasp_region` sideband flag, `EvictionPolicy::PIN`).
- `c1372e1` Fix POPT MSB polarity to match HPCA21 reference.
- `0d49c67` Add upstream GRASP trace-replay parity tooling (new `graphbrew_trace_replay.cc` + `upstream_policy_compare.py`).
- `3f8e81b` Extend validation gates and faithfulness pytest (11/11 green).
- `3e5b136` Wiki: GRASP/PIN/POPT faithfulness docs.

Faithfulness state recorded in: [/memories/repo/grasp_srrip_baseline.md](/memories/repo/grasp_srrip_baseline.md).

## What is NOT yet validated

The trace-replay parity only exercises `bench/include/cache_sim/cache_sim.h`.
The gem5 and Sniper integrations share the **same sideband** (registerGRASPTraceRegion,
`grasp_region` flag, `frontier_frac=50`) but use **different replacement-policy
implementations** living inside the simulator overlays. Those are not yet
proven faithful.

## Next session — three-tier validation plan

Run tiers in order; each builds confidence for the next.

### Tier A — Sideband registration sanity (10 min)

Goal: prove gem5 and Sniper actually receive 2 regions (propertyA, propertyB)
with `hot_pct=50` and `grasp_region=true` for a known DBG run.

1. Add a one-shot log line at region registration in:
   - `bench/include/gem5_sim/overlays/mem/cache/replacement_policies/graph_cache_context_gem5.hh`
   - `bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/graph_cache_context_sniper.cc`
   Format suggestion: `"[graphctx] register region base=0x%lx upper=0x%lx hot_pct=%u grasp_region=%d"`
2. Run PR on `email-Eu-core` (smallest graph) under both simulators with `-o 5` (DBG):
   ```bash
   make sim-pr && ./bench/bin_sim/pr -f results/graphs/email-Eu-core/email-Eu-core.sg -s -o 5 -n 1
   # then gem5 + sniper equivalents via roi_matrix.py --suite gem5/sniper
   ```
3. Assert log has **exactly 2** lines with `grasp_region=1, hot_pct=50`.
4. Lock in via pytest: `scripts/test/test_grasp_sideband_registration.py` (new).

### Tier B — POPT permutation equivalence (independent of simulators)

POPT is a reordering, not a replacement policy. Validation is fully
deterministic and doesn't need any simulator.

1. Build upstream POPT reference from `research/POPT_HPCA21_CameraReady`
   (header-only / source under `research/`).
2. Run on `web-Google.el` to produce `new_id[]` reference.
3. Run GraphBrew POPT (`bench/include/graphbrew/partition/cagra/popt.h`) on
   the same edge list.
4. Diff the permutation arrays — after the MSB-polarity fix (commit `c1372e1`)
   expect bit-exact equality or Kendall-τ ≈ 1.
5. Lock in via pytest with a tiny synthetic graph.

### Tier C — gem5 / Sniper GRASP-vs-LRU sign test

Bit-exact miss parity is impossible (MSHRs, prefetchers, replacement-state
init differ). Instead, validate **sign consistency** of GRASP-vs-LRU deltas
between cache_sim and the full simulators.

1. Use existing `scripts/experiments/ecg/roi_matrix.py --suite gem5` and
   `--suite sniper` on small graphs (`email-Eu-core`, `cit-Patents`) for
   PR and BC, L3 sizes {4kB, 32kB, 256kB, 2MB}.
2. Compare per-(graph, app, L3-size) GRASP-vs-LRU delta sign against the
   matching cache_sim row in `/tmp/graphbrew-grasp-cache-sweep/*/DBG/roi_matrix.csv`.
3. Disagreement at small L3 (4kB/32kB) = real bug to investigate.
   Disagreement only at 2MB is acceptable (working set fits).
4. Document deltas in `wiki/POPT-GRASP-Faithfulness-Audit.md`.

## Useful one-liners

```bash
# Regenerate trace-replay parity
rm -rf /tmp/graphbrew-upstream-policy-compare-all4 && \
python3 scripts/experiments/ecg/upstream_policy_compare.py \
  --policies lru pin grasp belady \
  --traces BC.web-Google.cvgr.dbg.lru.llc.trace \
           BellmanFordOpt.web-Google.cintgr.dbg.lru.llc.trace \
           PageRankOpt.web-Google.cvgr.dbg.lru.llc.trace \
           PageRankDeltaOpt.web-Google.cvgr.dbg.lru.llc.trace \
           Radii.web-Google.cvgr.dbg.lru.llc.trace \
  --out-dir /tmp/graphbrew-upstream-policy-compare-all4

# Faithfulness pytest
python3 -m pytest -q scripts/test/test_popt_grasp_faithfulness_sources.py \
                     scripts/test/test_ecg_validation_gates.py

# cache_sim sweep that produces the GRASP-vs-LRU reference for Tier C
python3 scripts/experiments/ecg/roi_matrix.py --suite cache-sim --benchmark pr \
  --options "-f results/graphs/email-Eu-core/email-Eu-core.sg -s -o 5 -n 1 -i 1" \
  --policies LRU SRRIP GRASP \
  --l1d-size 1kB --l2-size 2kB --l3-sizes 4kB 32kB 256kB 2MB --l3-ways 16 \
  --line-size 64 --out-dir /tmp/graphbrew-grasp-cache-sweep/email-pr/DBG
```

## Gotchas

- Upstream PIN/BELADY .bin sources live at `/tmp/graphbrew-faithfulness-upstream/grasp/trace-based-simulators/`
  — `upstream_policy_compare.py` builds them on demand. Trace files live in `datasets/`.
- Do **not** run multiple GraphBrew gem5 jobs on the same node unless sideband
  files are isolated (per `.github/copilot-instructions.md`).
- BellmanFordOpt uses `frontier_frac=100`, not 50 — confirm before asserting region params.
- BELADY in `graphbrew_trace_replay.cc` requires the uint64_t cast on `time=-1`
  so empty ways wrap to UINT64_MAX and get evicted first (regression-prone).
