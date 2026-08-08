# ECG Successor: Normative Paper SSOT

**This file is normative.** It is the sole scientific source of truth for the
ECG cache-replacement/placement architecture. Every claim below is either a
stated invariant/design decision or a number that links to
[`RESULTS.md`](RESULTS.md) (current measured tables) or
[`evidence/current/`](evidence/current/) (compact evidence notes). No other
document in this repository may assert a competing normative claim; if a
comment, README, or wiki page disagrees with this file, this file wins.
Reproduction commands live in [`ARTIFACT.md`](ARTIFACT.md), not here.

## 1. Thesis and scope

Graph analytics already stream an edge record before dereferencing an
irregular vertex property. K2 carries a compact future-reuse contract from
that edge onto the exact property Request, exposing graph semantics
unavailable to PC/address predictors without an eviction-time
rereference-matrix lookup. The design keeps the full configured LLC data
ways, discloses a modest side-metadata overhead, and requires no live P-OPT
matrix at replacement time.

The current scientific scope is intentionally narrow: a completed gem5 O3
PageRank screen over three sampled graphs, a degree-1 common-prefetcher
sensitivity, and an instruction-count disclosure obligation. Broader
full-graph, all-kernel, and Sniper-timing claims are **not yet made**; see
Section 6 (current evaluation plan) and Section 9 (limitations).

## 2. Prior publication

The preliminary paper is:

> A. T. Mughrabi, M. Baradaran, A. Samara, and K. Skadron, "ECG: Expressing
> Locality and Prefetching for Optimal Caching in Graph Structures," IEEE
> IPDPSW 2024, pp. 520-525, DOI 10.1109/IPDPSW59749.2024.00094.

This is an archival IEEE workshop publication. An HPCA submission built on it
must be materially different, and eligibility is not established by renaming
alone. Before registration:

1. send the PC chairs the workshop paper and a contribution-delta summary
   (Section 2.1);
2. cite the workshop paper in third person in the submission;
3. ensure title/abstract describe the new architecture, not the workshop
   mask/prefetch prototype;
4. retain written chair guidance with the artifact records.

### 2.1 Contribution delta

| IPDPSW 2024 ECG | HPCA successor |
|---|---|
| single metadata mask concept | K2 two-future-epoch records with a carried tier |
| preliminary replacement/prefetch study | static/online replacement plus StreamShield placement |
| basic trace-driven evaluation | cache_sim + gem5 + Sniper implementations sharing one victim-decision SSOT |
| conceptual graph instruction | executable computed-address masked property load (K2-M) |
| no complete overhead attribution | explicit instruction-count disclosure and traffic/time reporting together |
| PageRank-focused | PageRank complete-design screen now; BFS/SSSP/BC/CC extension scoped in Section 6 |

### 2.2 Chair-query draft

Use this draft when contacting the HPCA program chairs; attach the IPDPSW PDF,
the current abstract, a one-page old-vs-new contribution table, and the
current evaluation matrix (Section 6), plus an explicit list of reused text,
figures, and implementation components.

> Subject: HPCA prior-publication eligibility question: archival IPDPSW
> workshop paper.
>
> We are preparing a submission that builds on our six-page archival workshop
> paper (citation above). The proposed paper introduces a materially
> different architecture and evaluation: two-future-epoch K2 records rather
> than the workshop's single mask; StreamShield request-bound LLC
> placement/bypass; an executable masked property-load ISA; and real-graph
> timing, traffic, and instruction-count accounting against LRU, GRASP, and an
> optimistic charged P-OPT bound. We will cite the IPDPSW paper and describe
> the differences in the paper and submission form. Could you advise whether
> this contribution delta is eligible under the concurrent-submission and
> workshop policy?

## 3. Mechanism and encoding

### 3.1 K2 record

The canonical record carries a destination/property identifier, a 2-bit
carried tier, and two quantized future-rereference epoch stamps. The general
64-bit Schedule-2 layout is:

```text
63                 49 48                 34 33   32 31                  0
+---------------------+---------------------+-------+---------------------+
| epoch2 (15 bits)    | epoch1 (15 bits)    | tier  | destination (32)    |
+---------------------+---------------------+-------+---------------------+
```

`tier` is `1/2/3` for hot/moderate/cold (zero is invalid and never stamps
replacement metadata). The tier is computed once per property line as the
**hottest tier among all vertices sharing that line**, so it stays meaningful
without DBG physical reordering. `epoch1`/`epoch2` are the next two quantized
rereference epochs for the governed property line, constructed before the ROI
and streamed with the edge.

For the completed screen (Section 8), the record is **compressed to 4 bytes**:
destination id, 2 tier bits, and two epoch stamps sized to fit the remaining
budget. The record-scaling constraint is:

```text
id_bits + 2 + 2 * epoch_bits <= 32
```

The completed screen's sampled graphs use 16/16/18-bit identifiers, so all
three fit 5-bit epoch stamps and execute with 32 epochs. For the corresponding
three local **full graphs**, the maximum 4-byte resolutions are:

| Graph | Vertex id bits | Max epoch bits (4-byte fit) | Epochs |
|---|---:|---:|---:|
| web-Google | 20 | 5 | 32 |
| soc-pokec | 21 | 4 | 16 |
| cit-Patents | 22 | 4 | 16 |

These are the **only** epoch counts a 4-byte record supports for these three
graphs (32/16/16). Do not claim a "4-8 epoch" range for local full-graph work;
that range does not correspond to any admissible 4-byte configuration at these
graphs' vertex counts. Where a 4-byte record cannot carry the required
resolution, the general 8-byte layout (`dest32 | tier2 | epoch1_15 |
epoch2_15`) is used instead; it is unconditionally sufficient for all three
local graphs and is the mechanism's general-purpose fallback.

### 3.2 Property load and replacement

K2-M is a computed-address masked property load: it receives an
already-computed address and replaces an ordinary property load one-for-one,
carrying the record's tier/epoch mask on the exact governed Request. This is
the canonical ISA contribution. All simulators call one shared victim
selector so replacement decisions are architecturally identical across
cache_sim, gem5, and Sniper (delivery mechanics differ; see Section 4).

**The static primary for the completed screen is explicit
`rrip_first`+StreamShield** (`ECG:K2_RRIP_STREAMSHIELD`). A prior configuration
using `epoch_first` as PageRank's static primary was tried, failed its
frozen guards on the first cell, and is retired; it is not a current design
and must not be described as normative anywhere in this repository. Do not
restate a per-kernel "PR=epoch_first / BFS=degree_first / adaptive" mapping as
the current design: that wording described the retired configuration.

**Online K2 (`ECG:K2_ONLINE_STREAMSHIELD`) is characterization only** in the
completed screen: five-arm set dueling (RRIP-, GRASP-, epoch-, degree-, and
LRU-first) samples leader/follower arms at runtime without reading the
benchmark or graph name, but it does not control the screen's stop/go
decision; only the static RRIP+StreamShield candidate does.

**StreamShield** is a one-touch, request-bound placement control: record
loads that miss the LLC are returned without allocating, while ordinary L1/L2
fills and existing LLC hits are unaffected.

## 4. Simulator roles

Three simulators are used, and each has a fixed, non-overlapping role. No
document may claim a role reassignment without updating this section first.

| Simulator | Role | Timing claims |
|---|---|---|
| **gem5 (O3)** | Sole architectural timing authority | Yes -- the only simulator whose execution time may support a speedup claim |
| **cache_sim** | Full-graph functional/replacement/traffic authority | No -- reports no cycles, no instructions, no IPC; scores only cache/traffic direction |
| **Sniper** | Scale and direction corroboration | No -- never supports a K2-M architectural speedup claim |

cache_sim assumes the K2 record load costs nothing extra in instruction
count; that assumption is optimistic and is exactly why gem5, not cache_sim,
is the timing authority. Sniper shares the victim-selector SSOT, per-policy
variant receipts, realized-cache checks, and independently compiled P-OPT
lookup parity tests, so its cache/traffic direction is comparable in kind.
Sniper timing remains corroboration of scale and direction, never an
architectural K2-M speedup number, because Sniper models the mechanism at a
coarser transport granularity than gem5's per-Request O3 binding. Absolute
miss rates are never compared across simulators; only each simulator's own
ratio-to-its-own-LRU is comparable.

## 5. Metrics and claim rules

**Primary metrics**, always reported together:

1. execution time (from gem5 O3, the timing authority), and
2. total off-chip traffic (memory-controller bytes read plus written,
   including demand, prefetch, metadata, and writeback traffic).

**Aggregation** is the geometric mean of per-cell ratios over the frozen cell
set; win/tie/median counts may supplement but never replace it.

**Tie band** is +/-2% on a per-cell ratio.

**Comparison scope**: a ratio is formed inside a single simulator invocation
wherever the design allows it, because cross-run cells have been observed to
differ by up to ~1.7% at identical configuration; every invocation needed for
a genuine cross-run contrast must carry its own baseline cell.

**Within-build comparison**: one table may not mix rows from different guest
or simulator builds. Re-run the local baseline rather than importing it from
another build, because unrelated code layout has changed PageRank instruction
count materially even when the selected source path was unchanged.

**Admissibility of a mechanism**: a cell the runner marks
`timing_valid_for_speedup=0` may contribute traffic/instruction evidence but
never a speedup claim, regardless of how favorable its time looks.

**Prefetch interpretation**: with a prefetcher active, demand-miss reduction is
not performance evidence by itself because prefetching can convert demand
misses into prefetch traffic rather than remove work. Target time and total
off-chip traffic remain the comparative metrics.

**Idealized mechanisms**: a mechanism that cannot mispredict or has no finite
latency, bandwidth, queue, or MSHR backpressure is an upper bound, not a
measured performance result. Any permitted one-sided sensitivity must identify
which policy it favors and which conclusion is admissible.

**Symmetric overhead accounting**: if one policy's metadata stream is
simulated through the cache hierarchy, every competitor's metadata or
reference-structure stream must be too, with identical prefetch eligibility
and identical MSHR/queue/latency/bandwidth treatment. The completed screen does
not satisfy the target-time portion of this rule for P-OPT: it charges reserved
LLC capacity and cumulative matrix-stream bytes, but
`popt_target_time_charged=0` omits matrix-stream latency. Its time is therefore
an **optimistic charged P-OPT bound**, not a measured target-time P-OPT result.
Failing to beat this bound satisfies the preregistered STOP rule but does not
establish that a realistic P-OPT engine is faster.

**Metric stability**: the primary metrics (time, traffic) may not be changed
after seeing results; a new metric may be added as secondary, but the
headline comparison stays fixed. This guards against selecting a favorable
metric post hoc.

**IPC is not independent evidence**: IPC ratio is algebraically the
instruction ratio divided by the time ratio, so reporting both is reporting
one number twice. IPC may not be reintroduced as "corroborating" evidence for
a timing claim under a softer label.

**Counterfactual normalization** (e.g. dividing out instructions a future
hardware instruction would remove) is a sensitivity study, never a measured
result, and must be labeled as such wherever it appears.

### 5.1 Instruction-count disclosure rule (not a strict-parity guard)

K2 retired fewer instructions than conventional policies in the completed
screen (Section 8.2; see
[`evidence/current/instruction_decomposition.md`](evidence/current/instruction_decomposition.md)
for the exact ratios). The rule governing this is a **disclosure and
claim-classification rule**, not a requirement that instruction counts must
match before a comparison is admissible:

- Instruction inequality is **allowed** for a claim scoped to the
  **complete design** (record layout, transport, ISA, StreamShield, and
  replacement together) versus conventional baselines (LRU, GRASP, P-OPT).
  Such a claim must disclose the measured instruction ratio alongside time
  and traffic.
- Instruction inequality **prohibits** a "replacement-policy-alone" claim
  versus those same conventional baselines, because part of any measured
  delta could come from the record/ISA path rather than the eviction rule.
- The clean, parity-controlled replacement-only attribution is internal to
  the K2 family: **K2-RRIP+StreamShield versus K2-LRU+StreamShield**, which
  share the same record stream, delivery path, ISA, and instruction count.
  Only this comparison may be described as isolating replacement-policy
  effect.
- A claim must state which of these two classes it belongs to. An
  undifferentiated "K2 is faster" statement that mixes complete-design and
  replacement-only evidence is not admissible.
- Before promoting a complete-design speedup claim, the observed K2 versus
  conventional instruction delta must be classified as intended record/layout
  work, intended K2-M work, or asymmetric compiler/control-flow specialization.
  An unexplained delta leaves the claim pending; measured time is never divided
  by instruction count to manufacture parity.

## 6. Literature-backed baselines and scope

- The P-OPT paper (Balaji et al., HPCA'21) evaluates PageRank, Connected
  Components, PageRank-Delta, Radii, and MIS. **It does not evaluate BFS or
  SSSP.** For BFS/SSSP, any P-OPT comparison in this project is a **project
  extension**, not an author-evaluated baseline, and must be labeled as such
  everywhere it appears.
- P-OPT's own direct comparison against GRASP is **PageRank on DBG-ordered
  graphs**, with primary hardware prefetching disabled. This project's
  completed screen matches that configuration (DBG order via `-o 5`, no
  prefetcher) for its P-OPT/GRASP contrast.
- **Reordering benefit is input-dependent, and preprocessing cost is real.**
  DBG reordering is not free, and its benefit varies by graph. The completed
  screen used DBG order (`-o 5`) in every cell, so **K2's no-DBG-preprocessing
  advantage is untested** in that screen; no claim of a no-reordering
  advantage may be made from it. Natural-order PageRank sensitivity is planned
  (Section 7) specifically to test this.

## 7. Current evaluation plan

The plan below is scoped to data and hardware presently available; no cell
claims a graph or simulator that is not on hand.

**gem5 (bounded timing, architectural authority).**
- Current: three deterministic PageRank samples (web-Google-n16,
  soc-pokec-n16, cit-Patents-n18-sym) crossed with iterations 1/2/4/8, seven
  policies (LRU, GRASP, optimistic charged P-OPT bound, uncharged P-OPT,
  K2-LRU+StreamShield, static K2-RRIP+StreamShield, online
  K2+StreamShield) -- the completed screen (Section 8).
- Planned: natural-order (non-DBG) PageRank sensitivity, to test the reordering
  dependence noted in Section 6.

**cache_sim (full-graph functional/traffic authority).**
- Full graphs currently available locally: web-Google, soc-pokec,
  cit-Patents.
- PR and CC comparisons include the optimistic charged P-OPT bound (both are
  P-OPT-paper-evaluated kernels).
- BFS and SSSP primary baselines **exclude P-OPT**, or explicitly label any
  P-OPT row as a project extension (Section 6).
- BC is an appendix-only kernel: reported for completeness, not part of the
  headline claim set.

**Sniper (scale/direction corroboration).**
- Same three full graphs as cache_sim, run with equal semantic-edge visit
  limits across policies so cross-policy comparison is apples-to-apples.
- Never supports an architectural K2-M speedup claim (Section 4); reported as
  scale/direction corroboration only.

**Future acquisition (not currently available; do not describe as current).**
- UK-02 and/or URAND, to obtain graphs large enough for direct P-OPT
  comparability at scale.
- A road-network graph, planned later, for a topologically distinct negative
  control.

## 8. Current STOP interpretation

The completed no-prefetch gem5 O3 PageRank screen is 12 cells / 84 `ok` rows
across the three sample graphs and iterations 1/2/4/8, all seven policies.
Full detail: [`evidence/current/pr_screen_stop.md`](evidence/current/pr_screen_stop.md).

The preregistered static primary, K2-RRIP+StreamShield, versus:

| Baseline | Time ratio | Traffic ratio |
|---|---:|---:|
| LRU | 0.9061 | 0.7227 |
| GRASP | 0.9835 | 0.9455 |
| optimistic charged P-OPT bound | 1.0235 | 0.9351 |

**Result: STOP.** The screen is valid (baselines behave sanely, no masking
guard triggered), but static K2 misses the frozen aggregate time threshold
versus GRASP and the optimistic charged P-OPT bound.
`cit-Patents-n18-sym` is the decisive negative graph (Section 8.3). This is a
**complete-design comparison**: it beats LRU substantially, is within
tie-adjacent range of GRASP, and fails to beat the deliberately favorable
P-OPT bound. This does not rank K2 against a realistic target-time P-OPT
engine. No full-graph or all-kernel claim follows from a 12-cell sampled
screen; a STOP here ends this specific screen, not the architecture.

### 8.1 Degree-1 common-prefetcher sensitivity

A separate sensitivity applies the same finite-resource gem5 L2 stride
prefetcher (degree 1) to every policy, using the authoritative soc-pokec
`_restart_` run directory (the two-policy resume directory and the
incomplete pre-reboot directory must not be mixed in). The optimistic charged
P-OPT bound uses `analytic_prefetch_upper_bound`: reserved capacity and matrix
bytes remain charged, but the analytic matrix stream is deliberately given perfect
latency-hiding and no over-fetch, i.e. this favors P-OPT. Full detail:
[`evidence/current/prefetch_d1.md`](evidence/current/prefetch_d1.md).

Static K2-RRIP+StreamShield geomean:

| vs | Time ratio | Traffic ratio |
|---|---:|---:|
| LRU | 0.8636 | 0.7151 |
| GRASP | 0.9799 | 0.9153 |
| favored P-OPT bound | 1.0694 | 0.9794 |

This sensitivity **confirms, rather than rescues**, the primary STOP: K2
strongly beats LRU, roughly ties GRASP, and still trails the deliberately
favored P-OPT bound. `cit-Patents-n18-sym` remains the negative case under
prefetching too.

### 8.2 Instruction-count disclosure

K2/LRU retired-instruction ratios in the completed screen:

| Graph | Ratio |
|---|---:|
| web-Google-n16 | 0.8546 |
| soc-pokec-n16 | 0.8144 |
| cit-Patents-n18-sym | 0.9120 |
| **geomean** | **0.8594** |

Per the disclosure rule in Section 5.1, this inequality does not invalidate
the complete-design STOP above, but it does mean the STOP result cannot be
attributed to replacement policy alone. The parity-controlled internal
comparison, K2-RRIP+StreamShield versus K2-LRU+StreamShield (same record
stream, ISA, and instruction count), reports aggregate time ratio 0.9330 and
traffic ratio 0.7855 -- i.e. the replacement rule itself contributes a real,
separately-attributable improvement over K2's own LRU arm. Full detail:
[`evidence/current/instruction_decomposition.md`](evidence/current/instruction_decomposition.md).

### 8.3 cit-Patents as a first-class negative control

`cit-Patents-n18-sym` drives the screen's failures in both the no-prefetch
screen and the degree-1 sensitivity, and it is retained deliberately as a
**first-class negative/control graph**, not excluded or reweighted. Mechanistic
diagnosis of why it fails (e.g. sparse/low-overlap structure) is permitted, but
**thresholds or policy behavior must not be tuned to fix cit-Patents post hoc**.
Any design change motivated by this diagnosis becomes a new preregistration
generation and must be evaluated across the full frozen graph roster, not just
the graph that motivated it.

## 9. Limitations and open questions

- **Reordering/preprocessing.** DBG order was used in every completed-screen
  cell (`-o 5`); the benefit is input-dependent and the preprocessing cost is
  real. K2's hypothetical no-DBG advantage is unestablished.
- **Record scaling ceiling.** The 4-byte record's epoch budget is fixed by
  each graph's vertex-id width (Section 3.1); larger future graphs may force
  the general 8-byte layout, which changes the transport-cost profile
  measured in the current screen.
- **Instruction attribution.** See Section 5.1/8.2 -- unequal instruction
  counts are disclosed, not hidden, but they bound which claims are
  admissible.
- **BFS/SSSP versus P-OPT.** Any such comparison is a project extension
  beyond the P-OPT paper's own evaluated kernel set (Section 6) and must be
  labeled as such wherever it appears.
- **Sniper timing.** Never an architectural speedup authority (Section 4);
  full-graph Sniper corroboration at equal semantic-edge limits is planned
  but not yet complete.
- **Scale.** UK-02/URAND/road-network graphs are not yet available locally;
  claims are bounded to web-Google/soc-pokec/cit-Patents until they are
  acquired (Section 6).
- **Literature corpus.** Related-work classification of external baselines
  (P-OPT, GRASP, DROPLET, Hawkeye, SRRIP) is summarized here and in evaluation
  rationale rather than copied into the repository. Unrelated external
  material (a separate Gorder-focused agent's logs, physics/FLAIRS work) is
  out of scope. Bibliography consolidation remains a paper-authoring task; an
  alternate sync file elsewhere in the broader workspace omits active ECG
  citations and must not be substituted without a citation-coverage check.

## 10. Hardware and accounting policy

- K2 reserves **zero** LLC data ways; its cost is disclosed line/request
  metadata, never a data-capacity reservation.
- The minimum line state is 33 logical bits (two 15-bit epochs, 2-bit tier,
  valid). The configured contextual point is 49 logical bits per 512-bit data
  line after adding a 16-bit context id: 9.57% of data bits, or 1.531
  16-way baseline-way equivalents. The corresponding fractional equal-area
  capacity is 14.602 ways; 15-way and conservative 14-way rows are therefore
  sensitivities, not the primary implementation.
- The optimistic charged P-OPT bound is charged its reserved LLC capacity and
  cumulative matrix-stream bytes, while target-time stream latency remains
  omitted as disclosed in Section 5.
- The general (non-4-byte) K2 record layout is 64 bits:
  `dest32 | tier2 | epoch1_15 | epoch2_15`, sufficient for all three local
  full graphs (Section 3.1); the compressed 4-byte layout used in the
  completed screen is graph-size-bounded per the same formula.
- Equal-silicon-area sensitivities (e.g. 14/15-way K2 versus a 16-way
  baseline) are evaluation sensitivities, not the primary implementation,
  and must not be described as the mechanism's normal operating point.
- No document may claim "zero K2 hardware overhead" -- the claim is zero
  reserved *data* ways with disclosed metadata overhead, which is a
  materially different statement.
- Physical/CACTI-style area-and-energy accounting, where used, must come from
  actual synthesis/CACTI inputs, never analytical stand-ins presented as
  measured values. This document intentionally avoids restating specific
  SHA/attestation or build-integrity ceremony text; that is an engineering
  concern of the artifact's build tooling (see `ARTIFACT.md`), not a
  scientific claim of this paper.

## 11. Claims and scope (allowed / prohibited)

**Allowed, with the stated evidence link:**

- "The static K2-RRIP+StreamShield complete design misses its aggregate time
  threshold versus GRASP and the optimistic charged P-OPT bound on the
  completed 12-cell PageRank screen; it beats LRU." This is a preregistered
  STOP statement, not a claim that realistic P-OPT is faster. (Section 8,
  `evidence/current/pr_screen_stop.md`)
- "K2-RRIP+StreamShield improves over K2-LRU+StreamShield under matched
  instruction count and delivery path." (Section 8.2)
- "The degree-1 common-prefetcher sensitivity confirms the same ranking."
  (Section 8.1, `evidence/current/prefetch_d1.md`)
- "cit-Patents is a first-class negative control that the screen does not
  pass." (Section 8.3)

**Prohibited until new evidence changes this file:**

- Any claim that K2 "generally beats P-OPT" or "generally beats GRASP" on
  full graphs, all kernels, or at scale -- the completed evidence is a
  12-cell sampled PageRank screen only.
- Any claim attributing the screen's outcome to "replacement policy alone"
  without disclosing the instruction-count ratio (Section 5.1).
- Any claim that Sniper timing supports an architectural K2-M speedup
  (Section 4).
- Any claim that P-OPT is author-evaluated on BFS or SSSP (Section 6).
- Any claim of a no-DBG-preprocessing advantage from the completed screen
  (Section 6, Section 9).
- Any claim of "zero K2 hardware overhead" (Section 10).
- Any claim that K2 has lower hardware overhead than P-OPT before the
  equal-area/physical-cost gate is complete.
- Restating the retired `epoch_first`-primary / per-kernel-adaptive mapping
  as this project's current static design (Section 3.2).
- Tuning thresholds or policy specifically to pass cit-Patents (Section 8.3).

## 12. Where the numbers live

- Current measured tables: [`RESULTS.md`](RESULTS.md) (non-normative; links
  back here for interpretation).
- Compact evidence notes: [`evidence/current/`](evidence/current/).
- Build/reproduction commands: [`ARTIFACT.md`](ARTIFACT.md).
- Canonical preregistration: `preregistration/pr_screen.json`. It carries a
  `superseded_screens` field recording that an earlier `epoch_first`-primary
  generation was stopped after its first cell; that generation is not an
  active configuration, and its evidence lives only in `evidence/current/`
  and git history, not in an active v1/v2-named preregistration file.
- Canonical manifest profile: `ecg_pr_screen` in
  `scripts/experiments/ecg/final_paper_manifest.json`. This is the single
  active proposal-screen profile; no `_v1`/`_v2`-suffixed screen profile is
  active.

## 13. Document status

This file supersedes and merges the substance of the former
`ARCHITECTURE.md`, `METHODOLOGY.md`, `CLAIMS.md`, `RUNBOOK.md` (build content
moved to `ARTIFACT.md` instead), and `CHAIR_QUERY.md`. Those files are removed
from the active tree; their history remains in git. Historical, superseded,
and withdrawn scientific narrative (metric-selection churn, retired ISA
exploration, and pre-screen sensitivity studies) is not reproduced here by
design -- it remains available in git history and in
[`evidence/`](evidence/) for audit, but it is not part of the current
normative claim set.
