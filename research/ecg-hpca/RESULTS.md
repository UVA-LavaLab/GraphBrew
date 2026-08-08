# Results (current)

**Non-normative.** This file reports current measured tables only. Claim
scope, interpretation, and prohibited-claim boundaries are defined in
[`PAPER.md`](PAPER.md); read that file first. Superseded/withdrawn
chronological lab-notebook history has been removed from this file by design
-- it remains in git history and in [`evidence/`](evidence/) for audit, but it
is not restated here. Every table below links to its compact evidence note
under [`evidence/current/`](evidence/current/), which in turn names the raw
`results/ecg_experiments/` run directory.

## 0. Screen configuration

| Parameter | Value |
|---|---|
| Simulator | gem5, CPU type O3 |
| ISA variant | `mask` (K2-M computed-address property load) |
| K2 timing mode | `compact_trace_free` |
| Record width | 4 bytes (destination + 2 tier bits + two epoch stamps) |
| Epochs | 32 for all three samples (5 compact epoch bits) |
| Graphs | `web-Google-n16`, `soc-pokec-n16` (65,536 vertices each), `cit-Patents-n18-sym` (262,144 vertices) |
| PageRank iterations | 1, 2, 4, 8 |
| Order | DBG (`-o 5`) |
| Prefetcher | none (Section 1), gem5 L2 stride degree 1 (Section 3) |
| Config | `preregistration/pr_screen.json` |
| Manifest profile | `ecg_pr_screen` |

## 1. Completed gem5 O3 no-prefetch PageRank screen (STOP)

Run directory:
`results/ecg_experiments/final_paper_runs/proposal_sota_v2_2db0a04b_20260802`
Evidence note: [`evidence/current/pr_screen_stop.md`](evidence/current/pr_screen_stop.md)

12 cells, 84 `ok` rows: graphs `web-Google-n16`, `soc-pokec-n16`,
`cit-Patents-n18-sym`; PageRank iterations 1, 2, 4, 8; policies LRU, GRASP,
optimistic charged P-OPT bound, uncharged P-OPT, K2-LRU+StreamShield, static
K2-RRIP+StreamShield (preregistered primary), online K2+StreamShield
(characterization only).

| Baseline/control | Time ratio | Off-chip traffic ratio |
|---|---:|---:|
| LRU | 0.9061 | 0.7227 |
| GRASP | 0.9835 | 0.9455 |
| optimistic charged P-OPT bound | 1.0235 | 0.9351 |
| uncharged P-OPT | 1.0503 | 1.1776 |
| K2-LRU+StreamShield | 0.9330 | 0.7855 |

The P-OPT row charges reserved capacity and cumulative matrix bytes, but
`popt_target_time_charged=0` omits matrix-stream latency. Its time is an
optimistic bound; realistic target-time P-OPT performance remains unresolved.

**Result: STOP.** The screen is valid; static K2-RRIP+StreamShield misses the
frozen aggregate time threshold versus GRASP and the optimistic charged P-OPT
bound. This is a valid preregistered STOP, not evidence that a realistic
target-time P-OPT engine is faster. Decisive negative graph:
`cit-Patents-n18-sym`, iteration 8:

| Baseline | Time ratio | Off-chip traffic ratio |
|---|---:|---:|
| GRASP | 1.0613 | 1.1141 |
| optimistic charged P-OPT bound | 1.1373 | 1.0322 |

The screen used DBG order (`-o 5`) in every cell because P-OPT's own direct
GRASP comparison is PageRank on DBG-ordered graphs. It therefore does not
establish that K2 avoids DBG preprocessing (see `PAPER.md` Section 6/9).

An earlier generation of this screen used `epoch_first` as the static
primary and was stopped after its first cell. It is not an active
configuration; see `preregistration/pr_screen.json`'s `superseded_screens`
field and git history for that lineage.

## 2. Instruction-count disclosure

Evidence note:
[`evidence/current/instruction_decomposition.md`](evidence/current/instruction_decomposition.md)

The screen used equal semantic work and one build, but K2 retired fewer
instructions than conventional policies:

| Graph | K2/LRU retired-instruction ratio |
|---|---:|
| web-Google-n16 | 0.8546 |
| soc-pokec-n16 | 0.8144 |
| cit-Patents-n18-sym | 0.9120 |
| **geomean** | **0.8594** |

Per `PAPER.md` Section 5.1, this is a disclosure obligation, not a strict
parity requirement: instruction inequality is allowed for the complete-design
STOP above, but it prohibits attributing that result to replacement policy
alone. The parity-controlled attribution is internal to the K2 family:

| Comparison | Time ratio | Off-chip traffic ratio |
|---|---:|---:|
| K2-RRIP+StreamShield vs K2-LRU+StreamShield | 0.9330 | 0.7855 |

Same record stream, delivery path, ISA, and instruction count on both sides.
This is the only comparison in the current evidence set that isolates
replacement-policy effect from record/ISA/transport effect.

## 3. Degree-1 common-prefetcher sensitivity

Authoritative run directories:

- `results/ecg_experiments/proposal_prefetch_d1_web_google_i1_1bc77078`
- `results/ecg_experiments/proposal_prefetch_d1_soc_pokec_i1_restart_1bc77078`
- `results/ecg_experiments/proposal_prefetch_d1_cit_patents_i1_1bc77078`

Evidence note: [`evidence/current/prefetch_d1.md`](evidence/current/prefetch_d1.md)

The incomplete pre-reboot soc-pokec directory and the two-policy resume
directory (LRU targets differ by 4.8%) must not be substituted for the
`_restart_` directory above. Every policy used the same finite-resource gem5
L2 stride prefetcher, degree 1. The optimistic charged P-OPT bound used
`analytic_prefetch_upper_bound`: reserved capacity and matrix bytes are
charged, but the analytic matrix stream gets perfect latency-hiding and no
over-fetch -- a deliberately favorable bound for P-OPT.

Static K2-RRIP+StreamShield:

| Graph | vs LRU (time/traffic) | vs GRASP (time/traffic) | vs favored P-OPT bound (time/traffic) |
|---|---:|---:|---:|
| web-Google-n16 | 0.8142 / 0.6212 | 0.9274 / 0.7784 | 1.0085 / 0.9140 |
| soc-pokec-n16 | 0.8052 / 0.6431 | 0.9815 / 0.9091 | 1.0122 / 0.9587 |
| cit-Patents-n18-sym | 0.9824 / 0.9154 | 1.0337 / 1.0838 | 1.1983 / 1.0723 |
| **geomean** | **0.8636 / 0.7151** | **0.9799 / 0.9153** | **1.0694 / 0.9794** |

This sensitivity confirms rather than rescues the primary STOP: K2 strongly
beats LRU, approximately ties GRASP in aggregate, and remains slower than the
deliberately favored P-OPT bound. `cit-Patents-n18-sym` remains the negative
case, so iteration expansion was not pursued past this sensitivity.

## 4. Sniper parity status

Sniper shares the victim-selector plumbing, independently compiled P-OPT
lookup parity, realized-cache checks, and concurrency-safe online-dueling
evidence. Per `PAPER.md` Section 4, this establishes comparable cache/traffic
*direction* evidence, not an architectural timing claim: gem5 O3 remains the
sole simulator whose execution time may support a speedup number. No
full-graph Sniper matrix at equal semantic-edge limits is complete yet; the
plan for it is in `PAPER.md` Section 7. No Sniper row in current evidence
should be cited as a K2-M architectural speedup.

## 5. Current negative conclusions

- **cit-Patents-n18-sym is a first-class negative/control graph.** It drives
  the STOP result in both the no-prefetch screen (Section 1) and the
  degree-1 sensitivity (Section 3). It is retained deliberately, not
  excluded or reweighted, and no threshold or policy behavior has been (or
  may be) tuned post hoc to pass it. Any design change motivated by
  diagnosing its failure requires a new preregistration generation evaluated
  across the full frozen graph roster (`PAPER.md` Section 8.3).
- **Static K2-RRIP+StreamShield does not clear the frozen aggregate-time bar
  versus GRASP or the optimistic charged P-OPT bound** on the completed
  12-cell PageRank screen, under either no-prefetch or degree-1-prefetch
  configurations.
- **No full-graph, all-kernel, or Sniper-timing claim is currently
  supported.** The evidence above is a bounded, sampled PageRank screen; see
  `PAPER.md` Section 7 for the plan to extend it with currently-available
  local datasets, and Section 6/9 for literature-scope and reordering-benefit
  caveats that remain untested.
