# Degree-1 Common-Prefetcher Sensitivity

**Status:** current sensitivity; not the primary no-prefetch result.

Authoritative complete run directories:

- `results/ecg_experiments/proposal_prefetch_d1_web_google_i1_1bc77078`
- `results/ecg_experiments/proposal_prefetch_d1_soc_pokec_i1_restart_1bc77078`
- `results/ecg_experiments/proposal_prefetch_d1_cit_patents_i1_1bc77078`

The incomplete pre-reboot soc-pokec directory and the two-policy resume directory
must not be combined with these cells. Their LRU target times differ by 4.8%.

Every policy received the same finite-resource gem5 L2 stride prefetcher with
degree 1. The optimistic charged P-OPT bound used the explicit
`analytic_prefetch_upper_bound` mode:

- reserved LLC capacity and all matrix bytes were charged;
- matrix latency was perfectly hidden;
- the analytic matrix stream incurred no prefetch over-fetch;
- P-OPT time and off-chip bytes are therefore favorable lower bounds;
- P-OPT's analytically added matrix misses are invalid for prefetch comparison.

Static RRIP K2+StreamShield ratios:

| Graph | vs LRU time/traffic | vs GRASP time/traffic | vs favored P-OPT time/traffic |
|---|---:|---:|---:|
| web-Google-n16 | 0.8142 / 0.6212 | 0.9274 / 0.7784 | 1.0085 / 0.9140 |
| soc-pokec-n16 | 0.8052 / 0.6431 | 0.9815 / 0.9091 | 1.0122 / 0.9587 |
| cit-Patents-n18-sym | 0.9824 / 0.9154 | 1.0337 / 1.0838 | 1.1983 / 1.0723 |
| **geomean** | **0.8636 / 0.7151** | **0.9799 / 0.9153** | **1.0694 / 0.9794** |

The sensitivity confirms rather than rescues the primary STOP: K2 strongly
beats LRU, approximately ties GRASP in aggregate, and remains slower than the
deliberately favored P-OPT bound. `cit-Patents-n18-sym` remains the negative
case, so iteration expansion was stopped.
