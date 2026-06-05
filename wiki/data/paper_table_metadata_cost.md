# Paper Table 5 — Cache-substrate metadata cost: ECG vs DROPLET vs POPT

Runtime metadata storage required per graph for each cache-substrate
component. **ECG packs DBG eviction tier (GRASP-class) + POPT
re-reference quantization + prefetch target into ONE per-vertex mask.**
Baseline `combined` = GRASP + POPT + DROPLET state, separate per component.

## ECG mask bit decomposition

- DBG tier:          **2 bits**   (eviction tier — GRASP-class)
- POPT quantization: **7 bits**  (re-reference distance — POPT-class)
- Prefetch target:   **32 bits**   (direct vertex ID encoding)
- Reserved:          **23 bits**  (future per-vertex hints)
- **Total per vertex: 64 bits = 8 bytes**

## Per-graph storage (KB / MB)

| graph | vertices | edges | ECG (MB) | POPT (MB) | GRASP (KB) | DROPLET (KB) | baseline sum (MB) | ECG / POPT | ECG / baseline-sum |
|---|---|---|---|---|---|---|---|---|---|
| email-Eu-core | 1,005 | 16,064 | 0.008 | 0.015 | 20.00 | 16.00 | 0.051 | 0.499× | 0.152× |
| delaunay_n19 | 524,288 | 1,572,823 | 4.000 | 8.000 | 20.00 | 16.00 | 8.035 | 0.500× | 0.498× |
| roadNet-CA | 1,971,281 | 2,766,607 | 15.040 | 30.080 | 20.00 | 16.00 | 30.115 | 0.500× | 0.499× |
| web-Google | 875,713 | 4,322,051 | 6.681 | 13.363 | 20.00 | 16.00 | 13.398 | 0.500× | 0.499× |
| cit-Patents | 3,774,768 | 16,518,947 | 28.799 | 57.598 | 20.00 | 16.00 | 57.634 | 0.500× | 0.500× |
| soc-pokec | 1,632,803 | 22,301,964 | 12.457 | 24.915 | 20.00 | 16.00 | 24.950 | 0.500× | 0.499× |
| soc-LiveJournal1 | 4,847,571 | 42,851,237 | 36.984 | 73.968 | 20.00 | 16.00 | 74.003 | 0.500× | 0.500× |
| com-orkut | 3,072,626 | 117,185,083 | 23.442 | 46.885 | 20.00 | 16.00 | 46.920 | 0.500× | 0.500× |
| kron-s22 | 4,194,302 | 64,155,725 | 32.000 | 64.000 | 20.00 | 16.00 | 64.035 | 0.500× | 0.500× |
| kron-s24 | 16,777,212 | 260,376,710 | 128.000 | 256.000 | 20.00 | 16.00 | 256.035 | 0.500× | 0.500× |

## Aggregate across corpus

- Total vertices: **37,671,569**
- Total ECG mask storage: **287.4 MB**
- Total POPT matrix storage: **574.8 MB**
- Total GRASP per-line tags: **200.0 KB**
- Total DROPLET state: **160.0 KB**
- Total baseline-combined (GRASP+POPT+DROPLET): **575.2 MB**
- **ECG / POPT alone: 0.500× (50.0% smaller)**
- **ECG / baseline-combined: 0.500× (50.0% smaller)**

## Architectural simplicity (qualitative)

Beyond bytes, ECG requires only:
- 2 magic instructions (`SIM_CACHE_READ_MASKED`, `SIM_CACHE_PREFETCH_VERTEX`)
- A per-access mask decoder (few gates: bit shift + range compare)

DROPLET requires:
- 2 prefetch engines (stride detector + indirect engine)
- Stride table + indirect prediction table
- Edge-list and property-region monitors
- Per-engine state machines + coordination logic

POPT requires:
- Per-access rereference-matrix lookup unit
- MB-scale matrix storage (per-line × per-epoch)
- Offline preprocessing pass to build the matrix
