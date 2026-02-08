# Tuning Playbook — Leiden Variant Optimization

Practical guide for tuning parameters and ordering strategies to beat RabbitOrder.

---

## Strategy 1: Resolution Tuning

**What it controls:** Community granularity. Lower resolution → larger communities
(fewer, coarser). Higher resolution → smaller communities (more, finer).

**Why it matters for cache:** If communities are too large, they don't fit in L2
cache. If too small, inter-community edges dominate and locality drops.

**Sweet spots by graph type:**
| Graph Type | Suggested Resolution | Reasoning |
|------------|:---:|-----------|
| Social (power-law) | 0.5–1.0 | Natural communities are medium-sized |
| Web (link structure) | 0.75–1.5 | Denser link clusters need finer splitting |
| Road (grid-like) | 0.25–0.5 | Communities should be large (spatial locality) |
| Mesh (uniform) | 0.25–0.5 | Little community structure; large blocks best |
| Citation (DAG-ish) | 0.75–1.0 | Moderate community structure |

**How to test:**
```bash
# Quick sweep on 4 representative graphs
for R in 0.25 0.5 0.75 1.0 1.5 2.0; do
  python3 scripts/graphbrew_experiment.py --full --auto --skip-cache \
    --graph-list web-Google soc-Slashdot0902 roadNet-CA cit-Patents \
    --csr-variants "leiden:gveopt2:${R}" rabbit:csr \
    --benchmarks pr bfs --trials 5
done
```

**What to look for:**
- Community count at each resolution (logged in verbose output)
- If kernel time improves as resolution increases → communities were too coarse
- If kernel time degrades as resolution increases → over-splitting

---

## Strategy 2: Ordering Strategy Selection

**The landscape:**
```
Community Detection (Leiden) → Ordering (how vertices within/between communities are laid out)
```

The community detection is shared. The ordering strategy is where you win or lose.

**Strategy comparison:**

| Strategy | Best For | Why |
|----------|----------|-----|
| BFS connectivity | General (default) | Preserves graph-walk locality within communities |
| DFS dendrogram | Deep hierarchies | Keeps sub-communities contiguous in memory |
| HRAB | Social/Web | RabbitOrder's super-graph ordering is excellent for inter-community layout |
| HRAB+Gordi | Social/Web (large) | Adds Gorder-greedy intra-community for fine-grained locality |
| DBG | Power-law graphs | Groups by degree → better for algorithms that access hubs frequently |
| Hub extraction | Extreme power-law | Pulls top hubs out of communities → separate hot region |

**Recommendation:** Start with `vibe:hrab` as baseline. If it beats `rabbit:csr`,
try `vibe:hrab:gordi` to see if the extra intra-community ordering is worth
the cost. If `vibe:hrab` loses, diagnose whether the problem is inter-community
or intra-community ordering.

---

## Strategy 3: Refinement Tuning

**Leiden refinement** is the key difference from Louvain (which RabbitOrder uses).
It guarantees well-connected communities by refining the partition after
local-moving.

**The trade-off:** Refinement adds ~20-40% to reorder time but may produce
better communities (fewer disconnected sub-communities).

**When refinement helps:**
- Graphs with weak community structure (road networks, meshes)
- Large graphs where a few vertices are badly assigned

**When refinement hurts:**
- Graphs with strong natural communities (social networks)
- Small graphs where refinement cost > benefit

**Test:**
```cpp
// In reorder_vibe.h, VibeConfig:
bool useRefinement = false;  // disable to test
```
Then benchmark both versions on MEDIUM tier.

---

## Strategy 4: Dynamic Resolution

Instead of fixed resolution, adjust per aggregation pass:

- Early passes: high resolution (find fine communities)
- Later passes: lower resolution (merge into cache-sized blocks)

**Enable:** Set resolution to `dynamic` or `dynamic_1.5` (start at 1.5, then auto-adjust).

**Code location:** `VibeConfig::useDynamicResolution` (~L299 of reorder_vibe.h)

**When it helps:** Graphs where the natural community hierarchy doesn't align
with cache hierarchy. Dynamic resolution can adapt.

---

## Strategy 5: Gorder Window Tuning (gordi)

`vibe:hrab:gordi` uses a Gorder-greedy intra-community ordering with a
sliding window. The window size controls the locality radius.

**Default window:** 5 (small, fast)

**Larger window:** More expensive but considers more vertices for placement.
Diminishing returns past ~10 for most graphs.

**Test:**
```cpp
// In VibeConfig:
int gorderWindow = 10;  // try 5, 7, 10, 15
```

---

## Strategy 6: Hub Extraction

For extreme power-law graphs (social networks), extracting the top 0.1% of
hubs from communities and placing them contiguously can reduce cache thrashing
from hub accesses.

**Enable:** `vibe:hubx` or set `useHubExtraction = true` in VibeConfig.

**Why it could beat RabbitOrder:** RabbitOrder doesn't do hub extraction.
If hubs are scattered across communities, every community access touches
the same hot cache lines → cache contention. Extracting them creates a
dedicated hot region.

**Risk:** May fragment community structure. Only beneficial if hub access
pattern dominates (PR, BFS on social networks).

---

## Strategy 7: Tile Size Tuning

Cache blocking tile size affects how the Leiden algorithm processes vertices
during local-moving.

**Default:** 4096

**Trade-off:**
- Smaller tiles (1024–2048): Better L1 cache utilisation during reordering
- Larger tiles (8192–16384): Less overhead, better for large graphs

This affects reorder time, not kernel time. Only tune if reorder time is
the bottleneck.

---

## Debugging Poor Performance

### Symptom: RabbitOrder is faster to reorder AND better kernel time
**Diagnosis:** Leiden is over-computing. Try:
1. `leiden:fast` (fewer iterations)
2. Disable refinement
3. Lower `maxPasses` to 3

### Symptom: Good kernel time but reorder cost kills end-to-end
**Diagnosis:** Locality is good but Leiden is too expensive. Try:
1. `vibe:rabbit` (use Leiden detection with RabbitOrder-style aggregation)
2. Reduce `maxIterations` to 5
3. Use `FAST_LP` community mode (label propagation instead of full Leiden)

### Symptom: Good on social, bad on road networks
**Diagnosis:** Resolution is wrong for grid-like graphs. Try:
1. Lower resolution (0.25–0.5) for road networks
2. Consider DFS ordering (preserves spatial locality)
3. Road networks may not benefit from community-based reordering at all

### Symptom: Good L3 miss rate but no kernel speedup
**Diagnosis:** Bottleneck is elsewhere (L1/L2 or computation). Try:
1. Cache simulation to check L1/L2 specifically
2. Intra-community ordering change (DBG, Gorder)
3. The algorithm may be compute-bound, not memory-bound
