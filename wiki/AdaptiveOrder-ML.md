# AdaptiveOrder (research-only)

> **Not part of the VLDB 2026 submission.** This page describes the
> ML-based runtime algorithm selector kept in-tree for future work.
> Skip if you only care about reproducing the paper.

AdaptiveOrder is reordering algorithm ID **14**. Instead of picking a
fixed reordering, it extracts structural features from the input graph
(degree skew, modularity, clustering coefficient, etc.) and uses a
trained model to predict which fixed algorithm will perform best on
that graph + benchmark combination.

## Quick use

```bash
# Let AdaptiveOrder pick a reordering for PR
./bench/bin/pr -f your_graph.el -s -o 14 -n 3
```

The chosen algorithm is printed at the start of the run. If no trained
model is available the selector falls back to a heuristic default
(currently HUBCLUSTERDBG).

## Where the models live

| File | Contents |
|---|---|
| `results/data/benchmark.json` | feature/performance training data, populated by the standard benchmark pipeline |
| `results/data/adaptive_models.json` | trained perceptron / decision-tree / hybrid / kNN weights |

Both are auto-created by `ensure_prerequisites()` in `scripts/lib/`
on the first run that needs them.

## Training

Models are trained at runtime inside the C++ binary the first time
algorithm 14 is invoked with sufficient training data in
`benchmark.json`. Manual retraining is unnecessary for normal use.

To force a retrain from scratch:

```bash
python3 scripts/graphbrew_experiment.py --train --size small
```

## Selection modes

`-o 14:<mode>` chooses the selector. Default is `0` (perceptron).

| Mode | Selector |
|---|---|
| `0` | linear perceptron |
| `1` | decision tree |
| `2` | hybrid (perceptron + DT vote) |
| `3` | weighted kNN |
| `4` | database lookup (exact-match by graph name) |
| `5` | round-robin (debug only) |
| `6` | oracle (cheats — picks the actual best from benchmark.json) |

Mode 6 is useful as an upper bound when measuring how close the
trained selectors come to perfect prediction.

## Feature set

16 linear features + 5 quadratic cross-terms are extracted from each
graph: vertex count, edge count, average degree, max degree, degree
variance, Gini coefficient, clustering coefficient, modularity (from
a single Leiden pass), reciprocity, density, and a few others.
Definitions live in `bench/include/adaptive/features.h`.

## Why this is research-only

The VLDB submission's contribution is the **composable multilayered
reordering pipeline** (variants 12:*), not learned selection.
AdaptiveOrder is a natural follow-up but adds training overhead and
generalisation concerns that warrant a separate publication.

For details of the historical training pipeline, perceptron weights,
and cross-validation scaffolding, see git history before May 2026 or
the source files under `bench/include/adaptive/`.
