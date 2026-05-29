"""No-hub-graph invariant for GRASP.

GRASP's central design assumption is that the graph has a small,
identifiable set of *hot* vertices ("hubs") whose adjacency lists
benefit from being pinned in the L3 — i.e., GRASP is "hot-vertex
pinning". On graphs *without* hubs (e.g., road networks, where every
vertex has roughly the same degree ~3-4 and there is no power-law
tail), the design has nothing to pin and degrades into noise.

This file pins that *as a theory-driven invariant*: any graph in our
corpus with ``hub_concentration < HUB_LOW_THRESHOLD`` must NOT show a
meaningful GRASP-over-LRU win at the headline L3 size. If a future
GRASP variant claims to win on a no-hub graph, that's a sign of
either (a) overfitting to the GRASP-friendly graphs in our corpus,
(b) the no-hub graph actually having a long tail we mis-measured,
or (c) a real algorithmic improvement that should be documented and
celebrated — in any case, the test failure forces the reviewer to
look.

We pin both directions:

* ``test_no_hub_graph_present`` — make sure we actually have at least
  one no-hub graph in the corpus, so the invariant is *load-bearing*
  rather than vacuously satisfied.
* ``test_grasp_does_not_beat_lru_on_no_hub_graphs`` — the no-hub
  graphs must have ``GRASP miss >= LRU miss - TOLERANCE_PP`` for the
  headline kernel (PR) at the L3 size we have data for.

The list of no-hub graphs is discovered from ``corpus_diversity.json``
rather than hardcoded so that adding a new road-style graph
automatically extends the invariant.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_JSON = REPO_ROOT / "wiki" / "data" / "corpus_diversity.json"
LIT_FAITH_CSV = REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.csv"

# Theory threshold: a graph with hub_concentration < this value has no
# meaningful power-law tail of degree (every vertex has roughly the
# same degree, so there is no hot set to pin in L3). The 0.20 cut is
# well below the lowest hub_concentration of our social/web/citation
# graphs (the lowest non-road graph in the corpus has ~0.34) and
# comfortably above the roadNet-CA value (~0.14).
HUB_LOW_THRESHOLD = 0.20

# A GRASP win below this magnitude is noise (signal is comparable to
# the lit-faith within_tolerance band on this regime). A GRASP-over-LRU
# improvement exceeding this magnitude on a no-hub graph would be the
# surprising result that violates the design's stated assumption.
GRASP_WIN_NOISE_FLOOR_PP = 0.5


def _load_corpus() -> list[dict]:
    if not CORPUS_JSON.exists():
        pytest.skip(f"{CORPUS_JSON.relative_to(REPO_ROOT)} not on disk")
    return json.loads(CORPUS_JSON.read_text())


def _hub(card: dict) -> float | None:
    feats = card.get("features", {}) if isinstance(card, dict) else {}
    v = feats.get("hub_concentration") if isinstance(feats, dict) else None
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _no_hub_graphs() -> list[str]:
    return [
        c.get("graph", "")
        for c in _load_corpus()
        if (_hub(c) or 1.0) < HUB_LOW_THRESHOLD
    ]


def _miss_rate(graph: str, policy: str, benchmark: str = "pr") -> float | None:
    """Look up the cache_sim L3 miss rate for (graph, policy, benchmark)
    at any L3 size; returns the value at the first matching row. We
    don't pin a specific L3 here because the invariant must hold at
    *every* L3 size (a no-hub graph has no hubs at any L3).
    """
    if not LIT_FAITH_CSV.exists():
        return None
    with LIT_FAITH_CSV.open() as f:
        for row in csv.DictReader(f):
            if (
                row.get("graph") == graph
                and row.get("policy") == policy
                and (row.get("benchmark") or "pr") == benchmark
            ):
                try:
                    return float(row.get("miss_rate") or row.get("l3_miss_rate") or "nan")
                except (TypeError, ValueError):
                    continue
    return None


def test_no_hub_graph_present() -> None:
    """The corpus must contain at least one no-hub graph so the
    GRASP-needs-hubs invariant is load-bearing.
    """
    graphs = _no_hub_graphs()
    assert graphs, (
        f"corpus has zero graphs with hub_concentration < {HUB_LOW_THRESHOLD}; "
        f"add a road-style graph (e.g., roadNet-CA) so the "
        f"GRASP-needs-hubs invariant is load-bearing rather than vacuous"
    )


@pytest.mark.parametrize("graph", _no_hub_graphs() or ["__skip__"])
def test_grasp_does_not_beat_lru_on_no_hub_graphs(graph: str) -> None:
    """GRASP must not show a meaningful win over LRU on no-hub graphs."""
    if graph == "__skip__":
        pytest.skip("no no-hub graphs in corpus")
    lru = _miss_rate(graph, "LRU")
    grasp = _miss_rate(graph, "GRASP")
    if lru is None or grasp is None:
        pytest.skip(
            f"no LRU/GRASP cache_sim data for {graph} in lit-faith CSV "
            f"(have lru={lru!r}, grasp={grasp!r})"
        )
    grasp_minus_lru_pp = (grasp - lru) * 100.0
    assert grasp_minus_lru_pp > -GRASP_WIN_NOISE_FLOOR_PP, (
        f"GRASP beats LRU by {-grasp_minus_lru_pp:.3f}pp on no-hub graph "
        f"{graph} (grasp={grasp:.4f} lru={lru:.4f}); this contradicts the "
        f"GRASP-paper design assumption that GRASP wins by pinning a hot "
        f"set in L3. Either the corpus_diversity hub_concentration is "
        f"mis-measured, or GRASP has gained a no-hub-friendly improvement "
        f"that needs to be documented. Loosen "
        f"GRASP_WIN_NOISE_FLOOR_PP only after writing up the cause."
    )
