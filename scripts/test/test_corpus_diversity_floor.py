"""Diversity-floor gate for the literature sweep corpus.

The Faldu / Balaji baselines depend on the corpus actually spanning the
topological variety the papers report on (citation, web, social, etc.).
This file enforces a minimum-diversity floor over the
``wiki/data/corpus_diversity.json`` snapshot so a future contributor
can't quietly delete a graph and have the dashboard remain GREEN.

What we require:

  * **Graph-count floor**: ≥4 distinct graphs profiled, ≥1M nodes on
    average — the email-Eu-core smoke graph alone does not satisfy
    "diverse corpus".
  * **Family coverage**: at least one social, one web, and one
    citation graph in the corpus. These are the three families the
    papers actually evaluate on.
  * **Clustering spread**: range(clustering_coeff) ≥ 0.05 — i.e., the
    corpus is not all-clustered or all-sparse, so a hub-based
    replacement policy's win is *load-bearing*, not free.
  * **Hub-concentration spread**: range(hub_concentration) ≥ 0.10 — the
    corpus must contain both high-hub and low-hub graphs so GRASP's
    hot-vertex pinning is actually stressed.
  * **Working-set-ratio span**: ≥3 orders of magnitude (max / min ≥
    1000) — the corpus must span "fits in cache" to "vastly exceeds
    cache" so cache-policy results are not a single regime.

If a future curated corpus violates one of these floors the test
fails with a self-explanatory message naming the missing dimension.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_JSON = REPO_ROOT / "wiki" / "data" / "corpus_diversity.json"


def _load() -> list[dict]:
    if not CORPUS_JSON.exists():
        pytest.skip(f"{CORPUS_JSON.relative_to(REPO_ROOT)} not on disk")
    payload = json.loads(CORPUS_JSON.read_text())
    if isinstance(payload, dict):
        cards = payload.get("graphs") or payload.get("rows") or []
    else:
        cards = payload or []
    if not cards:
        pytest.fail("corpus_diversity.json has zero graphs")
    return cards


# Hard-coded mapping of corpus graph -> family, so the test is robust
# against future graphs being added. If a new graph is added without
# being tagged here, the family-coverage test will treat it as
# "unknown" (still counted in the total).
GRAPH_FAMILY = {
    "email-Eu-core": "social",        # email graph, treated as social
    "web-Google": "web",
    "cit-Patents": "citation",
    "soc-pokec": "social",
    "soc-LiveJournal1": "social",
    "com-orkut": "social",
    "roadNet-CA": "road",
    # Reserved tags for graphs we may add later:
    "road-CA": "road",                # historical alias for roadNet-CA
    "twitter-2010": "social",
    "uk-2005": "web",
}


def _feature(card: dict, key: str) -> float | None:
    feats = card.get("features", {})
    v = feats.get(key) if isinstance(feats, dict) else None
    if v is None and isinstance(card, dict):
        v = card.get(key)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def test_corpus_has_minimum_graph_count() -> None:
    cards = _load()
    assert len(cards) >= 4, (
        f"corpus has only {len(cards)} graph(s); need >=4 to claim "
        f"literature-grade diversity (one smoke graph + three "
        f"distinct topology families)"
    )


def test_corpus_has_minimum_average_size() -> None:
    cards = _load()
    nodes = [int(c.get("nodes", 0) or 0) for c in cards if (c.get("nodes") or 0) > 0]
    avg = sum(nodes) / len(nodes) if nodes else 0.0
    assert avg >= 1_000_000, (
        f"average node count is {avg:,.0f}; need >=1M nodes on average "
        f"so the corpus stresses cache hierarchies and is not "
        f"dominated by toy graphs"
    )


def test_corpus_covers_social_web_and_citation_families() -> None:
    cards = _load()
    families: set[str] = set()
    untagged: list[str] = []
    for c in cards:
        g = c.get("graph", "")
        fam = GRAPH_FAMILY.get(g)
        if fam:
            families.add(fam)
        else:
            untagged.append(g)
    required = {"social", "web", "citation"}
    missing = required - families
    assert not missing, (
        f"corpus is missing required topology families: {sorted(missing)}; "
        f"present families={sorted(families)}; untagged graphs={untagged} "
        f"(add a GRAPH_FAMILY entry in this test if you added a new graph)"
    )


def test_corpus_clustering_coefficient_spans_range() -> None:
    cards = _load()
    values = [_feature(c, "clustering_coeff") for c in cards]
    values = [v for v in values if v is not None]
    assert values, "no graph reported clustering_coeff"
    spread = max(values) - min(values)
    assert spread >= 0.05, (
        f"clustering_coeff range across corpus is only {spread:.3f} "
        f"(min={min(values):.4f}, max={max(values):.4f}); need >=0.05 "
        f"so hub-aware policies are tested across both clustered and "
        f"sparse regimes"
    )


def test_corpus_hub_concentration_spans_range() -> None:
    cards = _load()
    values = [_feature(c, "hub_concentration") for c in cards]
    values = [v for v in values if v is not None]
    assert values, "no graph reported hub_concentration"
    spread = max(values) - min(values)
    assert spread >= 0.10, (
        f"hub_concentration range across corpus is only {spread:.3f} "
        f"(min={min(values):.4f}, max={max(values):.4f}); need >=0.10 "
        f"so GRASP's hot-vertex pinning is stressed across both "
        f"hub-heavy and hub-light graphs"
    )


def test_corpus_working_set_ratio_spans_three_orders() -> None:
    cards = _load()
    values = [_feature(c, "working_set_ratio") for c in cards]
    # working_set_ratio = (edges / cache_capacity) in some normalized
    # unit; on the email-Eu-core smoke graph it's ~0.004, on com-orkut
    # it's ~29 — we want this 3+ orders of magnitude span so cache-size
    # sweeps actually traverse the "fits / spills" boundary.
    values = [v for v in values if v is not None and v > 0]
    assert values, "no graph reported working_set_ratio > 0"
    ratio = max(values) / min(values)
    assert ratio >= 1000.0, (
        f"working_set_ratio span is only {ratio:.1f}x (min={min(values):.4f}, "
        f"max={max(values):.4f}); need >=1000x so the corpus crosses "
        f"the 'fits in cache' -> 'spills L3' boundary"
    )
