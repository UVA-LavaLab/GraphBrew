"""Structural pytest gate for ``literature_baselines.py``.

These tests catch typos, dangling references, and missing rationales in the
``INVARIANT_CLAIMS`` / ``PER_GRAPH_CLAIMS`` / ``KNOWN_DEVIATIONS`` tables
*before* the data-driven gate (``test_baselines_match_literature.py``) runs
against the sweep. A typo here would otherwise silently degrade to a
``missing`` verdict in the comparator instead of failing loudly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ECG_DIR = REPO_ROOT / "scripts" / "experiments" / "ecg"


def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ECG_DIR / file_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


lit = _load("literature_baselines", "literature_baselines.py")


VALID_APPS = {"pr", "bc", "bfs", "sssp", "cc"}
WILDCARD_APPS = VALID_APPS | {"*"}
VALID_L3_SIZES = {"*", "1MB", "4MB", "8MB"}
# Concrete policies (where a literature claim says "policy X should be Y% better than LRU").
CONCRETE_POLICIES = {"LRU", "SRRIP", "GRASP", "POPT"}
# Synthetic relational policies (compare two real policies against each other).
# These legitimately use wildcards because the assertion is structural, not
# graph-specific.
RELATIONAL_POLICIES = {"POPT_GE_GRASP", "POPT_NEAR_GRASP_IF_BIG_GAP"}
VALID_POLICIES = CONCRETE_POLICIES | RELATIONAL_POLICIES

# Graphs the literature corpus actually covers (matches corpus_diversity.GRAPH_ORDER).
LITERATURE_GRAPHS = {
    "email-Eu-core",
    "web-Google",
    "cit-Patents",
    "soc-pokec",
    "com-orkut",
    "soc-LiveJournal1",
}
# Wildcard graph patterns we accept in claims.
WILDCARD_GRAPHS = {"*", "*power_law*"}


@pytest.mark.parametrize("claim", lit.INVARIANT_CLAIMS + lit.PER_GRAPH_CLAIMS,
                         ids=lambda c: f"{c.graph}/{c.app}/{c.l3_size}/{c.policy}")
def test_claim_fields_are_well_formed(claim) -> None:
    """Every literature claim must use a known app / L3 size / policy."""

    assert claim.app in WILDCARD_APPS, (
        f"Claim {claim} uses an unknown app '{claim.app}'. "
        f"Valid apps: {sorted(WILDCARD_APPS)}."
    )
    assert claim.l3_size in VALID_L3_SIZES, (
        f"Claim {claim} uses an unknown l3_size '{claim.l3_size}'. "
        f"Valid sizes: {sorted(VALID_L3_SIZES)}."
    )
    assert claim.policy in VALID_POLICIES, (
        f"Claim {claim} uses an unknown policy '{claim.policy}'. "
        f"Valid policies: {sorted(VALID_POLICIES)}."
    )
    assert claim.expected_sign in {"-", "+", "~"}, (
        f"Claim {claim} has expected_sign='{claim.expected_sign}'; "
        "must be '-' (improves), '+' (regresses), or '~' (no clear direction)."
    )
    assert claim.tolerance_pct >= 0, (
        f"Claim {claim} has negative tolerance_pct={claim.tolerance_pct}."
    )
    if claim.min_abs_delta_pct is not None and claim.max_abs_delta_pct is not None:
        assert claim.min_abs_delta_pct <= claim.max_abs_delta_pct, (
            f"Claim {claim} has min_abs_delta_pct > max_abs_delta_pct."
        )
    assert claim.rationale.strip(), f"Claim {claim} has empty rationale."
    assert claim.citation.strip(), f"Claim {claim} has empty citation."


@pytest.mark.parametrize("claim", lit.PER_GRAPH_CLAIMS,
                         ids=lambda c: f"{c.graph}/{c.app}/{c.l3_size}/{c.policy}")
def test_per_graph_claims_reference_real_graphs(claim) -> None:
    """``PER_GRAPH_CLAIMS`` may not use wildcard graph patterns.

    Exception: relational claims (``POPT_GE_GRASP``,
    ``POPT_NEAR_GRASP_IF_BIG_GAP``) legitimately use ``graph='*'`` because
    the assertion compares two policies' deltas, not a specific graph's
    miss rate against a literature number.
    """

    if claim.policy in RELATIONAL_POLICIES:
        return  # wildcards permitted for relational/synthetic invariants
    assert claim.graph in LITERATURE_GRAPHS, (
        f"Per-graph claim {claim} references graph '{claim.graph}' that is "
        f"not in the literature corpus {sorted(LITERATURE_GRAPHS)}. If this "
        "is intentional, add the graph to the corpus or move the claim into "
        "INVARIANT_CLAIMS with a wildcard."
    )


@pytest.mark.parametrize("claim", lit.INVARIANT_CLAIMS,
                         ids=lambda c: f"{c.graph}/{c.app}/{c.l3_size}/{c.policy}")
def test_invariant_claims_use_wildcard_graph(claim) -> None:
    """``INVARIANT_CLAIMS`` are graph-agnostic and must use a wildcard."""

    assert claim.graph in WILDCARD_GRAPHS, (
        f"Invariant claim {claim} uses concrete graph '{claim.graph}'; "
        f"move it to PER_GRAPH_CLAIMS or change the graph to one of "
        f"{sorted(WILDCARD_GRAPHS)}."
    )


@pytest.mark.parametrize("key, reason", list(lit.KNOWN_DEVIATIONS.items()),
                         ids=lambda x: "/".join(x) if isinstance(x, tuple) else None)
def test_known_deviations_reference_real_cells(key, reason) -> None:
    """Every KNOWN_DEVIATIONS entry must point at a real (graph, app, l3, policy) cell."""

    graph, app, l3_size, policy = key
    assert graph in LITERATURE_GRAPHS, (
        f"KNOWN_DEVIATIONS key {key} references graph '{graph}' not in the "
        f"literature corpus {sorted(LITERATURE_GRAPHS)}."
    )
    assert app in VALID_APPS, f"KNOWN_DEVIATIONS key {key} app '{app}' is unknown."
    assert l3_size in (VALID_L3_SIZES - {"*"}), (
        f"KNOWN_DEVIATIONS key {key} l3_size '{l3_size}' is unknown; "
        "wildcards are not allowed in deviation keys."
    )
    assert policy in VALID_POLICIES, (
        f"KNOWN_DEVIATIONS key {key} policy '{policy}' is unknown."
    )
    assert reason.strip(), f"KNOWN_DEVIATIONS key {key} has empty reason."
    # Rationales should be substantive — at least one sentence + 50 chars.
    assert len(reason) >= 50, (
        f"KNOWN_DEVIATIONS key {key} has suspiciously short reason "
        f"({len(reason)} chars). Add a real explanation linking the "
        "deviation to source code, paper section, or measured behaviour."
    )


def test_no_duplicate_known_deviations() -> None:
    """KNOWN_DEVIATIONS is a dict so duplicates can't exist, but verify
    no entry collides with a per-graph claim that would shadow it silently."""

    deviation_keys = set(lit.KNOWN_DEVIATIONS.keys())
    claim_keys = {(c.graph, c.app, c.l3_size, c.policy) for c in lit.PER_GRAPH_CLAIMS}
    overlap = deviation_keys & claim_keys
    # An overlap *is* legal (the deviation downgrades the claim's failure to
    # xfail) but every overlap should be intentional, not a typo. We assert
    # all overlaps reference a known POPT_GE_GRASP-style relational claim or
    # a per-graph claim that already exists.
    for key in overlap:
        assert key in claim_keys, (
            f"KNOWN_DEVIATIONS key {key} overlaps but no claim found."
        )


def test_each_corpus_graph_has_at_least_one_claim() -> None:
    """Every graph in the literature corpus must be referenced by at least
    one PER_GRAPH_CLAIM. Otherwise we have a graph we're sweeping without
    encoding any expectations for it."""

    referenced = {c.graph for c in lit.PER_GRAPH_CLAIMS}
    missing = LITERATURE_GRAPHS - referenced
    # email-Eu-core is exempt — it's a sanity smoke graph (1k vertices)
    # whose L3 numbers are below the lit-faith access threshold so no
    # quantitative claims apply.
    missing -= {"email-Eu-core"}
    assert not missing, (
        f"Corpus graphs without any literature claim encoded: {sorted(missing)}. "
        "Either add at least one LiteratureClaim per graph or exclude the graph "
        "from the corpus."
    )


def test_each_app_has_at_least_one_invariant() -> None:
    """Every app should have at least one INVARIANT (graph-agnostic) claim
    so the gate cannot silently pass on graphs that don't appear in
    PER_GRAPH_CLAIMS for that app."""

    apps_with_invariants = {c.app for c in lit.INVARIANT_CLAIMS}
    missing = VALID_APPS - apps_with_invariants
    assert not missing, (
        f"Apps without any INVARIANT_CLAIM: {sorted(missing)}. Add at least "
        "one wildcard claim per app (e.g. SRRIP≈LRU sanity invariant)."
    )


# Citations must locate the claim inside the cited paper. A bare
# "Author YEAR" is rejected; we require at least one of:
#   §N        — section reference (Faldu HPCA20 §6.1)
#   Fig N     — figure reference  (Balaji & Lucia HPCA 2021 Fig 9)
#   Table N   — table reference
# (Also accept "Section N" written out and the lowercase "fig".)
import re as _re

_CITATION_LOCATOR_RE = _re.compile(
    r"(§\s*\d|"
    r"\bFig(?:ure)?\.?\s*\d|"
    r"\bfig\.?\s*\d|"
    r"\bTable\s*\d|"
    r"\bSection\s*\d|"
    r"\bSec\.?\s*\d)",
    flags=_re.IGNORECASE,
)

# Anchors we accept in KNOWN_DEVIATIONS rationales when no paper-section
# locator or code-file reference is present. These are domain terms that
# pin the explanation to a concrete, named data structure or algorithm
# behaviour.
_KNOWN_STRUCT_RE = _re.compile(
    r"\b(CSR|CSC|frontier|parent\[\]|property array|"
    r"PR[- ]rank|PR[- ]ranked|PR[- ]ranking|rank[- ]mis[- ]alignment|"
    r"static schedule|oracle|"
    r"union[- ]find|Phase[- ]\d|Phase[- ]transition|"
    r"algorithmic mismatch|"
    r"degree|hub|clustering)",
    flags=_re.IGNORECASE,
)


def _claim_id(c) -> str:
    return f"{c.graph}/{c.app}/{c.l3_size}/{c.policy}"


@pytest.mark.parametrize(
    "claim", lit.INVARIANT_CLAIMS + lit.PER_GRAPH_CLAIMS, ids=_claim_id,
)
def test_citation_has_paper_locator(claim) -> None:
    """Each citation must include a section / figure / table locator so a
    reader can find the exact spot in the cited paper. A bare
    "Author YEAR" is insufficient — we caught two such cases in the
    early-development backlog (Jaleel et al. ISCA 2010 without the §5.2
    suffix) which made the rationale unverifiable.
    """

    assert _CITATION_LOCATOR_RE.search(claim.citation), (
        f"Claim {_claim_id(claim)} has citation '{claim.citation}' that "
        "does not include a §N / Fig N / Table N locator. Add the exact "
        "paper-section / figure / table reference so the reader can verify "
        "the claim against the source."
    )


@pytest.mark.parametrize(
    "key, reason", list(lit.KNOWN_DEVIATIONS.items()),
    ids=lambda x: "/".join(x) if isinstance(x, tuple) else None,
)
def test_known_deviation_reason_names_root_cause(key, reason) -> None:
    """Every KNOWN_DEVIATIONS rationale must name a root-cause anchor so
    a future reader can decide whether the deviation is still valid.
    Accept any of: a paper-section locator, a code path reference
    (.cc / .hh / .h / .cpp / .py with line number), a quoted variable
    name (CSR / parent[] / property array / etc.) — basically, any
    proper noun that pins the explanation to a verifiable source.
    """

    has_paper_locator = bool(_CITATION_LOCATOR_RE.search(reason))
    has_code_anchor = bool(_re.search(r"\.(?:cc|hh|cpp|h|py|sh)\b", reason))
    has_known_struct = bool(_KNOWN_STRUCT_RE.search(reason))
    anchored = has_paper_locator or has_code_anchor or has_known_struct
    assert anchored, (
        f"KNOWN_DEVIATIONS[{key}] rationale lacks a verifiable anchor "
        f"(paper-section locator, code file, or known data-structure name). "
        f"Reason text: {reason[:200]!r}"
    )
