"""Cross-file consistency gate for the ``GRAPH_FAMILY`` mapping.

Several modules and one test file each carry a private copy of the
``GRAPH_FAMILY`` dict that maps a graph name to its family label. These
labels drive per-family bucketing in almost every downstream artifact
(policy_winner_table, oracle_gap_report, literature_deviations_report,
popt_vs_grasp_report, winning_regime_taxonomy, family_saturation_distance,
test_corpus_diversity_floor). If any two copies disagree, downstream
artifacts silently re-bucket the same graph into a different family in
different reports and the dashboard loses its meaning.

Gate 107 locks the topology of the copies: the two "full" maps that
include reserved-for-future graphs must agree; the five "short" maps
that cover only the currently-shipped corpus must agree with each other;
and the short set is a strict subset of the full set with no
disagreement on any shared key.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Two "full" copies that include the reserved-for-future graph tags.
FULL_SOURCES = {
    "policy_winner_table": REPO_ROOT / "scripts" / "experiments" / "ecg" / "policy_winner_table.py",
    "test_corpus_diversity_floor": REPO_ROOT / "scripts" / "test" / "test_corpus_diversity_floor.py",
}

# Five "short" copies that cover only the currently-shipped corpus.
SHORT_SOURCES = {
    "literature_deviations_report": REPO_ROOT / "scripts" / "experiments" / "ecg" / "literature_deviations_report.py",
    "oracle_gap_report":            REPO_ROOT / "scripts" / "experiments" / "ecg" / "oracle_gap_report.py",
    "winning_regime_taxonomy":      REPO_ROOT / "scripts" / "experiments" / "ecg" / "winning_regime_taxonomy.py",
    "popt_vs_grasp_report":         REPO_ROOT / "scripts" / "experiments" / "ecg" / "popt_vs_grasp_report.py",
    "family_saturation_distance":   REPO_ROOT / "scripts" / "experiments" / "ecg" / "family_saturation_distance.py",
}

EXPECTED_FAMILIES = {"citation", "mesh", "road", "social", "web"}
EXPECTED_FULL_SIZE = 11
EXPECTED_SHORT_SIZE = 8
RESERVED_KEYS = {"road-CA", "twitter-2010", "uk-2005"}


def _extract_graph_family(path: Path) -> dict[str, str]:
    """Return the ``GRAPH_FAMILY`` dict literal from ``path`` via AST.

    Uses ``ast.literal_eval`` on the assignment value so we don't import
    or execute the surrounding module.
    """
    if not path.exists():
        pytest.fail(f"missing source file: {path.relative_to(REPO_ROOT)}")
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "GRAPH_FAMILY" in targets:
                return ast.literal_eval(node.value)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == "GRAPH_FAMILY" and node.value is not None:
                return ast.literal_eval(node.value)
    pytest.fail(f"GRAPH_FAMILY assignment not found in {path.relative_to(REPO_ROOT)}")
    return {}  # for type-checkers — unreachable


@pytest.fixture(scope="module")
def full_maps() -> dict[str, dict[str, str]]:
    return {name: _extract_graph_family(path) for name, path in FULL_SOURCES.items()}


@pytest.fixture(scope="module")
def short_maps() -> dict[str, dict[str, str]]:
    return {name: _extract_graph_family(path) for name, path in SHORT_SOURCES.items()}


# ---------------------------------------------------------------------------
# Group A — full-set internal consistency (3)
# ---------------------------------------------------------------------------


def test_full_maps_have_identical_keysets(full_maps: dict) -> None:
    keysets = {name: frozenset(m.keys()) for name, m in full_maps.items()}
    unique = set(keysets.values())
    assert len(unique) == 1, (
        "GRAPH_FAMILY 'full' maps disagree on keys:\n"
        + "\n".join(f"  {name}: {sorted(ks)}" for name, ks in keysets.items())
    )


def test_full_maps_have_identical_values(full_maps: dict) -> None:
    """For every key, every full map maps it to the same family value."""
    reference_name, reference = next(iter(full_maps.items()))
    bad = []
    for name, m in full_maps.items():
        if name == reference_name:
            continue
        for key, value in reference.items():
            other = m.get(key)
            if other != value:
                bad.append((key, f"{reference_name}={value}", f"{name}={other}"))
    assert not bad, f"GRAPH_FAMILY value disagreements between full maps: {bad}"


def test_full_map_has_expected_size_and_families(full_maps: dict) -> None:
    """Each full map has exactly EXPECTED_FULL_SIZE entries and exactly the
    EXPECTED_FAMILIES set as values."""
    for name, m in full_maps.items():
        assert len(m) == EXPECTED_FULL_SIZE, (
            f"{name}: GRAPH_FAMILY size={len(m)} expected={EXPECTED_FULL_SIZE}"
        )
        families = set(m.values())
        assert families == EXPECTED_FAMILIES, (
            f"{name}: GRAPH_FAMILY families={families} expected={EXPECTED_FAMILIES}"
        )


# ---------------------------------------------------------------------------
# Group B — short-set internal consistency (4)
# ---------------------------------------------------------------------------


def test_short_maps_have_identical_keysets(short_maps: dict) -> None:
    keysets = {name: frozenset(m.keys()) for name, m in short_maps.items()}
    unique = set(keysets.values())
    assert len(unique) == 1, (
        "GRAPH_FAMILY 'short' maps disagree on keys:\n"
        + "\n".join(f"  {name}: {sorted(ks)}" for name, ks in keysets.items())
    )


def test_short_maps_have_identical_values(short_maps: dict) -> None:
    reference_name, reference = next(iter(short_maps.items()))
    bad = []
    for name, m in short_maps.items():
        if name == reference_name:
            continue
        for key, value in reference.items():
            other = m.get(key)
            if other != value:
                bad.append((key, f"{reference_name}={value}", f"{name}={other}"))
    assert not bad, f"GRAPH_FAMILY value disagreements between short maps: {bad}"


def test_short_map_has_expected_size(short_maps: dict) -> None:
    """Each short map has exactly EXPECTED_SHORT_SIZE entries (no reserved
    keys leaked in)."""
    bad = []
    for name, m in short_maps.items():
        if len(m) != EXPECTED_SHORT_SIZE:
            bad.append((name, len(m)))
    assert not bad, (
        f"short GRAPH_FAMILY copies with wrong size (expected {EXPECTED_SHORT_SIZE}): {bad}"
    )


def test_short_maps_use_expected_families(short_maps: dict) -> None:
    bad = []
    for name, m in short_maps.items():
        families = set(m.values())
        if families != EXPECTED_FAMILIES:
            bad.append((name, sorted(families)))
    assert not bad, f"short GRAPH_FAMILY copies with unexpected family set: {bad}"


# ---------------------------------------------------------------------------
# Group C — cross-set agreement (3)
# ---------------------------------------------------------------------------


def test_short_keys_subset_of_full_keys(full_maps: dict, short_maps: dict) -> None:
    full_keys = set(next(iter(full_maps.values())).keys())
    bad = []
    for name, m in short_maps.items():
        extras = set(m.keys()) - full_keys
        if extras:
            bad.append((name, sorted(extras)))
    assert not bad, f"short GRAPH_FAMILY copies have keys not in full set: {bad}"


def test_shared_keys_have_matching_values(full_maps: dict, short_maps: dict) -> None:
    """Every key that appears in both a short map and a full map maps to the
    same family value in both."""
    reference_full = next(iter(full_maps.values()))
    bad = []
    for name, m in short_maps.items():
        for key, value in m.items():
            full_value = reference_full.get(key)
            if full_value != value:
                bad.append((name, key, value, full_value))
    assert not bad, f"GRAPH_FAMILY short/full value disagreements: {bad}"


def test_reserved_keys_only_appear_in_full(full_maps: dict, short_maps: dict) -> None:
    """The keys exclusive to the full set are exactly the reserved-for-future
    tags (no short map has crept any of them in, no full map has dropped one)."""
    full_keys = set(next(iter(full_maps.values())).keys())
    short_keys = set(next(iter(short_maps.values())).keys())
    only_full = full_keys - short_keys
    assert only_full == RESERVED_KEYS, (
        f"Full-only keys={sorted(only_full)} expected={sorted(RESERVED_KEYS)}"
    )


# ---------------------------------------------------------------------------
# Group D — hygiene (3)
# ---------------------------------------------------------------------------


def test_no_unknown_or_empty_family_values(full_maps: dict, short_maps: dict) -> None:
    bad = []
    for name, m in {**full_maps, **short_maps}.items():
        for key, value in m.items():
            if not value or value == "unknown":
                bad.append((name, key, value))
    assert not bad, f"GRAPH_FAMILY entries with empty/unknown family: {bad}"


def test_no_duplicate_keys_in_any_map_source(full_maps: dict, short_maps: dict) -> None:
    """``ast.literal_eval`` would silently keep the last occurrence of a
    duplicated key; we re-parse the source line-by-line and count occurrences
    of each key to guarantee no copy carries a duplicate entry."""
    import re
    key_pattern = re.compile(r'^\s*"([^"]+)"\s*:')
    bad = []
    for name, path in {**FULL_SOURCES, **SHORT_SOURCES}.items():
        text = path.read_text()
        # Find the GRAPH_FAMILY block boundaries.
        start = text.find("GRAPH_FAMILY")
        if start < 0:
            continue
        depth = 0
        end = start
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        block = text[start:end]
        seen: dict[str, int] = {}
        for line in block.splitlines():
            m = key_pattern.match(line)
            if not m:
                continue
            key = m.group(1)
            seen[key] = seen.get(key, 0) + 1
        dupes = {k: c for k, c in seen.items() if c > 1}
        if dupes:
            bad.append((name, dupes))
    assert not bad, f"GRAPH_FAMILY copies with duplicate keys: {bad}"


def test_total_graph_universe_matches_full(full_maps: dict, short_maps: dict) -> None:
    """Union of every key across every copy equals the full set (no rogue key
    appears only in one short copy)."""
    full_keys = set(next(iter(full_maps.values())).keys())
    union = set()
    for m in full_maps.values():
        union |= set(m.keys())
    for m in short_maps.values():
        union |= set(m.keys())
    assert union == full_keys, (
        f"union of GRAPH_FAMILY keys={sorted(union)} differs from full={sorted(full_keys)}; "
        f"only-in-union={sorted(union - full_keys)}, only-in-full={sorted(full_keys - union)}"
    )
