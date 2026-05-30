"""Tests for gate 245 — L3 regime-classifier consistency.

Locks in:

  * generator emits an "active" status (no scaffold mode);
  * zero violations on the current registry + canonical L3 grid;
  * R1 (every entry resolves to a callable);
  * R2 (byte-input classifiers stay inside their declared vocabulary);
  * R3 (within-family agreement on canonical grid);
  * R4 (source-pattern scan finds nothing unregistered);
  * R5 (non-byte-input classifiers have signature + note);
  * concrete in-family equivalence: policy_winner_table._l3_regime
    and popt_vs_grasp_report._l3_regime really do produce the same
    label on every canonical L3 (forces a real audit if either is
    ever changed independently);
  * concrete cross-family divergence: oracle_gap_report._regime
    really does differ from v1 at 32 kB AND 64 kB (forces a real
    audit if anyone "fixes" oracle_gap_report by editing one
    without re-uniting them);
  * the canonical L3 grid is non-empty and stable;
  * generator's JSON-on-disk matches audit() output (so committed
    artifacts can't silently drift from the live audit).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_regime_classifier.py"
JSON_OUT = ROOT / "wiki" / "data" / "lit_faith_regime_classifier.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load("lit_faith_regime_classifier_dyn", GEN_PATH)


@pytest.fixture(scope="module")
def audit(gen):
    return gen.audit()


def test_status_active(audit):
    assert audit["status"] == "active"


@pytest.mark.parametrize("rule_id", ["R1", "R2", "R3", "R4", "R5"])
def test_rule_documented(audit, rule_id):
    assert rule_id in audit["rules"]


def test_zero_violations(audit):
    vs = audit["violations"]
    assert vs == [], (
        f"gate 245 reports {len(vs)} violation(s); first few:\n"
        + "\n".join(f"  - {v}" for v in vs[:10])
    )


# ---------------------------------------------------------- R1 / R5 / registry shape

def test_registry_nonempty(gen):
    assert gen.REGIME_REGISTRY, "REGIME_REGISTRY must be non-empty"


def test_every_registry_entry_has_required_fields(gen):
    needed = {"path", "func", "family", "vocabulary", "signature"}
    for e in gen.REGIME_REGISTRY:
        missing = needed - set(e.keys())
        assert not missing, f"{e['func']} missing fields: {missing}"


def test_every_registry_entry_resolves(gen):
    for e in gen.REGIME_REGISTRY:
        p = gen.ROOT / e["path"]
        assert p.exists(), f"{e['path']} does not exist"
        mod = _load(f"resolve_{e['func']}", p)
        fn = getattr(mod, e["func"], None)
        assert callable(fn), f"{e['func']} in {e['path']} not callable"


# ---------------------------------------------------------- R2 vocabulary

def test_byte_label_classifiers_stay_in_vocabulary(gen):
    for e in gen.REGIME_REGISTRY:
        if e["signature"] != "byte_label":
            continue
        vocab = set(e["vocabulary"])
        for label in gen.CANONICAL_L3_GRID:
            got = gen._classify_byte_label(e, label)
            assert got in vocab, (
                f"{e['func']}@{e['path']} returned {got!r} for "
                f"{label!r}; not in vocabulary {sorted(vocab)}"
            )


# ---------------------------------------------------------- R3 in-family agreement

def test_tiny_small_large_v1_family_agrees(gen):
    """policy_winner_table._l3_regime and popt_vs_grasp_report._l3_regime
    must produce the SAME label on every canonical L3 — these are
    declared identical siblings, and silent drift between them would
    invisibly re-bucket POPT-vs-GRASP plots vs the winner table."""
    members = [e for e in gen.REGIME_REGISTRY
               if e["family"] == "tiny_small_large_v1"]
    assert len(members) >= 2, "v1 family should have ≥ 2 members"
    for label in gen.CANONICAL_L3_GRID:
        seen = {e["func"] + "@" + e["path"]: gen._classify_byte_label(e, label)
                for e in members}
        distinct = set(seen.values())
        assert len(distinct) == 1, (
            f"v1 family disagrees on {label!r}: {seen}"
        )


# ---------------------------------------------------------- explicit divergence

def test_v1_vs_v2_oracle_gap_divergence_is_real(gen):
    """oracle_gap_report._regime really does differ from v1 at 32 kB
    AND 64 kB. This locks in the *expected* divergence — if anyone
    "fixes" oracle_gap_report by editing one boundary, this test
    flips RED and forces an audit of the per-figure consequences.

    Note: 512 kB is "unknown" to BOTH families (it's not in either's
    L3_SIZE_BYTES table) so it doesn't surface as a regime-classifier
    divergence even though the underlying boundaries differ at 512 kB."""
    v1 = next(e for e in gen.REGIME_REGISTRY
              if e["path"].endswith("policy_winner_table.py"))
    v2 = next(e for e in gen.REGIME_REGISTRY
              if e["path"].endswith("oracle_gap_report.py"))
    # at 32 kB: v1 says "unknown" (32kB not in v1's L3_SIZE_BYTES table)
    # while v2 says "tiny" (oracle_gap_report includes 32kB)
    assert gen._classify_byte_label(v1, "32kB") == "unknown"
    assert gen._classify_byte_label(v2, "32kB") == "tiny"
    # at 64 kB: v1 says "small" (< boundary at exactly 64*1024 = False)
    # while v2 says "tiny" (<= boundary at exactly 64*1024 = True)
    assert gen._classify_byte_label(v1, "64kB") == "small"
    assert gen._classify_byte_label(v2, "64kB") == "tiny"
    # at 256 kB: BOTH return "small" (boundary lands on small/large split
    # the same way despite different operators)
    assert gen._classify_byte_label(v1, "256kB") == "small"
    assert gen._classify_byte_label(v2, "256kB") == "small"


# ---------------------------------------------------------- R4 defensive scan

def test_r4_scan_finds_no_unregistered_classifiers(gen):
    """The source-pattern scan should find every byte-label-shaped
    function we know about — and only those."""
    out = gen._rule_r4(gen.REGIME_REGISTRY)
    assert out == [], (
        f"unregistered regime-classifier functions found:\n"
        + "\n".join(f"  - {o}" for o in out)
    )


# ---------------------------------------------------------- R5

def test_non_byte_label_classifiers_have_notes(gen):
    for e in gen.REGIME_REGISTRY:
        if e["signature"] == "byte_label":
            continue
        assert e.get("note"), (
            f"{e['func']}@{e['path']} has signature={e['signature']} "
            f"but no explanatory note"
        )


# ---------------------------------------------------------- canonical grid shape

def test_canonical_grid_nonempty(gen):
    assert len(gen.CANONICAL_L3_GRID) >= 8, \
        "canonical L3 grid should span tiny..large with enough density"


def test_canonical_grid_has_key_paper_sizes(gen):
    """Every L3 size the paper plots in its main figures should be
    in the canonical grid (sanity guardrail)."""
    needed = {"4kB", "64kB", "256kB", "1MB", "4MB"}
    have = set(gen.CANONICAL_L3_GRID)
    missing = needed - have
    assert not missing, f"canonical grid missing key paper sizes: {missing}"


def test_canonical_grid_in_ascending_order(gen):
    """Reading the canonical grid top-to-bottom should be monotonic
    in bytes — helps reviewers eyeball the rendered table."""
    pw = _load("pw_for_245_order",
               gen.ROOT / "scripts/experiments/ecg/policy_winner_table.py")
    sizes = [pw._l3_bytes(l) for l in gen.CANONICAL_L3_GRID]
    # _l3_bytes returns -1 for "unknown to this module" labels; only check
    # monotonicity over the labels that policy_winner_table knows
    known = [(l, b) for l, b in zip(gen.CANONICAL_L3_GRID, sizes) if b >= 0]
    if len(known) >= 2:
        bs = [b for _, b in known]
        assert bs == sorted(bs), \
            f"known-label byte sizes not ascending: {known}"


# ---------------------------------------------------------- on-disk parity

def test_committed_json_matches_audit(audit):
    assert JSON_OUT.exists(), (
        f"{JSON_OUT} missing; run `make lit-regime-classifier`"
    )
    on_disk = json.loads(JSON_OUT.read_text())
    for k in ("status", "rules", "registry_size", "family_count",
              "canonical_grid", "totals"):
        assert on_disk.get(k) == audit.get(k), (
            f"committed JSON drifts from audit() at key {k!r}; "
            "run `make lit-regime-classifier` to refresh"
        )


# ---------------------------------------------------------- family inventory

def test_families_documented(audit):
    """Every family that appears in the registry must show up in
    audit['families'] with vocabulary + signature + members."""
    fams = audit["families"]
    assert fams, "audit must list families"
    for name, info in fams.items():
        assert info["vocabulary"], f"family {name} has empty vocabulary"
        assert info["signature"], f"family {name} has empty signature"
        assert info["members"], f"family {name} has no members"


def test_at_least_one_family_has_multiple_members(audit):
    """If no family has ≥2 members, R3 is vacuous. The whole point of
    the gate is to catch in-family drift; require at least one shared
    family today (tiny_small_large_v1)."""
    multi = [name for name, info in audit["families"].items()
             if len(info["members"]) >= 2]
    assert multi, "no family has multiple members — R3 would be vacuous"
