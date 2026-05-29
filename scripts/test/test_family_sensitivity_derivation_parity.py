"""Gate 192 (FSE-Der) — derivation parity of family_sensitivity.json.

Locks the byte-for-byte derivation of wiki/data/family_sensitivity.json
from its single upstream (oracle_gap.json#rows) so any silent drift in
the relabel-sensitivity bootstrap generator trips a pytest gate before
the dashboard regen step. Re-runs the generator with the same default
seed (1729) and n_resamples (2000) to assert seed-stability.

Five test groups:
  1. meta:               pinned constants (CANONICAL_FAMILY mapping,
                         ALL_FAMILIES alphabetical tuple, SIGN_CLAIMS
                         list of 7 ordered triples, STABILITY_FLOOR,
                         DEFAULT_N_RESAMPLES + DEFAULT_SEED).
  2. relabelings:        cartesian shape (8 graphs × 4 alternatives
                         = 32 perturbations); canonical-family never
                         appears as new_family; per_claim_flip_count
                         conservation.
  3. canonical claims:   one entry per SIGN_CLAIMS triple in identical
                         order; record shape (claim/family/policy_a/
                         policy_b/frac/mean_a/mean_b/n_a/n_b);
                         frac ∈ [0, 1] when not None; consistency with
                         canonical_state mapping.
  4. flip semantics:     each flipped record carries direction ∈
                         {lost, gained}; flip iff canonical-stable
                         differs from perturbed-stable at the 0.95
                         floor; n_flipping_relabelings == count with
                         non-empty flipped.
  5. byte parity:        full-file byte-for-byte vs build(...).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = REPO_ROOT / "scripts" / "experiments" / "ecg" / "family_sensitivity.py"
ARTIFACT_PATH = REPO_ROOT / "wiki" / "data" / "family_sensitivity.json"
ORACLE_PATH = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"


def _load_gen():
    spec = importlib.util.spec_from_file_location("fse_gen", GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_gen()
ARTIFACT = json.loads(ARTIFACT_PATH.read_text())
ORACLE = json.loads(ORACLE_PATH.read_text())
ROWS = ORACLE["rows"]


# ---------------------------------------------------------------------------
# Group 1 — meta constants & pinned invariants
# ---------------------------------------------------------------------------

def test_meta_canonical_family_pinned_eight_graphs():
    expected = {
        "cit-Patents":     "citation",
        "com-orkut":       "social",
        "delaunay_n19":    "mesh",
        "email-Eu-core":   "social",
        "roadNet-CA":      "road",
        "soc-LiveJournal1": "social",
        "soc-pokec":       "social",
        "web-Google":      "web",
    }
    assert GEN.CANONICAL_FAMILY == expected


def test_meta_all_families_alphabetical_tuple():
    assert GEN.ALL_FAMILIES == ("citation", "mesh", "road", "social", "web")
    assert isinstance(GEN.ALL_FAMILIES, tuple)


def test_meta_canonical_families_subset_of_all():
    assert set(GEN.CANONICAL_FAMILY.values()) <= set(GEN.ALL_FAMILIES)


def test_meta_sign_claims_count_and_order():
    assert len(GEN.SIGN_CLAIMS) == 7
    # Order is load-bearing: must match canonical_claims iteration order.
    assert GEN.SIGN_CLAIMS == [
        ("POPT", "GRASP", "road"),
        ("POPT", "GRASP", "social"),
        ("POPT", "GRASP", "mesh"),
        ("POPT", "GRASP", "citation"),
        ("POPT", "GRASP", "web"),
        ("POPT", "LRU",   "social"),
        ("GRASP", "LRU",  "social"),
    ]


def test_meta_stability_floor_pinned():
    assert GEN.STABILITY_FLOOR == 0.95


def test_meta_defaults_pinned():
    assert GEN.DEFAULT_N_RESAMPLES == 2000
    assert GEN.DEFAULT_SEED == 1729


def test_meta_block_match():
    assert ARTIFACT["meta"] == {
        "n_resamples": GEN.DEFAULT_N_RESAMPLES,
        "seed": GEN.DEFAULT_SEED,
        "stability_floor": GEN.STABILITY_FLOOR,
    }


# ---------------------------------------------------------------------------
# Group 2 — relabelings cartesian shape & conservation
# ---------------------------------------------------------------------------

def test_relabelings_count_is_eight_times_four():
    # 8 graphs × (5 ALL_FAMILIES − 1 canonical) = 32 perturbations
    assert ARTIFACT["n_relabelings"] == 8 * 4
    assert len(ARTIFACT["relabelings"]) == 8 * 4


def test_relabelings_never_self_relabel():
    for row in ARTIFACT["relabelings"]:
        assert row["new_family"] != row["canonical_family"]


def test_relabelings_canonical_family_matches_table():
    for row in ARTIFACT["relabelings"]:
        assert GEN.CANONICAL_FAMILY[row["graph"]] == row["canonical_family"]


def test_relabelings_new_family_in_all_families():
    for row in ARTIFACT["relabelings"]:
        assert row["new_family"] in GEN.ALL_FAMILIES


def test_relabelings_cartesian_completeness():
    seen = {(r["graph"], r["new_family"]) for r in ARTIFACT["relabelings"]}
    expected = {
        (g, f)
        for g, cf in GEN.CANONICAL_FAMILY.items()
        for f in GEN.ALL_FAMILIES
        if f != cf
    }
    assert seen == expected


def test_n_flipping_matches_relabelings_with_nonempty_flipped():
    n = sum(1 for r in ARTIFACT["relabelings"] if r["flipped"])
    assert ARTIFACT["n_flipping_relabelings"] == n


def test_per_claim_flip_count_conservation():
    """Sum of per-claim flip counts equals total flipped entries across
    all relabelings."""
    total = sum(len(r["flipped"]) for r in ARTIFACT["relabelings"])
    assert sum(ARTIFACT["per_claim_flip_count"].values()) == total


def test_per_claim_flip_count_keys_match_claims():
    expected = {f"{a} < {b} on {fam}" for a, b, fam in GEN.SIGN_CLAIMS}
    assert set(ARTIFACT["per_claim_flip_count"].keys()) == expected


def test_per_claim_flip_count_nonneg():
    for v in ARTIFACT["per_claim_flip_count"].values():
        assert v >= 0


# ---------------------------------------------------------------------------
# Group 3 — canonical claims shape & ordering
# ---------------------------------------------------------------------------

def test_canonical_claims_count_matches_sign_claims():
    assert len(ARTIFACT["canonical_claims"]) == len(GEN.SIGN_CLAIMS)


def test_canonical_claims_order_matches_sign_claims():
    for c, (a, b, fam) in zip(ARTIFACT["canonical_claims"], GEN.SIGN_CLAIMS):
        assert c["claim"] == f"{a} < {b} on {fam}"
        assert c["policy_a"] == a
        assert c["policy_b"] == b
        assert c["family"] == fam


def test_canonical_claims_record_shape():
    for c in ARTIFACT["canonical_claims"]:
        assert set(c.keys()) >= {
            "claim", "family", "policy_a", "policy_b",
            "frac", "n_a", "n_b",
        }
        # mean_a/mean_b appear iff n_a>0 and n_b>0 (frac not None)
        if c["frac"] is not None:
            assert "mean_a" in c and "mean_b" in c
        assert c["n_a"] >= 0
        assert c["n_b"] >= 0


def test_canonical_claims_frac_in_unit_interval():
    for c in ARTIFACT["canonical_claims"]:
        if c["frac"] is not None:
            assert 0.0 <= c["frac"] <= 1.0


def test_canonical_state_mirrors_canonical_claims():
    expected = {c["claim"]: c["frac"] for c in ARTIFACT["canonical_claims"]}
    assert ARTIFACT["canonical_state"] == expected


# ---------------------------------------------------------------------------
# Group 4 — flip semantics
# ---------------------------------------------------------------------------

def test_flip_record_shape():
    for r in ARTIFACT["relabelings"]:
        for f in r["flipped"]:
            assert set(f.keys()) == {
                "claim", "frac_canonical", "frac_perturbed", "direction",
            }
            assert f["direction"] in ("lost", "gained")


def test_flip_direction_matches_canonical_stability():
    canonical_by_claim = {
        c["claim"]: c["frac"] for c in ARTIFACT["canonical_claims"]
    }
    floor = GEN.STABILITY_FLOOR
    for r in ARTIFACT["relabelings"]:
        for f in r["flipped"]:
            c_frac = canonical_by_claim[f["claim"]]
            assert f["frac_canonical"] == c_frac
            canonical_stable = c_frac is not None and c_frac >= floor
            perturbed_stable = (
                f["frac_perturbed"] is not None and f["frac_perturbed"] >= floor
            )
            # Must actually be a flip (states differ).
            assert canonical_stable != perturbed_stable
            expected_dir = "lost" if canonical_stable else "gained"
            assert f["direction"] == expected_dir


def test_flip_claim_subset_of_sign_claims():
    valid = {f"{a} < {b} on {fam}" for a, b, fam in GEN.SIGN_CLAIMS}
    for r in ARTIFACT["relabelings"]:
        for f in r["flipped"]:
            assert f["claim"] in valid


def test_apply_reassignment_overrides_only_target_graph():
    rows = [
        {"graph": "g1", "family": "fA"},
        {"graph": "g2", "family": "fB"},
        {"graph": "g1", "family": "fA"},
    ]
    out = GEN._apply_reassignment(rows, "g1", "fZ")
    assert out[0]["family"] == "fZ"
    assert out[1]["family"] == "fB"
    assert out[2]["family"] == "fZ"
    # original rows must be untouched (deep copy of target rows)
    assert rows[0]["family"] == "fA"


# ---------------------------------------------------------------------------
# Group 5 — byte parity (seed-stable rebuild)
# ---------------------------------------------------------------------------

def test_full_artifact_byte_parity(tmp_path):
    """Re-run the full generator with default seed/n_resamples and assert
    byte-identical reproduction of the on-disk artifact (including the
    trailing newline emitted by `json.dumps(...) + '\\n'`)."""
    json_out = tmp_path / "family_sensitivity.json"
    md_out = tmp_path / "family_sensitivity.md"
    rc = GEN.main([
        "--oracle-json", str(ORACLE_PATH),
        "--json-out", str(json_out),
        "--md-out", str(md_out),
        "--n-resamples", str(GEN.DEFAULT_N_RESAMPLES),
        "--seed", str(GEN.DEFAULT_SEED),
    ])
    assert rc == 0
    assert json_out.read_text() == ARTIFACT_PATH.read_text()
