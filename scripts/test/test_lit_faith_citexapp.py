"""LIT-CXApp (gate 228): pytest invariants for cross-app rationale coherence.

Locks the zero-contradiction, full-sign-alignment, common-kernel, and
length-span invariants on `wiki/data/lit_faith_citexapp.json`. The
floors are intentionally tight (zero contradictions, zero sign misses,
≤ 3.0x length span) because the lit-faith corpus is small enough
(< 25 groups) that genuine drift is editable in minutes and should
always be fixed at the source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "wiki" / "data" / "lit_faith_citexapp.json"


@pytest.fixture(scope="module")
def audit() -> dict:
    assert AUDIT_PATH.exists(), (
        f"{AUDIT_PATH} missing — run `make lit-citexapp` (gate 228)"
    )
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def test_schema_keys(audit: dict) -> None:
    for key in ("schema_version", "summary", "sign_vocab", "opposing_pairs",
                "length_span_ceiling", "groups"):
        assert key in audit, f"missing top-level key {key!r}"
    assert audit["schema_version"] == 1


def test_length_span_ceiling_pinning(audit: dict) -> None:
    assert audit["length_span_ceiling"] == 3.0


def test_group_count_floor(audit: dict) -> None:
    s = audit["summary"]
    assert s["group_count"] >= 12, (
        f"(citation, sign) group count {s['group_count']} below 12 — "
        "the lit-faith corpus shrank a literature axis."
    )
    assert s["multi_member_group_count"] >= 8


def test_unique_rationale_floor(audit: dict) -> None:
    s = audit["summary"]
    assert s["unique_rationale_total"] >= 25, (
        f"unique rationale total {s['unique_rationale_total']} below 25 — "
        "rationale diversity collapsed."
    )


def test_no_contradictions(audit: dict) -> None:
    s = audit["summary"]
    if s["total_contradictions"]:
        details = []
        for g in audit["groups"]:
            for c in g["contradictions"]:
                details.append(
                    f"\n  sign={g['expected_sign']} cite={g['citation'][:50]}"
                    f"\n    A: {c['rationale_a'][:100]}"
                    f"\n    B: {c['rationale_b'][:100]}"
                )
        pytest.fail(
            f"{s['total_contradictions']} rationale contradiction(s):"
            + "".join(details[:5])
        )


def test_no_sign_misses(audit: dict) -> None:
    s = audit["summary"]
    if s["sign_alignment_misses"]:
        details = []
        for g in audit["groups"]:
            for r in g["rationales"]:
                if not r["sign_aligned"]:
                    details.append(
                        f"\n  sign={g['expected_sign']} "
                        f"cite={g['citation'][:50]}"
                        f"\n    rationale: {r['rationale'][:120]}"
                    )
        pytest.fail(
            f"{s['sign_alignment_misses']} rationale(s) missing sign vocabulary:"
            + "".join(details[:5])
        )


def test_no_kernel_failures(audit: dict) -> None:
    s = audit["summary"]
    if s["groups_failing_kernel"]:
        bad = [g for g in audit["groups"] if not g["kernel_ok"]]
        pytest.fail(
            f"{s['groups_failing_kernel']} group(s) have no common-kernel "
            f"vocabulary across rationales: " +
            "; ".join(f"sign={g['expected_sign']} cite={g['citation'][:40]}"
                       for g in bad[:5])
        )


def test_no_length_span_failures(audit: dict) -> None:
    s = audit["summary"]
    if s["groups_failing_span"]:
        bad = [g for g in audit["groups"] if not g["length_span_ok"]]
        pytest.fail(
            f"{s['groups_failing_span']} group(s) exceed length-span ratio "
            f"3.0×: " +
            "; ".join(
                f"cite={g['citation'][:40]} ratio={g['length_span_ratio']:.2f}"
                for g in bad[:5]
            )
        )


def test_sign_vocab_completeness(audit: dict) -> None:
    sv = audit["sign_vocab"]
    for sign in ("-", "+", "~"):
        assert sign in sv, f"sign_vocab missing {sign!r}"
        assert len(sv[sign]) >= 5, f"sign_vocab[{sign!r}] suspiciously short"
    assert "dominates" in sv["-"]
    assert "worse"     in sv["+"]
    assert "near"      in sv["~"]


def test_opposing_pairs_present(audit: dict) -> None:
    pairs = audit["opposing_pairs"]
    assert len(pairs) >= 2
    flat_pos = {t for p in pairs for t in p["pos"]}
    flat_neg = {t for p in pairs for t in p["neg"]}
    assert "outperforms" in flat_pos
    assert "underperforms" in flat_neg


def test_every_group_has_member_count(audit: dict) -> None:
    for g in audit["groups"]:
        assert g["member_count"] >= 1
        assert g["rationale_count"] >= 1
        assert g["rationale_count"] <= g["member_count"]


def test_group_rationale_member_count_sum(audit: dict) -> None:
    """Per-rationale member counts must sum to the group's member count."""
    for g in audit["groups"]:
        per_rt_total = sum(r["member_count"] for r in g["rationales"])
        assert per_rt_total == g["member_count"], (
            f"group sign={g['expected_sign']} cite={g['citation'][:30]} "
            f"per-rationale member sum {per_rt_total} != "
            f"group member_count {g['member_count']}"
        )


def test_large_groups_have_at_least_one_kernel_term(audit: dict) -> None:
    """Any (citation, sign) group with >=5 rationales must share at least
    2 kernel terms — these large groups carry the most evidentiary weight
    and need strong inter-rationale anchoring."""
    for g in audit["groups"]:
        if g["rationale_count"] >= 5:
            assert g["kernel_size"] >= 2, (
                f"large group sign={g['expected_sign']} "
                f"cite={g['citation'][:40]} has only {g['kernel_size']} "
                "kernel terms across rationales"
            )


def test_expected_signs_all_known(audit: dict) -> None:
    for g in audit["groups"]:
        assert g["expected_sign"] in ("-", "+", "~"), (
            f"unknown sign {g['expected_sign']!r} on cite={g['citation'][:40]}"
        )


def test_app_coverage_floor(audit: dict) -> None:
    """Across all groups, the union of apps must cover at least 4 apps —
    catches a regression where the corpus loses an app axis."""
    apps = {a for g in audit["groups"] for a in g["app_set"]}
    assert len(apps) >= 4, f"only {len(apps)} apps across all groups: {apps}"


def test_graph_coverage_floor(audit: dict) -> None:
    """Across all groups, the union of graphs must cover at least 5
    graphs."""
    graphs = {g_ for g in audit["groups"] for g_ in g["graph_set"]}
    assert len(graphs) >= 5, f"only {len(graphs)} graphs across all groups"


def test_policy_coverage_floor(audit: dict) -> None:
    policies = {p for g in audit["groups"] for p in g["policy_set"]}
    assert len(policies) >= 3, f"only {len(policies)} policies"


def test_negation_context_handling_via_directional_tokens(audit: dict) -> None:
    """Smoke test: at least one rationale in the corpus contains a
    negated form like 'must NOT regress' or 'should not regress' — this
    is the regression that motivated the negation-context detector. If
    the corpus stops containing such phrasing, the negation detector is
    dead code and we want to know."""
    found = False
    for g in audit["groups"]:
        for r in g["rationales"]:
            t = r["rationale"].lower()
            if "not regress" in t or "no regression" in t or "must not" in t:
                found = True
                break
        if found:
            break
    assert found, (
        "no negated directional phrasing found in any rationale; the "
        "negation-context detector in lit_faith_citexapp is unexercised"
    )


def test_dedup_per_group(audit: dict) -> None:
    seen = set()
    for g in audit["groups"]:
        k = (g["citation"], g["expected_sign"])
        assert k not in seen, f"duplicate group key {k}"
        seen.add(k)


def test_summary_counts_consistent(audit: dict) -> None:
    s = audit["summary"]
    assert s["group_count"] == len(audit["groups"])
    assert s["unique_rationale_total"] == sum(g["rationale_count"]
                                              for g in audit["groups"])
    assert s["total_contradictions"] == sum(g["contradiction_count"]
                                            for g in audit["groups"])
