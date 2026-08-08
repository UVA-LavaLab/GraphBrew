"""The frozen-metric contract must stay in the normative SSOT.

A 2026-07-25 review found the headline metric had moved five times, each time
after a result unfavourable to K2. `PAPER.md` Section 5 ("Metrics and claim
rules") pins the primary metrics and the rules that stop that recurring.
These tests fail if that section is weakened or deleted, so the contract
cannot quietly erode between edits.

`PAPER.md` is the sole normative document (see
`test_ecg_paper_ssot.py::test_paper_ssot_is_sole_normative_doc_with_correct_simulator_roles_and_scope`);
the former `METHODOLOGY.md` this file used to check is retired, and its
frozen-metrics section is merged into `PAPER.md` Section 5. `RESULTS.md` is
now current-measurements-only (no chronological lab notebook), so the
withdrawn/retraction-banner checks that used to bind historical sections of
`RESULTS.md` no longer apply and are not reproduced here; that history lives
in git and in `evidence/` instead.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "research/ecg-hpca/PAPER.md"
RESULTS = ROOT / "research/ecg-hpca/RESULTS.md"


def paper() -> str:
    return PAPER.read_text(errors="ignore")


def metrics_section() -> str:
    text = paper()
    assert "## 5. Metrics and claim rules" in text, "metrics/claim-rules section deleted"
    section = text.split("## 5. Metrics and claim rules", 1)[1]
    section = section.split("\n## 6.", 1)[0].lower()
    return " ".join(section.split())


def test_metrics_and_claim_rules_section_exists():
    assert "## 5. Metrics and claim rules" in paper()


def test_primary_metrics_are_time_and_traffic():
    section = metrics_section()
    assert "execution time" in section
    assert "total off-chip traffic" in section
    assert "always reported together" in section
    assert "memory-controller bytes" in section


def test_no_negated_contract_clauses():
    """A negation would satisfy a naive substring search while gutting the rule."""
    section = metrics_section()
    for phrase in (
            "not required to be reported together",
            "need not be reported together"):
        assert phrase not in section, "contract negated: " + repr(phrase)


def test_selection_effect_and_ipc_guards_are_documented():
    section = metrics_section()
    assert "may not be changed after seeing results" in section
    assert "algebraically" in section          # IPC is not independent evidence
    assert "counterfactual" in section         # instruction-normalised time
    # IPC must not be reintroduced under a softer label.
    assert "corroborating" in section


def test_aggregation_is_pinned_to_geomean():
    """The original wording let an extreme cell justify switching to median."""
    section = metrics_section()
    assert "geometric mean" in section
    assert "never replace it" in section


def test_tie_band_and_comparison_scope_are_pinned():
    section = metrics_section()
    assert "+/-2%" in section
    assert "own baseline cell" in section, (
        "the per-invocation baseline requirement is the operative part of "
        "the rule; without it 'prefer within-run' is advice, not a method")


def test_row_admissibility_fails_closed():
    section = metrics_section()
    assert "timing_valid_for_speedup=0" in section
    assert "never a speedup claim" in section


def test_symmetric_overhead_accounting_is_required():
    section = metrics_section()
    assert "every competitor" in section
    assert "identical prefetch eligibility" in section


def test_prefetcher_demand_misses_are_not_performance_evidence():
    section = metrics_section()
    assert "prefetcher active" in section
    assert "demand-miss reduction is not performance evidence" in section
    assert "total off-chip traffic" in section


def test_idealized_mechanisms_are_upper_bounds_not_measurements():
    section = metrics_section()
    assert "cannot mispredict" in section
    assert "upper bound, not a measured performance result" in section
    assert "finite latency" in section
    assert "mshr" in section


def test_within_build_rule_is_required():
    section = metrics_section()
    assert "within-build comparison" in section
    assert "may not mix rows from different" in section
    assert "re-run the local baseline" in section


def test_instruction_disclosure_rule_is_a_claim_classification_not_parity_guard():
    """Unequal instruction counts must be disclosed, not required to match.

    This is the explicit rule requested in the paper-SSOT consolidation: it
    replaces an implicit strict-parity expectation with a stated
    claim-classification split between complete-design claims (instruction
    inequality allowed, disclosed) and replacement-policy-alone claims
    (instruction parity required).
    """
    section = paper().split("### 5.1 Instruction-count disclosure rule", 1)[1]
    section = section.split("\n## 6.", 1)[0].lower()
    assert "not a requirement that instruction counts must" in section
    assert "k2-rrip+streamshield versus k2-lru+streamshield" in section
    assert "replacement-policy-alone" in section
    assert "before promoting a complete-design speedup claim" in section
    assert "asymmetric compiler/control-flow specialization" in section


def test_results_is_current_measurements_only_and_non_normative():
    """RESULTS.md must not re-introduce a chronological lab-notebook narrative.

    The consolidation deliberately removed withdrawn/retracted historical
    sections from RESULTS.md; this test guards against that content
    creeping back in, rather than (as before) checking that it was
    correctly banner-marked.
    """
    text = RESULTS.read_text(errors="ignore")
    assert "Non-normative" in text
    for marker in ("WITHDRAWN", "RETRACTION", "SUPERSEDED-RESULT"):
        assert marker not in text, (
            f"RESULTS.md should contain current tables only; found a "
            f"{marker!r} marker implying withdrawn history was reintroduced")
