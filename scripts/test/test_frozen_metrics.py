"""The frozen-metric contract must stay in the methodology SSOT and bind RESULTS.

The 2026-07-25 review found that the headline metric had moved five times, each
time after a result unfavourable to K2. `METHODOLOGY.md` now pins the primary
metrics and the rules that stop that recurring. These tests fail if that section
is weakened or deleted, so the contract cannot quietly erode between runs.

A first version of these tests only searched for substrings in the methodology
prose. That was theatre: it passed while `RESULTS.md` still presented the
retracted gem5 timing argument as a live result. The tests below therefore also
parse `RESULTS.md` by section and require that every retracted claim lives
inside a section that a reader cannot mistake for a finding.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODOLOGY = ROOT / "research/ecg-hpca/METHODOLOGY.md"
RESULTS = ROOT / "research/ecg-hpca/RESULTS.md"

# Phrases that only ever belonged to a retracted argument. Each maps to the
# review finding that withdrew it.
RETRACTED_PHRASES = {
    "the decisive number is ipc": "IPC quoted as independent evidence",
    "instruction-bound, not memory-bound": "gem5 result built on superseded rows",
    "instr-normalised time": "counterfactual normalisation presented as a result",
}

# A section is non-authoritative only if its heading says so. Prose disclaimers
# inside an otherwise normal-looking section are not sufficient.
WITHDRAWN_MARKERS = ("WITHDRAWN", "RETRACTION", "KNOWN FLAW", "SUPERSEDED")


def methodology() -> str:
    return METHODOLOGY.read_text(errors="ignore")


def frozen_section() -> str:
    text = methodology()
    assert "## Frozen evaluation metrics" in text, "frozen-metric section deleted"
    section = text.split("## Frozen evaluation metrics", 1)[1]
    return section.split("\n## ", 1)[0].lower()


def results_sections() -> list:
    """Split RESULTS.md into (heading, body) pairs on `###` headings."""
    text = RESULTS.read_text(errors="ignore")
    parts = re.split(r"^###\s+(.*)$", text, flags=re.MULTILINE)
    sections = [("(preamble)", parts[0])]
    for i in range(1, len(parts), 2):
        sections.append((parts[i], parts[i + 1]))
    return sections


def test_frozen_metric_section_exists():
    assert "## Frozen evaluation metrics" in methodology()


def test_primary_metrics_are_time_and_traffic():
    section = frozen_section()
    assert "execution time" in section
    assert "total off-chip traffic" in section
    assert "reported together" in section
    # The traffic unit is pinned so cells cannot shop between DRAM bytes, LLC
    # fill bytes and miss counts.
    assert "memory-controller bytes" in section
    assert "may never be substituted" in section


def test_no_negated_contract_clauses():
    """A negation would satisfy a naive substring search while gutting the rule."""
    section = frozen_section()
    for phrase in ("not required to be reported together", "need not be reported together"):
        assert phrase not in section, "contract negated: " + repr(phrase)


def test_selection_effect_guards_are_documented():
    section = frozen_section()
    # Each guard corresponds to a specific failure found in review.
    assert "may not be changed after seeing results" in section
    assert "prefetcher" in section and "demand" in section
    assert "algebraically" in section          # IPC is not independent evidence
    assert "counterfactual" in section         # instruction-normalised time
    # IPC must not be reintroduced under a softer label.
    assert "corroborating" in section


def test_aggregation_is_pinned_to_geomean():
    """The original wording let an extreme cell justify switching to median."""
    section = frozen_section()
    assert "geometric mean" in section
    assert "never replace it" in section


def test_idealised_mechanisms_cannot_support_performance_claims():
    """The STRIDE8 lead came from an oracle-classified, unbounded prefetcher."""
    section = frozen_section()
    assert "cannot mispredict" in section
    assert "ineligible for a performance claim" in section
    for resource in ("mshr", "queue", "bandwidth"):
        assert resource in section


def test_row_admissibility_fails_closed():
    section = frozen_section()
    assert "timing_valid_for_speedup=1" in section
    assert "superseded" in section
    assert "rejection, not a pass" in section


def test_decision_rule_defines_success():
    """Reporting both primaries is disclosure; success needed its own definition."""
    section = frozen_section()
    assert "tie band" in section
    assert "worst cell" in section
    assert "%" in section, "no numeric effect size declared"


def test_symmetric_overhead_accounting_is_required():
    section = frozen_section()
    assert "every competitor" in section
    # Symmetry must cover mechanism eligibility, not just hierarchy presence.
    assert "identical prefetch eligibility" in section


def test_retraction_remains_visible():
    """The withdrawn conclusions must not be quietly deleted from RESULTS."""
    text = RESULTS.read_text(errors="ignore")
    assert "RETRACTION" in text
    assert "ecg.load2" in text
    assert "idealised" in text.lower() or "idealized" in text.lower()


def test_retracted_claims_live_only_in_withdrawn_sections():
    """The retraction must bind the tables, not just precede them.

    This is the check that the first version of this file lacked: RESULTS.md
    carried the retraction and then presented the retracted gem5 timing
    argument several sections later with no marking at all.
    """
    offenders = []
    for heading, body in results_sections():
        if any(marker in heading.upper() for marker in WITHDRAWN_MARKERS):
            continue
        lowered = body.lower()
        for phrase, reason in RETRACTED_PHRASES.items():
            if phrase in lowered:
                offenders.append(repr(heading) + " still asserts " + repr(phrase) + " (" + reason + ")")
    assert not offenders, "retracted claims presented as live results:\n" + "\n".join(offenders)


def test_withdrawn_sections_carry_an_explicit_banner():
    """A heading marker alone is easy to miss when skimming a table."""
    for heading, body in results_sections():
        # Only withdrawn *results* need the banner. The retraction narrative
        # itself is already unambiguous.
        if not heading.upper().startswith("WITHDRAWN"):
            continue
        assert "withdrawn" in body.lower(), repr(heading) + " lacks a body banner"
        assert "do not cite" in body.lower(), repr(heading) + " lacks a do-not-cite warning"
