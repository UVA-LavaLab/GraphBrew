"""Structural pytest gate for ECG validation gates.

While ``test_ecg_validation_gates.py`` tests behaviour with synthetic
proof-matrix rows, this file enforces *catalog* invariants between
``scripts/experiments/ecg/proof_matrix.py`` (the source of every
ablation label we run) and
``scripts/experiments/ecg/ecg_validation_gates.py`` (the source of every
gate verdict we emit).

The invariants we check:

  * every ECG_* / GRASP_DBG_only / POPT_only / *_PFX ablation referenced
    in the gate evaluator exists in ``proof_matrix.ABLATIONS`` — i.e.,
    we never have a gate that references an ablation we don't actually
    run, and vice-versa we never silently drop an ablation from the
    gate report.

  * every gate emitted by ``evaluate()`` (using a fully populated
    synthetic row set) has a non-empty ``notes`` string, with reasoning
    that at least mentions either the candidate, the baseline, or the
    metric. This is the ECG-side equivalent of the citation-locator
    discipline we enforce on ``literature_baselines.KNOWN_DEVIATIONS``.

  * every gate has one of the four known statuses (``pass``,
    ``activation_only``, ``fail``, ``missing``).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ECG_DIR = REPO_ROOT / "scripts" / "experiments" / "ecg"


def _load(mod_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, ECG_DIR / file_name)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


gates = _load("ecg_validation_gates", "ecg_validation_gates.py")
proof = _load("proof_matrix", "proof_matrix.py")


CANONICAL_ABLATIONS = {a.label for a in proof.ABLATIONS}


def _row(benchmark: str, ablation: str, memory: int, useful: int = 0, fills: int = 0) -> dict[str, str]:
    group = "ecg_replacement" if ablation.startswith("ECG_") else "cache_alone"
    return {
        "benchmark": benchmark,
        "ablation": ablation,
        "ablation_group": group,
        "status": "ok",
        "memory_accesses": str(memory),
        "total_memory_traffic": str(memory),
        "l3_misses": str(memory),
        "l3_size": "4kB",
        "prefetch_useful": str(useful),
        "prefetch_fills": str(fills),
    }


def _fully_populated_rows() -> list[dict[str, str]]:
    """Provide one row per canonical ablation so evaluate() can emit
    every gate (no ``missing`` verdicts) — used to enumerate the gate
    catalog deterministically."""

    rows: list[dict[str, str]] = []
    for i, label in enumerate(sorted(CANONICAL_ABLATIONS)):
        # PFX_* rows need positive useful prefetches so pfx_gate can
        # classify them as pass/activation_only rather than fail.
        useful = 1 if label.startswith("PFX_") or label.endswith("_PFX") else 0
        rows.append(_row("pr", label, 100 - i, useful=useful, fills=useful))
    return rows


def _evaluate() -> list[dict[str, str]]:
    return gates.evaluate(
        _fully_populated_rows(),
        metric="memory_accesses",
        parity_tolerance=0.05,
        benefit_tolerance=0.0,
        embedded_tolerance=0.10,
    )


VALID_STATUSES = {"pass", "activation_only", "fail", "missing"}


def test_evaluate_emits_only_known_statuses() -> None:
    for gate in _evaluate():
        assert gate["status"] in VALID_STATUSES, (
            f"gate {gate['gate']!r} reports unknown status {gate['status']!r}; "
            f"expected one of {sorted(VALID_STATUSES)}"
        )


def test_every_gate_has_non_empty_notes() -> None:
    bad = [g for g in _evaluate() if not (g.get("notes") or "").strip()]
    assert not bad, (
        "These gates emit empty notes (notes must explain the verdict):\n"
        + "\n".join(f"  {g['benchmark']}/{g['gate']}" for g in bad)
    )


_NOTE_ANCHOR_RE = re.compile(
    r"(ECG|GRASP|POPT|LRU|SRRIP|PFX|DBG|memory|cache|baseline|candidate|prefetch|"
    r"tolerance|insertion|replacement|hybrid|adaptive|oracle|stronger|"
    r"matched|match|hint|tiebreak|epoch|embedded)",
    re.IGNORECASE,
)


def test_gate_notes_reference_labels_or_concepts() -> None:
    """Each note must mention one of the ECG concepts (ECG / GRASP / POPT /
    LRU / SRRIP / PFX / DBG / cache / baseline / candidate / prefetch /
    tolerance / hint / tiebreak / embedded / adaptive / oracle …) so the
    reader doesn't see a free-form string disconnected from the gate."""

    bad: list[str] = []
    for g in _evaluate():
        note = g.get("notes", "")
        if not _NOTE_ANCHOR_RE.search(note):
            bad.append(
                f"{g['benchmark']}/{g['gate']}: {note!r} has no anchor "
                f"keyword (expected ECG/GRASP/POPT/LRU/SRRIP/PFX/DBG/…)"
            )
    assert not bad, "\n".join(bad)


def test_every_gate_references_canonical_ablation_or_special() -> None:
    """Each ``candidate`` / ``baseline`` slot must be either empty
    (missing-gate row) or a canonical ablation label from
    ``proof_matrix.ABLATIONS``. Catches typos in the gate evaluator and
    silent ablation renames."""

    bad: list[str] = []
    for g in _evaluate():
        for slot in ("candidate", "baseline"):
            label = g.get(slot, "")
            if label and label not in CANONICAL_ABLATIONS:
                bad.append(
                    f"{g['benchmark']}/{g['gate']}.{slot} = {label!r} "
                    f"is not in proof_matrix.ABLATIONS"
                )
    assert not bad, "\n".join(bad)


def test_gate_catalog_is_non_empty_and_unique_per_benchmark() -> None:
    """Sanity: at least one gate per benchmark and no duplicate gate IDs
    per benchmark (would mean a gate function is being called twice)."""

    seen: dict[tuple[str, str], int] = {}
    for g in _evaluate():
        key = (g["benchmark"], g["gate"])
        seen[key] = seen.get(key, 0) + 1
    assert seen, "evaluate() produced no gates at all"
    dupes = {k: c for k, c in seen.items() if c > 1}
    assert not dupes, f"duplicate gate IDs emitted: {dupes}"


@pytest.mark.parametrize("ablation", sorted(CANONICAL_ABLATIONS))
def test_canonical_ablation_is_used_by_at_least_one_gate(ablation: str) -> None:
    """Forces every ablation we run to be evaluated by at least one
    gate. Catches the silent-skip case where someone adds a new
    ``ECG_FOOBAR`` ablation to ``proof_matrix.ABLATIONS`` but forgets
    to wire it into a gate."""

    used: set[str] = set()
    for g in _evaluate():
        for slot in ("candidate", "baseline"):
            label = g.get(slot, "")
            if label:
                used.add(label)
    # Some ablations are deliberately not evaluated as a primary
    # candidate/baseline (they only exist for sensitivity sweeps), so
    # we accept the explicit allowlist below.
    not_required_to_be_gated = {
        # PFX_degree_only is only used as a sensitivity probe — there is
        # no degree-only-replacement gate by design.
        "PFX_degree_only",
        # SRRIP_cache_only is collected as a generic-replacement
        # reference point, but ecg_validation_gates measures against
        # GRASP_DBG_only / POPT_only as the prior-art baselines (per
        # Balaji HPCA21's framing). SRRIP shows up in the per-cell
        # report but is not used as a gate baseline.
        "SRRIP_cache_only",
    }
    if ablation in not_required_to_be_gated:
        pytest.skip(f"{ablation} is on the not-required-to-be-gated allowlist")
    assert ablation in used, (
        f"canonical ablation {ablation!r} is not referenced by any "
        f"ecg_validation_gates verdict — either add a gate for it or "
        f"put it on the allowlist in this test."
    )
