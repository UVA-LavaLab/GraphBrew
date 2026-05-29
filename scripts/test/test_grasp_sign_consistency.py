"""Tier C: GRASP-vs-LRU sign consistency across simulators.

Each test pair joins the cache_sim reference sweep with the gem5 and/or
Sniper sweeps and asserts that the *sign* of ``miss_rate(GRASP) - miss_rate(LRU)``
agrees with cache_sim at the mandatory L3 sizes (4kB and 32kB).

Sweeps live under ``/tmp/graphbrew-grasp-{cache,gem5,sniper}-sweep`` (the
locations the handoff documents). If the cache_sim reference is missing,
all tests are skipped — they are intentionally hermetic w.r.t. recomputing
the sweeps so that they can run as quick post-sweep gates without spending
hours on simulation.

A successful gem5/sniper sweep must produce ``DBG/roi_matrix.csv``; if a
simulator sweep is missing the matching simulator test is skipped, but the
cache_sim self-consistency check still runs.

Known disagreements (see ``KNOWN_DISAGREEMENTS`` below) are converted to
``xfail`` so they remain visible in CI without permanently failing the
build. If the underlying issue is later fixed, the test becomes ``XPASS``
and the entry should be removed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIGN_SCRIPT = PROJECT_ROOT / "scripts" / "experiments" / "ecg" / "sign_consistency.py"
_spec = importlib.util.spec_from_file_location("sign_consistency", SIGN_SCRIPT)
sign_consistency = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["sign_consistency"] = sign_consistency
_spec.loader.exec_module(sign_consistency)

CACHE_ROOT = Path(
    os.environ.get(
        "GRAPHBREW_TIERC_CACHE_ROOT",
        "/tmp/graphbrew-grasp-cache-sweep",
    )
)
GEM5_ROOT = Path(
    os.environ.get(
        "GRAPHBREW_TIERC_GEM5_ROOT",
        "/tmp/graphbrew-grasp-gem5-sweep",
    )
)
SNIPER_ROOT = Path(
    os.environ.get(
        "GRAPHBREW_TIERC_SNIPER_ROOT",
        "/tmp/graphbrew-grasp-sniper-sweep",
    )
)

PAIRS: list[tuple[str, str]] = [
    ("email-Eu-core", "pr"),
    ("email-Eu-core", "bc"),
    ("email-Eu-core", "sssp"),
    ("email-Eu-core", "bfs"),
    ("cit-Patents", "pr"),
    ("cit-Patents", "bc"),
    ("cit-Patents", "sssp"),
    ("cit-Patents", "bfs"),
]

# (graph, app, simulator) -> human-readable reason. Disagreements at the
# matching simulator that involve only the listed (graph, app, simulator)
# tuple are converted to xfail. When the underlying issue is fixed,
# remove the entry so the test will XPASS and alert maintainers.
KNOWN_DISAGREEMENTS: dict[tuple[str, str, str], str] = {
    ("email-Eu-core", "pr", "gem5"): (
        "gem5 at L3=4kB: both SRRIP and GRASP miss-rate jump well above LRU "
        "(GRASP delta +0.21 vs cache_sim −0.02). cache_sim and Sniper agree "
        "GRASP slightly helps at 4kB. See wiki/POPT-GRASP-Faithfulness-Audit.md "
        "(Tier C section) — likely a gem5 RRPV-insertion or hot-region masking "
        "issue affecting SRRIP-family policies under aggressive L3 capacity "
        "pressure."
    ),
    ("email-Eu-core", "sssp", "sniper"): (
        "Sniper sssp on email-Eu-core: cache_sim deltas are noise-level "
        "(|delta| ≤ 4e-4) while Sniper shows |delta| up to 3.3e-2 with "
        "opposite sign at 4kB and 32kB. The working set is too small "
        "(1005 nodes) to drive a stable miss-rate signal; sign agreement "
        "is undefined when cache_sim's delta is below the noise floor. "
        "Tracked separately from the headline lit-faithfulness gates."
    ),
    ("cit-Patents", "sssp", "sniper"): (
        "Sniper sssp on cit-Patents disagrees at 4kB: cache_sim shows GRASP "
        "marginally worse (+0.5pp) while Sniper shows GRASP slightly better "
        "(−6.4pp). At 32kB/256kB Sniper deltas collapse to ~0pp while "
        "cache_sim still shows GRASP +7-9pp worse. Suspected cause: Sniper "
        "TimingSimpleCPU has different miss queue / MSHR pressure under "
        "Bellman-Ford-style relaxations, which masks the GRASP penalty "
        "that cache_sim's pure functional model exposes. See "
        "wiki/POPT-GRASP-Faithfulness-Audit.md (Tier C section)."
    ),
    ("cit-Patents", "bfs", "sniper"): (
        "Sniper bfs on cit-Patents disagrees at 4kB and 32kB: cache_sim "
        "shows GRASP substantially worse (+36pp at 4kB, +31pp at 32kB) "
        "while Sniper shows Sniper-BFS deltas at near-zero (+/-2pp). BFS "
        "produces a single-visit traversal with limited temporal reuse, so "
        "Sniper's pipeline + write-back dynamics dampen the policy-level "
        "miss-rate signal that cache_sim's tight loop magnifies. Tracked "
        "as a documented Tier C limitation."
    ),
}


def _cache_csv(graph: str, app: str) -> Path:
    return CACHE_ROOT / f"{graph}-{app}" / "DBG" / "roi_matrix.csv"


def _sim_csv(root: Path, graph: str, app: str) -> Path:
    return root / f"{graph}-{app}" / "DBG" / "roi_matrix.csv"


@pytest.mark.parametrize("graph,app", PAIRS, ids=[f"{g}-{a}" for g, a in PAIRS])
def test_cache_sim_reference_present(graph: str, app: str) -> None:
    path = _cache_csv(graph, app)
    if not path.exists():
        pytest.skip(
            f"cache_sim reference {path} not produced yet — re-run the "
            "Tier C cache_sim sweep first."
        )
    rows = sign_consistency.load_roi_matrix(path, "cache_sim", graph, app)
    assert rows, f"{path} has no rows with simulator=cache_sim"
    policies = {r.policy for r in rows}
    assert {"LRU", "GRASP"}.issubset(policies), (
        f"{path} missing LRU/GRASP rows; got {sorted(policies)}"
    )
    sizes = {r.l3_size for r in rows}
    assert {"4kB", "32kB"}.issubset(sizes), (
        f"{path} missing mandatory L3 sizes; got {sorted(sizes)}"
    )


def _evaluate_pair(simulator: str, sim_root: Path, graph: str, app: str) -> dict:
    if not sim_root.exists():
        pytest.skip(f"{simulator} sweep root {sim_root} not present")
    sim_csv = _sim_csv(sim_root, graph, app)
    if not sim_csv.exists():
        pytest.skip(
            f"{simulator} roi_matrix.csv missing for {graph}/{app}; "
            "sweep may still be running or workload unsupported "
            "(e.g. Sniper does not yet have a BC kernel)"
        )
    cache_csv = _cache_csv(graph, app)
    if not cache_csv.exists():
        pytest.skip(f"cache_sim reference missing for {graph}/{app}")
    return sign_consistency.evaluate(
        cache_root=CACHE_ROOT,
        gem5_root=sim_root if simulator == "gem5" else None,
        sniper_root=sim_root if simulator == "sniper" else None,
        pairs=[(graph, app)],
    )


def _format_violations(records: list[dict]) -> str:
    out: list[str] = []
    for r in records:
        sim = r["simulator"]
        out.append(
            f"  - graph={r['graph']} app={r['app']} l3={r['l3_size']} "
            f"cache_sim={r.get('cache_sim_sign')}/{r.get('cache_sim_delta')} "
            f"{sim}={r.get(f'{sim}_sign')}/{r.get(f'{sim}_delta')}"
        )
    return "\n".join(out)


def _assert_or_xfail(
    simulator: str, graph: str, app: str, summary: dict
) -> None:
    violations = summary["mandatory_violations"]
    if not violations:
        return
    key = (graph, app, simulator)
    reason = KNOWN_DISAGREEMENTS.get(key)
    if reason is not None:
        pytest.xfail(
            f"known {simulator} disagreement for {graph}/{app}: {reason}\n"
            f"violations:\n{_format_violations(violations)}"
        )
    pytest.fail(
        f"{simulator} GRASP-vs-LRU sign disagrees with cache_sim at "
        f"mandatory L3 sizes for {graph}/{app}:\n"
        f"{_format_violations(violations)}\n"
        f"warnings (non-mandatory): {summary['warnings']}\n"
        f"If this is a known investigation, add ('{graph}', '{app}', "
        f"'{simulator}') to KNOWN_DISAGREEMENTS with a reason linking to the "
        f"audit doc."
    )


@pytest.mark.parametrize("graph,app", PAIRS, ids=[f"{g}-{a}" for g, a in PAIRS])
def test_gem5_sign_agreement_at_small_l3(graph: str, app: str) -> None:
    summary = _evaluate_pair("gem5", GEM5_ROOT, graph, app)
    _assert_or_xfail("gem5", graph, app, summary)


@pytest.mark.parametrize("graph,app", PAIRS, ids=[f"{g}-{a}" for g, a in PAIRS])
def test_sniper_sign_agreement_at_small_l3(graph: str, app: str) -> None:
    summary = _evaluate_pair("sniper", SNIPER_ROOT, graph, app)
    _assert_or_xfail("sniper", graph, app, summary)
