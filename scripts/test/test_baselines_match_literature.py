"""Literature-baseline faithfulness pytest gate.

This test compares the GraphBrew cache_sim sweep produced under
``/tmp/graphbrew-lit-baseline`` (or ``GRAPHBREW_LIT_BASELINE_ROOT``)
against the published expectations encoded in
:mod:`scripts.experiments.ecg.literature_baselines`.

The test parametrizes per ``(graph, app, l3_size, policy)`` claim so a
single regression localizes immediately rather than collapsing into a
single combined failure.

If a sweep CSV is missing, the test ``skip``s instead of failing — this
keeps CI green while a long-running sweep finishes.

Known deviations live in ``literature_baselines.KNOWN_DEVIATIONS`` and
convert ``disagree`` outcomes into ``xfail`` so a fix flips them to
XPASS and demands an update.
"""

from __future__ import annotations

import importlib.util
import os
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
faith = _load("literature_faithfulness", "literature_faithfulness.py")


def _sweep_root() -> Path:
    env = os.environ.get("GRAPHBREW_LIT_BASELINE_ROOT")
    if env:
        return Path(env)
    return Path("/tmp/graphbrew-lit-baseline")


def _sweep_subdir() -> str:
    return os.environ.get("GRAPHBREW_LIT_BASELINE_SUBDIR", "lit")


def _load_observations():
    root = _sweep_root()
    if not root.exists():
        return None, root
    obs = faith.load_observations(root, _sweep_subdir())
    if not obs:
        return None, root
    return faith.index(obs), root


def _all_concrete_claims():
    """Yield (graph, app, l3, policy, claim) for non-wildcard claims."""
    for claim in lit.PER_GRAPH_CLAIMS:
        if claim.graph in ("*", "*power_law*"):
            continue
        if claim.app == "*" or claim.l3_size == "*":
            continue
        yield (claim.graph, claim.app, claim.l3_size, claim.policy, claim)


_CONCRETE_CLAIMS = list(_all_concrete_claims())


def _min_accesses() -> int:
    try:
        return int(os.environ.get("GRAPHBREW_LIT_MIN_ACCESSES", "10000"))
    except ValueError:
        return 10000


@pytest.mark.parametrize(
    "graph,app,l3_size,policy,claim",
    _CONCRETE_CLAIMS,
    ids=[f"{g}-{a}-{l}-{p}" for g, a, l, p, _ in _CONCRETE_CLAIMS],
)
def test_literature_claim_holds(graph, app, l3_size, policy, claim):
    obs_idx, root = _load_observations()
    if obs_idx is None:
        pytest.skip(f"No literature sweep CSVs under {root}/*/{_sweep_subdir()}/.")
    lru = obs_idx.get((graph, app, l3_size, "LRU"))
    policy_obs = obs_idx.get((graph, app, l3_size, policy))
    if lru is None or policy_obs is None:
        pytest.skip(f"Missing observation for {graph}/{app}/{l3_size}/{policy}")
    min_acc = _min_accesses()
    if max(lru.accesses, policy_obs.accesses) < min_acc:
        pytest.skip(
            f"Insufficient L3 traffic for {graph}/{app}/{l3_size}/{policy} "
            f"(max accesses={max(lru.accesses, policy_obs.accesses)} < {min_acc})"
        )
    delta_pct = (policy_obs.miss_rate - lru.miss_rate) * 100.0
    status = faith._classify(claim, delta_pct)
    key = (graph, app, l3_size, policy)
    if status == "disagree":
        reason = lit.KNOWN_DEVIATIONS.get(key)
        if reason:
            pytest.xfail(
                f"Known deviation for {graph}/{app}/{l3_size}/{policy} "
                f"(Δ={delta_pct:+.3f}pp, expected sign={claim.expected_sign}): {reason}"
            )
        pytest.fail(
            f"{graph}/{app}/{l3_size}/{policy} disagrees with literature "
            f"(observed Δ={delta_pct:+.3f}pp vs expected sign={claim.expected_sign}, "
            f"min_abs={claim.min_abs_delta_pct}, max_abs={claim.max_abs_delta_pct}, "
            f"tolerance={claim.tolerance_pct}). Source: {claim.citation}. "
            f"If this deviation is intentional, register it in "
            f"literature_baselines.KNOWN_DEVIATIONS."
        )
    # ok or within_tolerance counts as pass.


def test_popt_at_least_as_good_as_grasp_when_both_present():
    obs_idx, root = _load_observations()
    if obs_idx is None:
        pytest.skip(f"No literature sweep CSVs under {root}/*/{_sweep_subdir()}/.")
    min_acc = _min_accesses()
    failures: list[str] = []
    for (graph, app, l3, policy), obs in obs_idx.items():
        if policy != "POPT":
            continue
        grasp = obs_idx.get((graph, app, l3, "GRASP"))
        if grasp is None:
            continue
        if max(obs.accesses, grasp.accesses) < min_acc:
            continue
        diff_pct = (obs.miss_rate - grasp.miss_rate) * 100.0
        if diff_pct > 1.0:  # POPT worse than GRASP by >1pp is suspect
            key = (graph, app, l3, "POPT_GE_GRASP")
            if key in lit.KNOWN_DEVIATIONS:
                continue
            failures.append(
                f"{graph}/{app}/L3={l3}: POPT miss_rate={obs.miss_rate:.5f} "
                f"vs GRASP={grasp.miss_rate:.5f} (Δ=+{diff_pct:.3f}pp, "
                f"accesses={obs.accesses})"
            )
    if failures:
        pytest.fail(
            "P-OPT is an oracle and must not lose to GRASP by more than 1pp:\n  "
            + "\n  ".join(failures)
        )


def test_sweep_root_has_some_observations():
    obs_idx, root = _load_observations()
    if obs_idx is None:
        pytest.skip(f"No literature sweep CSVs under {root}/*/{_sweep_subdir()}/.")
    assert len(obs_idx) > 0, f"No observations parsed under {root}"
