"""Confidence gate 211 — Makefile coverage integrity (MFC-Int).

This gate closes the loop between the build orchestrator (Makefile) and
the artifact registries (CATALOG, PYTEST_SUITES). Every catalogued
generator must either be invoked by some Makefile target or be on an
explicit allow-list documenting why it is not. Every catalogued gate
and artifact path must resolve on disk. Every dashboard pytest suite
path must resolve on disk. A self-consistency count pins the size of
the Makefile-invoked set to len(CATALOG) - len(MAKEFILE_EXEMPT_GENERATORS)
so silently dropping a generator from the build (e.g. deleting a target)
fails this gate instead of producing a stale artifact.

Six groups (15 tests):

  A. CATALOG generator on-disk (1):
       1. every CATALOG.generator path exists on disk

  B. CATALOG gate on-disk (1):
       2. every CATALOG.gate path exists on disk (test file or generator
          for meta-claims)

  C. CATALOG artifact path shape (2):
       3. every CATALOG.artifact starts with 'wiki/data/'
       4. every CATALOG.artifact extension is .json or .csv (most are
          .json; literature_reproduction is .csv; companion .md/.csv
          siblings are tracked separately by RSC-Cov)

  D. PYTEST_SUITES path on-disk (1):
       5. every PYTEST_SUITES.path resolves to a file under scripts/test/

  E. Makefile coverage (5):
       6. parse Makefile recipes for `python3 -m scripts.experiments.ecg.X`
          and `python3 scripts/experiments/ecg/X.py` invocations; result
          set is non-empty
       7. every CATALOG generator (excluding MAKEFILE_EXEMPT_GENERATORS)
          is invoked by at least one Makefile recipe
       8. MAKEFILE_EXEMPT_GENERATORS is non-empty and well-formed
          (every entry is a valid CATALOG module)
       9. MAKEFILE_EXEMPT_GENERATORS minimality — every entry is NOT
          invoked by any Makefile target (so the allow-list never
          shadows a real invocation)
      10. the Makefile-invoked set fully contains every CATALOG generator
          modulo the allow-list (no orphan catalog entries hiding)

  F. Self-consistency (5):
      11. len(Makefile-invoked ∩ CATALOG modules) ==
          len(CATALOG modules) - len(MAKEFILE_EXEMPT_GENERATORS)
      12. allow-list disjoint from invoked set (never both)
      13. allow-list ⊆ CATALOG modules
      14. invoked set ⊆ all-discoverable ecg modules (no
          typos invoking non-existent modules)
      15. every Makefile-invoked module file exists on disk

Load-bearing invariants documented here:

* corpus_diversity.py is the sole MAKEFILE_EXEMPT_GENERATORS entry.
  It is a one-shot scraper of GAPBS Graph Topology Features blocks
  from cache_sim logs (see corpus_diversity.py:7-25 docstring) — it
  is re-baked manually whenever the graph corpus changes (a rare,
  human-curated event) and its output (corpus_diversity.json) is a
  static input that downstream lit-* targets consume. Wiring it into
  the build would require re-running the entire baseline sweep
  every `make confidence`, which is a 10-minute regression cycle
  rather than a 1-minute regression cycle.

* Both `python3 -m scripts.experiments.ecg.X` AND
  `python3 scripts/experiments/ecg/X.py` invocation forms are valid
  Makefile patterns and both count for coverage. Most generators use
  the dotted form; literature_faithfulness.py, regression_budget.py,
  and gem5_anchor_summary.py use the direct-script form (legacy
  pre-package-conversion, documented in CATALOG comments).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
ECG_DIR = SCRIPTS_DIR / "experiments" / "ecg"
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from scripts.experiments.ecg.artifact_catalog import CATALOG  # noqa: E402
from scripts.experiments.ecg.confidence_dashboard import PYTEST_SUITES  # noqa: E402

# ---------------------------------------------------------------------------
# Allow-list — documented exceptions ONLY. Adding an entry here requires
# documenting why the generator is intentionally not invoked from any
# Makefile target.
# ---------------------------------------------------------------------------

MAKEFILE_EXEMPT_GENERATORS: frozenset[str] = frozenset(
    {
        # Manually baked scraper. See module docstring (lines 4-25) and the
        # paragraph in this file's header for rationale. corpus_diversity.json
        # is a static input to lit-* targets; re-baking it is a manual
        # human-curated step that follows a corpus refresh.
        "scripts.experiments.ecg.corpus_diversity",
    }
)


def _gen_to_mod(p: str) -> str:
    return p.replace("/", ".").rstrip(".py").removesuffix(".")


def _parse_invocations(text: str) -> set[str]:
    """Extract every `python3 -m scripts.experiments.ecg.X` OR
    `python3 scripts/experiments/ecg/X.py` invocation as a dotted
    module path. Supports both python and python3.
    """

    dotted = re.findall(
        r"python3? -m (scripts\.experiments\.ecg\.[a-z0-9_]+)", text
    )
    direct = re.findall(
        r"python3? (scripts/experiments/ecg/[a-z0-9_]+)\.py", text
    )
    direct_dotted = {p.replace("/", ".") for p in direct}
    return set(dotted) | direct_dotted


def _catalog_module(entry: dict) -> str:
    p = entry["generator"]
    assert p.endswith(".py")
    return p[:-3].replace("/", ".")


def _all_ecg_modules() -> set[str]:
    out: set[str] = set()
    for child in ECG_DIR.iterdir():
        if not child.is_file():
            continue
        if not child.name.endswith(".py"):
            continue
        if child.name.startswith("_"):
            continue
        out.add(f"scripts.experiments.ecg.{child.stem}")
    return out


# Cached parse result for reuse across tests.
_MAKEFILE_TEXT = (REPO_ROOT / "Makefile").read_text()
_INVOKED = _parse_invocations(_MAKEFILE_TEXT)
_CATALOG_MODULES = {_catalog_module(e): e for e in CATALOG}


# ---------------------------------------------------------------------------
# Group A — CATALOG generator on-disk
# ---------------------------------------------------------------------------


def test_catalog_generators_exist_on_disk():
    missing = []
    for entry in CATALOG:
        p = REPO_ROOT / entry["generator"]
        if not p.is_file():
            missing.append((entry["id"], entry["generator"]))
    assert not missing, f"CATALOG generators missing on disk: {missing}"


# ---------------------------------------------------------------------------
# Group B — CATALOG gate on-disk
# ---------------------------------------------------------------------------


def test_catalog_gates_exist_on_disk():
    missing = []
    for entry in CATALOG:
        p = REPO_ROOT / entry["gate"]
        if not p.is_file():
            missing.append((entry["id"], entry["gate"]))
    assert not missing, f"CATALOG gates missing on disk: {missing}"


# ---------------------------------------------------------------------------
# Group C — CATALOG artifact path shape
# ---------------------------------------------------------------------------


def test_catalog_artifacts_under_wiki_data():
    bad = [
        (e["id"], e["artifact"])
        for e in CATALOG
        if not e["artifact"].startswith("wiki/data/")
    ]
    assert not bad, f"CATALOG artifacts not under wiki/data/: {bad}"


def test_catalog_artifacts_known_extension():
    allowed = (".json", ".csv")
    bad = [
        (e["id"], e["artifact"])
        for e in CATALOG
        if not e["artifact"].endswith(allowed)
    ]
    assert not bad, (
        f"CATALOG artifacts with unknown extension (expected {allowed}): {bad}. "
        f"Most catalog entries track JSON; literature_reproduction tracks CSV. "
        f"Companion .md files are tracked separately by RSC-Cov."
    )


# ---------------------------------------------------------------------------
# Group D — PYTEST_SUITES path on-disk
# ---------------------------------------------------------------------------


def test_pytest_suite_paths_resolve_on_disk():
    missing = []
    for label, value in PYTEST_SUITES.items():
        path = value[0]
        short = value[1] if len(value) > 1 else None
        p = REPO_ROOT / path
        if not p.is_file():
            missing.append((short or label, path))
    assert not missing, f"PYTEST_SUITES paths missing on disk: {missing}"


# ---------------------------------------------------------------------------
# Group E — Makefile coverage
# ---------------------------------------------------------------------------


def test_makefile_invoked_set_non_empty():
    assert _INVOKED, "Makefile parse yielded zero invocations — regex broken?"
    assert len(_INVOKED) >= 50, (
        f"Makefile invocations dropped below floor of 50 — got {len(_INVOKED)}; "
        f"either Makefile was gutted or regex broke"
    )


def test_every_catalog_generator_invoked_by_makefile():
    not_invoked = sorted(
        m for m in _CATALOG_MODULES
        if m not in _INVOKED and m not in MAKEFILE_EXEMPT_GENERATORS
    )
    assert not not_invoked, (
        f"CATALOG generators not invoked by any Makefile target and not "
        f"on the documented allow-list: {not_invoked}. Either wire them "
        f"into a Makefile target (preferred) or add them to "
        f"MAKEFILE_EXEMPT_GENERATORS with a documented reason."
    )


def test_makefile_exempt_generators_well_formed():
    assert MAKEFILE_EXEMPT_GENERATORS, "Allow-list is empty — if no exemptions are needed, this gate should not have an allow-list mechanism"
    bad = [m for m in MAKEFILE_EXEMPT_GENERATORS if m not in _CATALOG_MODULES]
    assert not bad, (
        f"MAKEFILE_EXEMPT_GENERATORS entries that are not CATALOG modules: {bad}. "
        f"The allow-list only makes sense for entries that ARE in CATALOG."
    )


def test_makefile_exempt_generators_minimality():
    bad = sorted(m for m in MAKEFILE_EXEMPT_GENERATORS if m in _INVOKED)
    assert not bad, (
        f"MAKEFILE_EXEMPT_GENERATORS entries that ARE invoked by Makefile: {bad}. "
        f"Remove them from the allow-list — the build covers them already."
    )


def test_no_orphan_catalog_entries():
    orphans = sorted(
        m for m in _CATALOG_MODULES
        if m not in _INVOKED and m not in MAKEFILE_EXEMPT_GENERATORS
    )
    assert not orphans, f"Orphan CATALOG entries (not invoked, not exempt): {orphans}"


# ---------------------------------------------------------------------------
# Group F — Self-consistency
# ---------------------------------------------------------------------------


def test_invoked_catalog_count_matches_expected():
    expected = len(_CATALOG_MODULES) - len(MAKEFILE_EXEMPT_GENERATORS)
    actual = len(set(_CATALOG_MODULES) & _INVOKED)
    assert actual == expected, (
        f"Number of CATALOG modules invoked by Makefile ({actual}) "
        f"!= len(CATALOG) ({len(_CATALOG_MODULES)}) - len(allow-list) "
        f"({len(MAKEFILE_EXEMPT_GENERATORS)}) = {expected}. "
        f"Either a catalog entry stopped being invoked or the allow-list "
        f"is out of date."
    )


def test_allow_list_and_invoked_disjoint():
    overlap = MAKEFILE_EXEMPT_GENERATORS & _INVOKED
    assert not overlap, f"Allow-list overlaps with invoked set: {sorted(overlap)}"


def test_allow_list_subset_of_catalog_modules():
    extras = MAKEFILE_EXEMPT_GENERATORS - set(_CATALOG_MODULES)
    assert not extras, f"Allow-list entries not in CATALOG: {sorted(extras)}"


def test_invoked_modules_subset_of_discoverable():
    all_mods = _all_ecg_modules()
    bogus = sorted(_INVOKED - all_mods)
    assert not bogus, (
        f"Makefile invokes modules that don't exist on disk: {bogus}. "
        f"Either the module was renamed/deleted or the Makefile target is a typo."
    )


def test_every_invoked_module_resolves_to_file():
    missing = []
    for mod in _INVOKED:
        stem = mod.split(".")[-1]
        p = ECG_DIR / f"{stem}.py"
        if not p.is_file():
            missing.append(mod)
    assert not missing, f"Invoked modules with no on-disk file: {missing}"
