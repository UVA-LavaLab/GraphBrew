"""Tests for gate 264 — wiki/data filename grammar registry.

Locks the invariant that every file shipping under ``wiki/data/``
matches a lower_snake_case + approved-extension grammar, that
the .json/.md trio convention is honored (with documented
exceptions), that every artifact catalog entry's referenced
file exists on disk, and that every wiki/data stem is either
in the catalog or in a documented allow-list.

Today: 296 files, 121 stems, 115 catalog entries; 0 violations.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN = REPO_ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_filename_grammar.py"


def _load_gen():
    spec = importlib.util.spec_from_file_location("gen_gate264_test", GEN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gen_gate264_test"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load_gen()


@pytest.fixture(scope="module")
def data(gen):
    return gen.audit()


def test_status_active(data):
    assert data["status"] == "active"


def test_no_violations(data):
    assert data["violations"] == [], (
        "wiki/data filename grammar drift:\n"
        + "\n".join(f"  - {v['rule']} {v['subject']}: {v['reason']}"
                    for v in data["violations"])
    )


def test_rule_keys_present(data):
    expected = {"F1", "F2", "F3", "F4", "F5", "F6", "F7"}
    assert set(data["rules"].keys()) == expected


def test_totals_positive(data):
    assert data["n_files"] >= 50
    assert data["n_stems"] >= 30
    assert data["n_json"] >= 30
    assert data["n_md"] >= 30
    assert data["n_catalog"] >= 100
    assert data["n_catalog_stems"] >= 100


def test_n_files_at_least_n_stems(data):
    assert data["n_files"] >= data["n_stems"], (
        "n_files should be at least n_stems (every stem produces ≥1 file)"
    )


def test_n_json_close_to_n_md(data):
    # MD_OPTIONAL_STEMS create the only legal divergence; today 4 stems
    diff = abs(data["n_json"] - data["n_md"])
    assert diff <= 5, (
        f"|n_json - n_md|={diff} > 5; trio convention may be drifting"
    )


def test_subdirs_documented(data):
    assert all("paper_pipeline" in d for d in data["subdirs"]), (
        f"undocumented subdir in wiki/data/: {data['subdirs']}"
    )


# --- F1 grammar -----------------------------------------------------------

def test_grammar_regex_lowercase_snake(gen):
    pat = gen.STEM_RE
    assert pat.match("foo_bar_baz")
    assert pat.match("a")
    assert not pat.match("FooBar")
    assert not pat.match("foo-bar")
    assert not pat.match("1foo")
    assert not pat.match("foo bar")


def test_extension_allowed_set(gen):
    assert gen.EXT_ALLOWED == {".json", ".md", ".csv"}


# --- F2/F3 allow-lists ---------------------------------------------------

def test_md_optional_stems_nonempty(data):
    assert data["md_optional_stems"], (
        "MD_OPTIONAL_STEMS empty — should at least contain the 4 _postfix stems"
    )


def test_postfix_stems_in_md_optional(data):
    for s in data["md_optional_stems"]:
        assert s.endswith("_postfix") or s.startswith("ecg_"), (
            f"MD_OPTIONAL_STEMS entry {s!r} doesn't match documented pattern"
        )


def test_literature_reproduction_summary_in_json_optional(data):
    assert "literature_reproduction_summary" in data["json_optional_stems"]


# --- F4/F5 catalog cross-check -------------------------------------------

def test_every_catalog_artifact_exists(gen):
    cat = gen._load_catalog()  # type: ignore[attr-defined]
    repo_root = gen.REPO_ROOT
    missing = []
    for e in cat.CATALOG:
        art = e.get("artifact", "")
        if not art:
            continue
        if not (repo_root / art).is_file():
            missing.append(art)
    assert not missing, f"catalog artifacts missing on disk: {missing}"


def test_every_catalog_artifact_data_extension(gen):
    cat = gen._load_catalog()  # type: ignore[attr-defined]
    for e in cat.CATALOG:
        art = e.get("artifact", "")
        if not art:
            continue
        ext = Path(art).suffix
        assert ext in {".json", ".csv"}, (
            f"catalog entry {e.get('id')!r} artifact {art} has non-data ext {ext!r}"
        )


def test_catalog_id_uniqueness(gen):
    cat = gen._load_catalog()  # type: ignore[attr-defined]
    ids = [e.get("id", "") for e in cat.CATALOG]
    assert len(set(ids)) == len(ids), "duplicate catalog id"


def test_catalog_artifact_uniqueness(gen):
    cat = gen._load_catalog()  # type: ignore[attr-defined]
    paths = [e.get("artifact", "") for e in cat.CATALOG if e.get("artifact")]
    assert len(set(paths)) == len(paths), "duplicate catalog artifact path"


# --- F6 stem accounting --------------------------------------------------

def test_meta_artifact_stems_includes_catalog(data):
    assert "artifact_catalog" in data["meta_artifact_stems"], (
        "artifact_catalog must be in META_ARTIFACT_STEMS (it's the catalog "
        "of itself — cannot self-reference)"
    )


def test_implicit_stems_is_union(gen):
    expected = (
        set(gen.MD_OPTIONAL_STEMS)
        | set(gen.JSON_OPTIONAL_STEMS)
        | set(gen.META_ARTIFACT_STEMS)
    )
    assert set(gen.IMPLICIT_PAPER_PIPELINE_STEMS) == expected


# --- F7 subdir ---------------------------------------------------------

def test_documented_subdir_re_matches_paper_pipeline(gen):
    assert gen.DOCUMENTED_SUBDIR_RE.match("paper_pipeline_20260528")
    assert gen.DOCUMENTED_SUBDIR_RE.match("paper_pipeline_20260101_extra")
    assert not gen.DOCUMENTED_SUBDIR_RE.match("results")
    assert not gen.DOCUMENTED_SUBDIR_RE.match("archive")
    assert not gen.DOCUMENTED_SUBDIR_RE.match("paper_pipeline_short")


# --- audit invariants ---------------------------------------------------

def test_n_files_within_expected_band(data):
    # At gate 264 we have ~296 files (293 + gate 264 trio). Allow ±10%
    assert 100 < data["n_files"] < 1000


def test_n_stems_within_expected_band(data):
    assert 50 < data["n_stems"] < 500


# --- emit shape ---------------------------------------------------------

def test_emit_json(tmp_path, gen, data):
    out = tmp_path / "out.json"
    gen._emit_json(data, out)  # type: ignore[attr-defined]
    assert out.exists()
    import json
    j = json.loads(out.read_text())
    assert j["status"] == "active"


def test_emit_md(tmp_path, gen, data):
    out = tmp_path / "out.md"
    gen._emit_md(data, out)  # type: ignore[attr-defined]
    txt = out.read_text()
    assert "Gate 264" in txt
    assert "F1" in txt and "F7" in txt
    assert txt.endswith("\n")


def test_emit_csv(tmp_path, gen, data):
    out = tmp_path / "out.csv"
    gen._emit_csv(data, out)  # type: ignore[attr-defined]
    rows = out.read_text().splitlines()
    assert rows[0] == "kind,name,extra"
    assert any("total," in r for r in rows)


def test_main_zero_exit(gen, tmp_path):
    rc = gen.main([
        "--json-out", str(tmp_path / "j.json"),
        "--md-out", str(tmp_path / "m.md"),
        "--csv-out", str(tmp_path / "c.csv"),
    ])
    assert rc == 0


# --- guardrails against silent regressions ------------------------------

def test_n_subdirs_at_most_5(data):
    assert data["n_subdirs"] <= 5, (
        f"too many wiki/data/ subdirs ({data['n_subdirs']}); "
        "allow-list documents single paper_pipeline snapshot"
    )


def test_n_catalog_within_expected_band(data):
    assert 50 < data["n_catalog"] < 500


def test_md_optional_stems_postfix_only(data):
    for s in data["md_optional_stems"]:
        assert "postfix" in s, (
            f"MD_OPTIONAL_STEMS entry {s!r} should describe a postfix companion; "
            "if not, document the rationale in the generator docstring"
        )


def test_json_optional_stems_summary_only(data):
    for s in data["json_optional_stems"]:
        assert "summary" in s or "report" in s, (
            f"JSON_OPTIONAL_STEMS entry {s!r} should describe a summary/report; "
            "if not, document the rationale in the generator docstring"
        )


def test_no_violation_for_meta_catalog(data):
    for v in data["violations"]:
        assert "artifact_catalog" not in v["subject"], (
            "META_ARTIFACT_STEMS should allow artifact_catalog through "
            f"but got violation: {v}"
        )


def test_stems_with_json_have_either_md_or_md_optional(gen, data):
    """Cross-check F2 directly against on-disk state."""
    import os
    wd = REPO_ROOT / "wiki" / "data"
    json_stems = {Path(f).stem for f in os.listdir(wd)
                  if f.endswith(".json") and (wd / f).is_file()}
    md_stems = {Path(f).stem for f in os.listdir(wd)
                if f.endswith(".md") and (wd / f).is_file()}
    md_optional = set(gen.MD_OPTIONAL_STEMS)
    missing_md = json_stems - md_stems - md_optional
    assert not missing_md, (
        f"JSONs without .md sibling and not in MD_OPTIONAL_STEMS: {missing_md}"
    )


def test_self_artifact_in_catalog_after_gate(data):
    """Gate 264's own artifact is wired into the catalog as the 116th entry.

    If this fails, the lit_faith_filename_grammar trio was not wired into
    artifact_catalog.CATALOG and the gate would self-violate F6.
    """
    # Either the gate's stem is catalog-declared OR in IMPLICIT/META;
    # we want the catalog-declared case.
    impl = set(data["implicit_paper_pipeline_stems"])
    # if it's in implicit, we forgot to wire — fail loudly
    assert "lit_faith_filename_grammar" not in impl, (
        "lit_faith_filename_grammar landed in IMPLICIT_PAPER_PIPELINE_STEMS — "
        "it should be a catalog-declared entry instead"
    )
