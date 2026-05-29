"""Tests for scripts/experiments/ecg/confidence_dashboard.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DASH_PATH = REPO_ROOT / "scripts" / "experiments" / "ecg" / "confidence_dashboard.py"


def _load_dashboard():
    spec = importlib.util.spec_from_file_location("confidence_dashboard", DASH_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # required for @dataclass to resolve annotations.
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dash():
    return _load_dashboard()


def _suite(dash, label="Tier A", passed=5, failed=0, errors=0, skipped=0):
    return dash.SuiteResult(
        label=label, short=label, path="x.py",
        passed=passed, failed=failed, skipped=skipped,
        xfailed=0, xpassed=0, errors=errors,
        runtime_s=0.1, raw_tail="",
    )


def test_parse_pytest_summary_passed_only(dash):
    out = dash._parse_pytest_summary("===== 12 passed in 0.5s =====")
    assert out["passed"] == 12
    assert out["failed"] == 0
    assert out["errors"] == 0


def test_parse_pytest_summary_with_failures(dash):
    txt = "==== 4 passed, 2 failed, 1 skipped in 0.5s ===="
    out = dash._parse_pytest_summary(txt)
    assert out["passed"] == 4
    assert out["failed"] == 2
    assert out["skipped"] == 1


def test_parse_pytest_summary_with_xfailed(dash):
    txt = "== 8 passed, 3 skipped, 1 xfailed in 0.2s =="
    out = dash._parse_pytest_summary(txt)
    assert out["passed"] == 8
    assert out["xfailed"] == 1
    assert out["failed"] == 0


def test_parse_pytest_summary_with_errors(dash):
    txt = "==== 3 passed, 2 errors in 0.1s ===="
    out = dash._parse_pytest_summary(txt)
    assert out["passed"] == 3
    assert out["errors"] == 2


def test_headline_green_when_all_pass(dash):
    results = [_suite(dash)]
    lit = {"summary": {"disagree": 0, "claims_total": 10, "ok": 10}}
    assert "GREEN" in dash._headline_verdict(results, lit)


def test_headline_red_when_suite_fails(dash):
    results = [_suite(dash, failed=1)]
    lit = {"summary": {"disagree": 0}}
    verdict = dash._headline_verdict(results, lit)
    assert "RED" in verdict
    assert "Tier A" in verdict


def test_headline_red_when_lit_disagrees(dash):
    results = [_suite(dash)]
    lit = {"summary": {"disagree": 3}}
    verdict = dash._headline_verdict(results, lit)
    assert "RED" in verdict
    assert "3 unexplained" in verdict


def test_render_includes_all_sections(dash):
    results = [_suite(dash)]
    lit = {"summary": {"claims_total": 10, "ok": 10, "disagree": 0,
                       "within_tolerance": 0, "known_deviation": 0,
                       "insufficient_data": 0, "missing": 0}}
    corpus = [{"graph": "g1", "nodes": 100, "edges": 1000,
               "features": {"hub_concentration": 0.5, "avg_degree": 10.0,
                            "clustering_coeff": 0.1, "working_set_ratio": 0.5}}]
    md = dash.render(results, lit, corpus)
    assert "Headline" in md
    assert "Tier & gate pytest results" in md
    assert "Literature-faithfulness comparator" in md
    assert "Corpus diversity coverage" in md
    assert "Graphs profiled:** 1" in md


def test_corpus_section_handles_list_and_dict(dash):
    as_list = [{"graph": "g1", "nodes": 1, "edges": 2,
                "features": {"hub_concentration": 0.1, "avg_degree": 1.0,
                             "clustering_coeff": 0.0, "working_set_ratio": 0.0}}]
    as_dict = {"graphs": as_list}
    a = "\n".join(dash._corpus_section(as_list))
    b = "\n".join(dash._corpus_section(as_dict))
    assert a == b
    assert "g1" in a


def test_budget_section_renders_kind_table_and_fragile_rows(dash):
    budget = {
        "summary": {
            "cells_total": 20,
            "cells_in_distribution": 15,
            "min_margin_pp": 0.5,
            "p10_margin_pp": 0.7,
            "median_margin_pp": 3.0,
            "p90_margin_pp": 7.0,
            "max_margin_pp": 10.0,
            "by_kind": {
                "cache_policy": {"n": 10, "min_pp": 1.0, "median_pp": 3.5},
                "popt_ge_grasp": {"n": 5, "min_pp": 0.5, "median_pp": 2.0},
            },
        },
        "fragile_cache_policy_cells": [
            {"graph": "G", "app": "pr", "l3_size": "1MB",
             "policy": "SRRIP", "delta_pct": -2.5, "margin_pp": 1.0},
        ],
    }
    md = "\n".join(dash._budget_section(budget))
    assert "Regression budget" in md
    assert "cache_policy" in md
    assert "popt_ge_grasp" in md
    assert "5 most fragile cache-policy cells" in md
    assert "G | pr | 1MB" in md


def test_budget_section_missing_input_emits_stub(dash):
    md = "\n".join(dash._budget_section(None))
    assert "No regression_budget JSON found" in md


def test_render_includes_budget_section_when_provided(dash):
    results = [_suite(dash)]
    lit = {"summary": {"claims_total": 1, "ok": 1, "disagree": 0,
                       "within_tolerance": 0, "known_deviation": 0,
                       "insufficient_data": 0, "missing": 0}}
    corpus = []
    budget = {
        "summary": {
            "cells_total": 1, "cells_in_distribution": 1,
            "min_margin_pp": 2.0, "p10_margin_pp": 2.0,
            "median_margin_pp": 2.0, "p90_margin_pp": 2.0,
            "max_margin_pp": 2.0, "by_kind": {},
        },
        "fragile_cache_policy_cells": [],
    }
    md = dash.render(results, lit, corpus, budget=budget)
    assert "Regression budget" in md
    assert "Min margin (any kind): **2.000 pp**" in md


def test_lit_faith_section_missing_input(dash):
    out = "\n".join(dash._lit_faith_section(None))
    assert "No literature_faithfulness JSON" in out
