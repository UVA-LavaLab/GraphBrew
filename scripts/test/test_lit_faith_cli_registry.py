"""Pytest for gate 260 — GAPBS kernel CLI vocabulary registry."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import is_dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN = REPO_ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_cli_registry.py"


def _load():
    spec = importlib.util.spec_from_file_location("gate260", GEN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gate260"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gate():
    return _load()


@pytest.fixture(scope="module")
def data(gate):
    return gate.audit()


# --- shape ----------------------------------------------------------------

def test_module_loads(gate):
    assert hasattr(gate, "audit")
    assert hasattr(gate, "CL_CLASSES")
    assert hasattr(gate, "KERNEL_CL_CLASS")
    assert hasattr(gate, "FLAG_PURPOSE")
    assert hasattr(gate, "main")


def test_clclass_is_frozen_dataclass(gate):
    assert is_dataclass(gate.CLClass)
    sample = gate.CL_CLASSES[0]
    with pytest.raises(Exception):
        sample.name = "mutated"  # type: ignore[misc]


def test_audit_top_level_keys(data):
    for k in (
        "status", "n_cl_classes", "n_kernels", "n_distinct_flags",
        "n_kernel_source_checks", "n_kernel_source_ok",
        "cl_classes", "kernels", "kernel_source_status",
        "flag_purpose", "flag_arity", "rules", "violations",
    ):
        assert k in data, f"missing key {k}"


def test_active(data):
    assert data["status"] == "active"


def test_no_violations(data):
    assert data["violations"] == [], data["violations"]


# --- CL class universe ----------------------------------------------------

def test_six_cl_classes(data):
    names = [c["name"] for c in data["cl_classes"]]
    assert names == [
        "CLBase", "CLApp", "CLIterApp", "CLPageRank", "CLDelta", "CLConvert"
    ]


def test_clbase_has_no_parent(data):
    by_name = {c["name"]: c for c in data["cl_classes"]}
    assert by_name["CLBase"]["parent"] is None
    for n in ("CLApp", "CLIterApp", "CLPageRank", "CLDelta", "CLConvert"):
        assert by_name[n]["parent"] is not None


def test_chains_terminate_at_clbase(data):
    for c in data["cl_classes"]:
        assert c["chain"][-1] == "CLBase"


# --- kernel→class mapping ------------------------------------------------

def test_eleven_kernels(data):
    assert data["n_kernels"] == 11


@pytest.mark.parametrize("kernel,expected", [
    ("pr", "CLPageRank"),
    ("pr_spmv", "CLPageRank"),
    ("bfs", "CLApp"),
    ("sssp", "CLDelta"),
    ("bc", "CLIterApp"),
    ("cc", "CLApp"),
    ("cc_sv", "CLApp"),
    ("tc", "CLApp"),
    ("tc_p", "CLApp"),
    ("ecg_preprocess", "CLApp"),
    ("converter", "CLConvert"),
])
def test_kernel_cl_class(data, kernel, expected):
    by_kernel = {k["kernel"]: k for k in data["kernels"]}
    assert by_kernel[kernel]["cl_class"] == expected


def test_pr_has_tolerance(data):
    by_kernel = {k["kernel"]: k for k in data["kernels"]}
    assert "t" in by_kernel["pr"]["full_flags"]
    assert "t" in by_kernel["pr_spmv"]["full_flags"]


def test_bfs_lacks_tolerance(data):
    by_kernel = {k["kernel"]: k for k in data["kernels"]}
    # CLApp does NOT include -t; bfs would reject -t 1e-6.
    assert "t" not in by_kernel["bfs"]["full_flags"]


def test_bfs_lacks_iters(data):
    by_kernel = {k["kernel"]: k for k in data["kernels"]}
    # CLApp does NOT include -i; bfs would silently ignore -i sweeps.
    assert "i" not in by_kernel["bfs"]["full_flags"]


def test_sssp_has_delta(data):
    by_kernel = {k["kernel"]: k for k in data["kernels"]}
    assert "d" in by_kernel["sssp"]["full_flags"]


def test_bc_has_iters(data):
    by_kernel = {k["kernel"]: k for k in data["kernels"]}
    assert "i" in by_kernel["bc"]["full_flags"]


def test_converter_has_convert_flags(data):
    by_kernel = {k["kernel"]: k for k in data["kernels"]}
    for f in ("e", "b", "x", "q", "p", "y", "V", "w"):
        assert f in by_kernel["converter"]["full_flags"]


def test_all_kernels_have_ordering_flag(data):
    # -o is the POPT/GRASP entry-point; every kernel MUST accept it.
    for k in data["kernels"]:
        assert "o" in k["full_flags"]


def test_all_kernels_have_graph_loader(data):
    for k in data["kernels"]:
        assert "f" in k["full_flags"]
        assert "g" in k["full_flags"]


# --- src-on-disk parity --------------------------------------------------

def test_kernel_source_ok_count(data):
    assert data["n_kernel_source_ok"] == data["n_kernel_source_checks"]


def test_kernel_source_status_nonempty(data):
    assert data["n_kernel_source_checks"] > 0


def test_pr_source_uses_clpagerank(data):
    rows = [r for r in data["kernel_source_status"] if r["kernel"] == "pr"]
    assert rows
    for r in rows:
        assert r["cl"] == "CLPageRank"
        assert r["ok"] is True


def test_bfs_source_uses_clapp(data):
    rows = [r for r in data["kernel_source_status"] if r["kernel"] == "bfs"]
    assert rows
    for r in rows:
        assert r["cl"] == "CLApp"
        assert r["ok"] is True


# --- flag universe --------------------------------------------------------

def test_flag_count(data):
    assert data["n_distinct_flags"] == 28


def test_every_flag_has_purpose(data):
    for f in data["flag_arity"]:
        assert f in data["flag_purpose"], f"flag {f!r} missing purpose"


def test_flag_arity_consistent(data):
    # -f, -g, -k, -u, -m, -o, -j, -D, -n, -r, -i, -t, -d, -b, -e,
    # -x, -q, -p, -y, -V all take values; -h, -s, -z, -S, -l, -a,
    # -v, -w do not.
    takes_value = {f for f, t in data["flag_arity"].items() if t}
    no_value = {f for f, t in data["flag_arity"].items() if not t}
    for f in ("f", "g", "k", "u", "m", "o", "j", "D",
              "n", "r", "i", "t", "d", "b", "e", "x", "q", "p", "y", "V"):
        assert f in takes_value, f"flag {f!r} should take a value"
    for f in ("h", "s", "z", "S", "l", "a", "v", "w"):
        assert f in no_value, f"flag {f!r} should NOT take a value"


def test_clbase_full_flags_subset_of_clapp(data):
    by_name = {c["name"]: c for c in data["cl_classes"]}
    base = set(by_name["CLBase"]["full_flags"])
    app = set(by_name["CLApp"]["full_flags"])
    assert base.issubset(app), base - app


def test_clapp_full_flags_subset_of_descendants(data):
    by_name = {c["name"]: c for c in data["cl_classes"]}
    app = set(by_name["CLApp"]["full_flags"])
    for n in ("CLIterApp", "CLPageRank", "CLDelta"):
        assert app.issubset(set(by_name[n]["full_flags"])), (
            f"{n} missing CLApp flags"
        )


# --- artifact round-trip --------------------------------------------------

def test_cli_round_trip(gate, tmp_path: Path):
    j = tmp_path / "x.json"
    m = tmp_path / "x.md"
    c = tmp_path / "x.csv"
    rc = subprocess.run(
        [sys.executable, str(GEN),
         "--json-out", str(j),
         "--md-out", str(m),
         "--csv-out", str(c)],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    payload = json.loads(j.read_text())
    assert payload["status"] == "active"
    assert payload["violations"] == []
    txt = m.read_text()
    assert "Gate 260" in txt
    assert txt.endswith("\n") and not txt.endswith("\n\n")
    csv_txt = c.read_text()
    assert "cl_class" in csv_txt
    assert "kernel" in csv_txt
    assert "flag" in csv_txt
