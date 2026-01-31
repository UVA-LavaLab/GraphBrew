import pytest
#!/usr/bin/env python3
"""
Test Weight Flow - Verify weights are generated and read from correct locations.

This test verifies:
1. Python writes weights to scripts/weights/active/
2. C++ reads from scripts/weights/active/
3. Weight merger saves runs to scripts/weights/runs/
4. Weight merger merges to scripts/weights/merged/
5. Use-run and use-merged copy to active/

Usage:
    python -m scripts.test.test_weight_flow
    python -m scripts.test.test_weight_flow -v  # verbose
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.utils import WEIGHTS_DIR, ACTIVE_WEIGHTS_DIR
from scripts.lib.weights import (
    DEFAULT_WEIGHTS_DIR,
    save_type_weights,
    load_type_weights,
    save_type_registry,
    load_type_registry,
)
from scripts.lib.weight_merger import (
    get_weights_dir,
    get_active_dir,
    get_runs_dir,
    get_merged_dir,
    save_current_run,
    use_run,
    use_merged,
    list_runs,
)


class ResultsTracker:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def check(self, condition: bool, message: str):
        """Check a condition and record result."""
        if condition:
            self.passed += 1
            print(f"  ✓ {message}")
        else:
            self.failed += 1
            self.errors.append(message)
            print(f"  ✗ {message}")
    
    def summary(self):
        """Print summary."""
        print()
        print("=" * 60)
        total = self.passed + self.failed
        print(f"Results: {self.passed}/{total} passed")
        if self.errors:
            print("\nFailed tests:")
            for err in self.errors:
                print(f"  - {err}")
        return self.failed == 0


@pytest.fixture
def results():
    return ResultsTracker()


@pytest.fixture(autouse=True)
def weights_env(tmp_path, monkeypatch):
    import scripts.lib.utils as utils
    import scripts.lib.weights as weights
    import scripts.lib.weight_merger as wm

    tmp_weights = tmp_path / "weights"
    tmp_active = tmp_weights / "active"
    tmp_runs = tmp_weights / "runs"
    tmp_merged = tmp_weights / "merged"
    for d in [tmp_active, tmp_runs, tmp_merged]:
        d.mkdir(parents=True, exist_ok=True)

    # Patch utils
    monkeypatch.setattr(utils, "WEIGHTS_DIR", tmp_weights)
    monkeypatch.setattr(utils, "ACTIVE_WEIGHTS_DIR", tmp_active)

    # Patch weights module
    monkeypatch.setattr(weights, "DEFAULT_WEIGHTS_DIR", str(tmp_active))

    # Patch weight_merger path functions
    monkeypatch.setattr(wm, "get_weights_dir", lambda: tmp_weights)
    monkeypatch.setattr(wm, "get_active_dir", lambda: tmp_active)
    monkeypatch.setattr(wm, "get_runs_dir", lambda: tmp_runs)
    monkeypatch.setattr(wm, "get_merged_dir", lambda: tmp_merged)

    # Seed default files
    default_weights = {
        "Algo0": {"bias": 0.5, "w_modularity": 0.1, "w_log_nodes": 0.0}
    }
    (tmp_active / "type_0.json").write_text(json.dumps(default_weights))
    (tmp_active / "type_registry.json").write_text(json.dumps({
        "type_0": {"centroid": [0]*7, "graph_count": 1, "algorithms": ["Algo0"]}
    }))

    return tmp_weights, tmp_active, tmp_runs, tmp_merged


def test_path_constants(results: ResultsTracker):
    """Test that path constants are correct."""
    print("\n1. Testing Path Constants")
    print("-" * 40)
    
    # WEIGHTS_DIR should be scripts/weights
    results.check(
        WEIGHTS_DIR.name == "weights" and WEIGHTS_DIR.parent.name == "scripts",
        f"WEIGHTS_DIR points to scripts/weights: {WEIGHTS_DIR}"
    )
    
    # ACTIVE_WEIGHTS_DIR should be scripts/weights/active
    results.check(
        ACTIVE_WEIGHTS_DIR.name == "active" and ACTIVE_WEIGHTS_DIR.parent == WEIGHTS_DIR,
        f"ACTIVE_WEIGHTS_DIR points to scripts/weights/active: {ACTIVE_WEIGHTS_DIR}"
    )
    
    # DEFAULT_WEIGHTS_DIR should equal ACTIVE_WEIGHTS_DIR
    results.check(
        Path(DEFAULT_WEIGHTS_DIR) == ACTIVE_WEIGHTS_DIR,
        f"DEFAULT_WEIGHTS_DIR equals ACTIVE_WEIGHTS_DIR"
    )
    
    # weight_merger functions should return correct paths
    results.check(
        get_weights_dir() == WEIGHTS_DIR,
        f"get_weights_dir() returns WEIGHTS_DIR"
    )
    
    results.check(
        get_active_dir() == ACTIVE_WEIGHTS_DIR,
        f"get_active_dir() returns ACTIVE_WEIGHTS_DIR"
    )
    
    results.check(
        get_runs_dir() == WEIGHTS_DIR / "runs",
        f"get_runs_dir() returns scripts/weights/runs"
    )
    
    results.check(
        get_merged_dir() == WEIGHTS_DIR / "merged",
        f"get_merged_dir() returns scripts/weights/merged"
    )


def test_directory_structure(results: ResultsTracker):
    """Test that directory structure exists."""
    print("\n2. Testing Directory Structure")
    print("-" * 40)
    
    results.check(
        WEIGHTS_DIR.exists(),
        f"scripts/weights/ exists"
    )
    
    results.check(
        ACTIVE_WEIGHTS_DIR.exists(),
        f"scripts/weights/active/ exists"
    )
    
    # Check for type files in active
    type_files = list(ACTIVE_WEIGHTS_DIR.glob("type_*.json"))
    results.check(
        len(type_files) > 0,
        f"Found {len(type_files)} type files in active/"
    )
    
    registry_file = ACTIVE_WEIGHTS_DIR / "type_registry.json"
    results.check(
        registry_file.exists(),
        f"type_registry.json exists in active/"
    )


def test_weights_write_to_active(results: ResultsTracker):
    """Test that weights.py writes to active/ directory."""
    print("\n3. Testing Weights Write to Active")
    print("-" * 40)
    
    # Create a test type with unique name
    test_type = "type_test_flow"
    test_weights = {
        "TestAlgo": {
            "bias": 1.5,
            "w_modularity": 0.1,
            "w_log_nodes": 0.2,
        }
    }
    
    # Save using weights.py (should go to active/)
    save_type_weights(test_type, test_weights)
    
    # Verify it was written to active/
    expected_path = ACTIVE_WEIGHTS_DIR / f"{test_type}.json"
    results.check(
        expected_path.exists(),
        f"save_type_weights wrote to active/{test_type}.json"
    )
    
    # Load and verify content
    loaded = load_type_weights(test_type)
    results.check(
        loaded.get("TestAlgo", {}).get("bias") == 1.5,
        f"load_type_weights reads from active/"
    )
    
    # Clean up
    if expected_path.exists():
        expected_path.unlink()
        print(f"  [cleanup] Removed {test_type}.json")


def test_cpp_path_constants(results: ResultsTracker):
    """Test that C++ header has correct paths."""
    print("\n4. Testing C++ Path Constants")
    print("-" * 40)
    
    builder_h = PROJECT_ROOT / "bench" / "include" / "gapbs" / "builder.h"
    
    results.check(
        builder_h.exists(),
        f"builder.h exists"
    )
    
    if builder_h.exists():
        content = builder_h.read_text()
        
        # Check DEFAULT_WEIGHTS_FILE
        results.check(
            'DEFAULT_WEIGHTS_FILE = "scripts/weights/active/type_0.json"' in content,
            f"DEFAULT_WEIGHTS_FILE points to scripts/weights/active/"
        )
        
        # Check TYPE_WEIGHTS_DIR
        results.check(
            'TYPE_WEIGHTS_DIR = "scripts/weights/active/"' in content,
            f"TYPE_WEIGHTS_DIR points to scripts/weights/active/"
        )


def test_weight_merger_flow(results: ResultsTracker):
    """Test weight merger save/use flow."""
    print("\n5. Testing Weight Merger Flow")
    print("-" * 40)
    
    # Create test weights in active/
    test_type = "type_merger_test"
    test_weights = {
        "MergerTestAlgo": {
            "bias": 2.5,
            "w_modularity": 0.5,
        }
    }
    
    # Save to active/
    test_path = ACTIVE_WEIGHTS_DIR / f"{test_type}.json"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_path, 'w') as f:
        json.dump(test_weights, f)
    
    # Create a minimal registry
    registry_path = ACTIVE_WEIGHTS_DIR / "type_registry.json"
    original_registry = None
    if registry_path.exists():
        with open(registry_path) as f:
            original_registry = json.load(f)
    
    test_registry = {
        test_type: {
            "centroid": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "graph_count": 1,
            "algorithms": ["MergerTestAlgo"]
        }
    }
    with open(registry_path, 'w') as f:
        json.dump(test_registry, f)
    
    # Save current run
    run_path = save_current_run("test_flow_run")
    results.check(
        run_path.exists(),
        f"save_current_run created run folder"
    )
    
    saved_type_file = run_path / f"{test_type}.json"
    results.check(
        saved_type_file.exists(),
        f"Run folder contains {test_type}.json"
    )
    
    # List runs should include our test run
    runs = list_runs()
    test_run = next((r for r in runs if r.timestamp == "test_flow_run"), None)
    results.check(
        test_run is not None,
        f"list_runs() includes test_flow_run"
    )
    
    # Clean up
    if test_path.exists():
        test_path.unlink()
        print(f"  [cleanup] Removed {test_type}.json from active/")
    
    if run_path.exists():
        shutil.rmtree(run_path)
        print(f"  [cleanup] Removed test_flow_run from runs/")
    
    # Restore original registry
    if original_registry:
        with open(registry_path, 'w') as f:
            json.dump(original_registry, f, indent=2)
        print(f"  [cleanup] Restored original registry")


def test_existing_weights_valid(results: ResultsTracker):
    """Test that existing weights in active/ are valid."""
    print("\n6. Testing Existing Weights Validity")
    print("-" * 40)
    
    # Check type files
    type_files = sorted(ACTIVE_WEIGHTS_DIR.glob("type_[0-9]*.json"))
    results.check(
        len(type_files) > 0,
        f"Found {len(type_files)} type weight files"
    )
    
    for type_file in type_files:
        try:
            with open(type_file) as f:
                weights = json.load(f)
            
            # Check it has algorithms
            algos = [k for k in weights if not k.startswith('_')]
            valid = len(algos) > 0
            
            # Check first algo has required fields
            if algos:
                first_algo = weights[algos[0]]
                valid = valid and 'bias' in first_algo
                valid = valid and 'w_modularity' in first_algo
            
            results.check(
                valid,
                f"{type_file.name}: {len(algos)} algorithms, valid structure"
            )
        except Exception as e:
            results.check(False, f"{type_file.name}: Error - {e}")
    
    # Check registry
    registry_file = ACTIVE_WEIGHTS_DIR / "type_registry.json"
    if registry_file.exists():
        try:
            with open(registry_file) as f:
                registry = json.load(f)
            
            # Check it has type entries
            type_entries = [k for k in registry if k.startswith('type_')]
            results.check(
                len(type_entries) > 0,
                f"type_registry.json: {len(type_entries)} type entries"
            )
        except Exception as e:
            results.check(False, f"type_registry.json: Error - {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Weight Flow Test Suite")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Weights dir:  {WEIGHTS_DIR}")
    print(f"Active dir:   {ACTIVE_WEIGHTS_DIR}")
    
    results = TestResults()
    
    test_path_constants(results)
    test_directory_structure(results)
    test_weights_write_to_active(results)
    test_cpp_path_constants(results)
    test_weight_merger_flow(results)
    test_existing_weights_valid(results)
    
    success = results.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
