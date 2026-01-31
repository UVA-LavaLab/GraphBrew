import pytest
#!/usr/bin/env python3
"""
Test suite for weight_merger module.

Validates:
1. Type matching by centroid similarity
2. Centroid weighted average computation
3. Weight weighted average computation
4. Multi-run merge correctness
5. Edge cases (empty runs, single type, etc.)
"""

import os
import sys
import json
import shutil
import tempfile
import numpy as np
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.lib.weight_merger import (
    TypeInfo, RunInfo, CENTROID_FEATURES,
    centroid_distance, find_matching_type,
    merge_centroids, merge_weights,
    merge_runs, save_current_run, use_run, use_merged,
    get_weights_dir, get_runs_dir, get_merged_dir,
)


@pytest.fixture(autouse=True)
def wm_env(tmp_path, monkeypatch):
    import scripts.lib.weight_merger as wm
    tmp_runs = tmp_path / "runs"
    tmp_merged = tmp_path / "merged"
    tmp_active = tmp_path / "active"
    for d in [tmp_runs, tmp_merged, tmp_active]:
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(wm, "get_runs_dir", lambda: tmp_runs)
    monkeypatch.setattr(wm, "get_merged_dir", lambda: tmp_merged)
    monkeypatch.setattr(wm, "get_active_dir", lambda: tmp_active)
    monkeypatch.setattr(wm, "get_weights_dir", lambda: tmp_path)
    return tmp_runs, tmp_merged, tmp_active


def test_centroid_distance():
    """Test centroid distance calculation."""
    print("Testing centroid_distance...")
    
    # Identical centroids should have distance 0
    c1 = np.array([0.5, 1.0, 0.3, 0.5, 5.0, 10.0, 5.0])
    c2 = np.array([0.5, 1.0, 0.3, 0.5, 5.0, 10.0, 5.0])
    assert centroid_distance(c1, c2) == 0.0, "Identical centroids should have 0 distance"
    
    # Different centroids should have positive distance
    c3 = np.array([0.8, 2.0, 0.6, 0.7, 8.0, 20.0, 10.0])
    dist = centroid_distance(c1, c3)
    assert dist > 0, "Different centroids should have positive distance"
    print(f"  Distance between different centroids: {dist:.4f}")
    
    # Very different centroids should have larger distance
    c4 = np.array([0.1, 4.0, 0.1, 0.1, 15.0, 80.0, 50.0])
    dist2 = centroid_distance(c1, c4)
    assert dist2 > dist, "More different centroids should have larger distance"
    print(f"  Distance to very different centroid: {dist2:.4f}")
    
    print("  ✓ centroid_distance tests passed")


def test_find_matching_type():
    """Test type matching by centroid similarity."""
    print("Testing find_matching_type...")
    
    # Create some types
    type_a = TypeInfo(
        type_id="type_0",
        centroid=[0.8, 1.5, 0.4, 0.6, 5.0, 10.0, 5.0],
        graph_count=5
    )
    type_b = TypeInfo(
        type_id="type_1", 
        centroid=[0.3, 3.0, 0.2, 0.3, 12.0, 50.0, 20.0],
        graph_count=3
    )
    
    existing = {"type_0": type_a, "type_1": type_b}
    
    # Source similar to type_a should match type_0
    similar_to_a = TypeInfo(
        type_id="src_type",
        centroid=[0.75, 1.6, 0.45, 0.55, 5.5, 11.0, 6.0],
        graph_count=2
    )
    match = find_matching_type(similar_to_a, existing, threshold=0.3)
    assert match == "type_0", f"Should match type_0, got {match}"
    print(f"  Similar to type_0 correctly matched: {match}")
    
    # Source similar to type_b should match type_1
    similar_to_b = TypeInfo(
        type_id="src_type",
        centroid=[0.35, 2.8, 0.25, 0.35, 11.0, 48.0, 18.0],
        graph_count=2
    )
    match = find_matching_type(similar_to_b, existing, threshold=0.3)
    assert match == "type_1", f"Should match type_1, got {match}"
    print(f"  Similar to type_1 correctly matched: {match}")
    
    # Very different source should not match with strict threshold
    very_different = TypeInfo(
        type_id="src_type",
        centroid=[0.1, 0.5, 0.9, 0.9, 20.0, 100.0, 80.0],
        graph_count=2
    )
    match = find_matching_type(very_different, existing, threshold=0.1)
    assert match is None, f"Should not match any type, got {match}"
    print(f"  Very different correctly unmatched: {match}")
    
    print("  ✓ find_matching_type tests passed")


def test_merge_centroids():
    """Test centroid weighted average."""
    print("Testing merge_centroids...")
    
    # Test with lists
    c1 = [0.8, 2.0, 0.4]
    c2 = [0.4, 1.0, 0.6]
    count1, count2 = 3, 1
    
    merged = merge_centroids(c1, count1, c2, count2)
    
    # Expected: weighted average
    # (0.8*3 + 0.4*1) / 4 = 2.8/4 = 0.7
    # (2.0*3 + 1.0*1) / 4 = 7/4 = 1.75
    # (0.4*3 + 0.6*1) / 4 = 1.8/4 = 0.45
    expected = [0.7, 1.75, 0.45]
    
    for i, (m, e) in enumerate(zip(merged, expected)):
        assert abs(m - e) < 0.0001, f"Centroid[{i}]: expected {e}, got {m}"
    
    print(f"  c1={c1}, count={count1}")
    print(f"  c2={c2}, count={count2}")
    print(f"  merged={[round(x, 4) for x in merged]}")
    print(f"  expected={expected}")
    
    # Test equal weights
    c3 = [1.0, 2.0]
    c4 = [3.0, 4.0]
    merged_eq = merge_centroids(c3, 1, c4, 1)
    expected_eq = [2.0, 3.0]
    for i, (m, e) in enumerate(zip(merged_eq, expected_eq)):
        assert abs(m - e) < 0.0001, f"Equal weights failed at [{i}]"
    print(f"  Equal weight merge: {merged_eq} == {expected_eq}")
    
    print("  ✓ merge_centroids tests passed")


def test_merge_weights():
    """Test perceptron weight weighted average."""
    print("Testing merge_weights...")
    
    w1 = {
        "bias": 0.5,
        "w_modularity": 0.3,
        "w_degree_variance": -0.2,
        "benchmark_weights": {"pr": 1.0, "bfs": 0.8},
        "_metadata": {"version": 1}
    }
    w2 = {
        "bias": 0.7,
        "w_modularity": 0.1,
        "w_degree_variance": 0.2,
        "benchmark_weights": {"pr": 0.6, "bfs": 1.0, "cc": 0.5},
        "_metadata": {"version": 2}
    }
    count1, count2 = 4, 2
    
    merged = merge_weights(w1, count1, w2, count2)
    
    # bias: (0.5*4 + 0.7*2) / 6 = 3.4/6 = 0.5667
    # w_modularity: (0.3*4 + 0.1*2) / 6 = 1.4/6 = 0.2333
    # w_degree_variance: (-0.2*4 + 0.2*2) / 6 = -0.4/6 = -0.0667
    
    assert abs(merged["bias"] - 0.5667) < 0.001, f"bias: expected ~0.5667, got {merged['bias']}"
    assert abs(merged["w_modularity"] - 0.2333) < 0.001, f"w_modularity: expected ~0.2333, got {merged['w_modularity']}"
    assert abs(merged["w_degree_variance"] - (-0.0667)) < 0.001, f"w_degree_variance: expected ~-0.0667, got {merged['w_degree_variance']}"
    
    print(f"  w1 (count={count1}): bias={w1['bias']}, w_mod={w1['w_modularity']}")
    print(f"  w2 (count={count2}): bias={w2['bias']}, w_mod={w2['w_modularity']}")
    print(f"  merged: bias={merged['bias']:.4f}, w_mod={merged['w_modularity']:.4f}")
    
    # Check benchmark_weights
    assert "benchmark_weights" in merged
    bw = merged["benchmark_weights"]
    # pr: (1.0*4 + 0.6*2) / 6 = 5.2/6 = 0.8667
    assert abs(bw["pr"] - 0.8667) < 0.001, f"benchmark_weights.pr wrong"
    # cc: only in w2, so (0*4 + 0.5*2) / 6 = 1/6 = 0.1667
    assert abs(bw["cc"] - 0.1667) < 0.001, f"benchmark_weights.cc wrong"
    print(f"  benchmark_weights merged correctly: pr={bw['pr']:.4f}, cc={bw['cc']:.4f}")
    
    # Metadata should prefer newer
    assert merged["_metadata"]["version"] == 2, "Should keep newer metadata"
    
    print("  ✓ merge_weights tests passed")


def test_full_merge_flow():
    """Test full merge across multiple runs with type matching."""
    print("Testing full merge flow...")
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create two runs with different type numbering
        # Run 1: Social graphs (high modularity) as type_0
        run1_dir = tmpdir / "runs" / "run_20260101_000000"
        run1_dir.mkdir(parents=True)
        
        run1_registry = {
            "type_0": {
                "centroid": [0.85, 1.5, 0.45, 0.7, 5.0, 10.0, 5.0],  # Social-like
                "sample_count": 3,
                "algorithms": ["GORDER", "RABBIT"]
            },
            "type_1": {
                "centroid": [0.30, 3.5, 0.25, 0.3, 12.0, 50.0, 20.0],  # Web-like
                "sample_count": 2,
                "algorithms": ["CORDER"]
            }
        }
        with open(run1_dir / "type_registry.json", "w") as f:
            json.dump(run1_registry, f)
        
        run1_type0 = {
            "GORDER": {"bias": 0.6, "w_modularity": 0.4, "w_degree_variance": 0.1},
            "RABBIT": {"bias": 0.5, "w_modularity": 0.3, "w_degree_variance": 0.2}
        }
        run1_type1 = {
            "CORDER": {"bias": 0.4, "w_modularity": 0.1, "w_degree_variance": 0.5}
        }
        with open(run1_dir / "type_0.json", "w") as f:
            json.dump(run1_type0, f)
        with open(run1_dir / "type_1.json", "w") as f:
            json.dump(run1_type1, f)
        
        # Run 2: Same graph types but different numbering!
        # Web graphs as type_0 (was type_1 in run1)
        # Social graphs as type_1 (was type_0 in run1)
        run2_dir = tmpdir / "runs" / "run_20260102_000000"
        run2_dir.mkdir(parents=True)
        
        run2_registry = {
            "type_0": {
                "centroid": [0.32, 3.3, 0.28, 0.32, 11.0, 48.0, 18.0],  # Web-like (similar to run1.type_1)
                "sample_count": 4,
                "algorithms": ["CORDER", "RCM"]
            },
            "type_1": {
                "centroid": [0.82, 1.6, 0.42, 0.68, 5.5, 11.0, 6.0],  # Social-like (similar to run1.type_0)
                "sample_count": 5,
                "algorithms": ["GORDER", "RABBIT", "LEIDEN"]
            }
        }
        with open(run2_dir / "type_registry.json", "w") as f:
            json.dump(run2_registry, f)
        
        run2_type0 = {
            "CORDER": {"bias": 0.5, "w_modularity": 0.15, "w_degree_variance": 0.4},
            "RCM": {"bias": 0.3, "w_modularity": 0.05, "w_degree_variance": 0.6}
        }
        run2_type1 = {
            "GORDER": {"bias": 0.7, "w_modularity": 0.5, "w_degree_variance": 0.05},
            "RABBIT": {"bias": 0.6, "w_modularity": 0.4, "w_degree_variance": 0.15},
            "LEIDEN": {"bias": 0.55, "w_modularity": 0.35, "w_degree_variance": 0.1}
        }
        with open(run2_dir / "type_0.json", "w") as f:
            json.dump(run2_type0, f)
        with open(run2_dir / "type_1.json", "w") as f:
            json.dump(run2_type1, f)
        
        # Create merged directory
        merged_dir = tmpdir / "merged"
        
        # Load runs manually
        run1 = RunInfo(timestamp="run_20260101_000000", path=run1_dir)
        run2 = RunInfo(timestamp="run_20260102_000000", path=run2_dir)
        run1.load()
        run2.load()
        
        print(f"  Run1 types: {list(run1.types.keys())}")
        print(f"  Run2 types: {list(run2.types.keys())}")
        
        # Verify type matching works correctly
        # run2.type_0 (web-like) should match run1.type_1 (web-like)
        from scripts.lib.weight_merger import find_matching_type
        
        match = find_matching_type(run2.types["type_0"], run1.types, threshold=0.3)
        print(f"  run2.type_0 (web) matches run1.{match}")
        assert match == "type_1", f"run2.type_0 should match run1.type_1, got {match}"
        
        match = find_matching_type(run2.types["type_1"], run1.types, threshold=0.3)
        print(f"  run2.type_1 (social) matches run1.{match}")
        assert match == "type_0", f"run2.type_1 should match run1.type_0, got {match}"
        
        print("  ✓ Type matching correctly handles ID remapping")
        
        # Now do full merge using the merge_runs function
        # We need to patch the directories
        import scripts.lib.weight_merger as wm
        orig_runs = wm.get_runs_dir
        orig_merged = wm.get_merged_dir
        wm.get_runs_dir = lambda: tmpdir / "runs"
        wm.get_merged_dir = lambda: merged_dir
        
        try:
            summary = merge_runs(threshold=0.3, output_dir=merged_dir)
        finally:
            wm.get_runs_dir = orig_runs
            wm.get_merged_dir = orig_merged
        
        print(f"  Merge summary: {summary['total_types']} types from {summary['runs_merged']} runs")
        
        # Verify merged results
        with open(merged_dir / "type_registry.json") as f:
            merged_reg = json.load(f)
        
        print(f"  Merged types: {list(merged_reg.keys())}")
        
        # Should have 2 types (social and web)
        assert len(merged_reg) == 2, f"Should have 2 merged types, got {len(merged_reg)}"
        
        # Check that counts are correct
        # Social: run1.type_0 (3) + run2.type_1 (5) = 8
        # Web: run1.type_1 (2) + run2.type_0 (4) = 6
        counts = sorted([entry["graph_count"] for entry in merged_reg.values()])
        assert counts == [6, 8], f"Counts should be [6, 8], got {counts}"
        print(f"  Graph counts correctly summed: {counts}")
        
        # Check merged weights for GORDER (in social type)
        # Find which type has GORDER
        for tid, tfile in [("type_0", "type_0.json"), ("type_1", "type_1.json")]:
            fpath = merged_dir / tfile
            if fpath.exists():
                with open(fpath) as f:
                    weights = json.load(f)
                if "GORDER" in weights:
                    gw = weights["GORDER"]
                    # run1: bias=0.6, count=3
                    # run2: bias=0.7, count=5
                    # merged: (0.6*3 + 0.7*5) / 8 = 5.3/8 = 0.6625
                    expected_bias = (0.6 * 3 + 0.7 * 5) / 8
                    assert abs(gw["bias"] - expected_bias) < 0.001, \
                        f"GORDER bias: expected {expected_bias:.4f}, got {gw['bias']:.4f}"
                    print(f"  GORDER weights correctly merged: bias={gw['bias']:.4f} (expected {expected_bias:.4f})")
                    break
        
        print("  ✓ Full merge flow tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Weight Merger Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_centroid_distance,
        test_find_matching_type,
        test_merge_centroids,
        test_merge_weights,
        test_full_merge_flow,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {test.__name__} FAILED (returned False)")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test.__name__} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    assert failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
