#!/usr/bin/env python3
"""
Test Weight Flow - Verify weights are generated and read from correct locations.

This test verifies:
1. Python writes weights to results/models/perceptron/
2. C++ reads from results/models/perceptron/
3. Weight merger saves runs to results/models/perceptron/runs/
4. Weight merger merges to results/models/perceptron/merged/
5. Use-run and use-merged copy to weights/

Usage:
    pytest scripts/test/test_weight_flow.py -v
    python -m scripts.test.test_weight_flow
"""

import pytest
import sys
import json
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.utils import (
    WEIGHTS_DIR, ACTIVE_WEIGHTS_DIR,
    VARIANT_PREFIXES, VARIANT_ALGO_IDS, DISPLAY_TO_CANONICAL,
    ALGORITHMS, RABBITORDER_VARIANTS, RCM_VARIANTS, GRAPHBREW_VARIANTS,
    CHAIN_SEPARATOR, CHAINED_ORDERINGS, _CHAINED_ORDERING_OPTS,
    is_chained_ordering_name,
    get_all_algorithm_variant_names, resolve_canonical_name, is_variant_prefixed,
    canonical_algo_key, algo_converter_opt, get_algo_variants,
    canonical_name_from_converter_opt, chain_canonical_name,
    get_algorithm_name,
    LEGACY_ALGO_NAME_MAP,
)
from scripts.lib.weights import (
    DEFAULT_WEIGHTS_DIR,
    save_type_weights,
    load_type_weights,
)
from scripts.lib.weight_merger import (
    get_weights_dir,
    get_active_dir,
    get_runs_dir,
    get_merged_dir,
    save_current_run,
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

    tmp_weights = tmp_path / "models" / "perceptron"
    tmp_active = tmp_weights
    tmp_runs = tmp_weights / "runs"
    tmp_merged = tmp_weights / "merged"
    for d in [tmp_active, tmp_runs, tmp_merged]:
        d.mkdir(parents=True, exist_ok=True)

    # Patch utils
    monkeypatch.setattr(utils, "WEIGHTS_DIR", tmp_weights)
    monkeypatch.setattr(utils, "ACTIVE_WEIGHTS_DIR", tmp_weights)

    # Patch weights module
    monkeypatch.setattr(weights, "DEFAULT_WEIGHTS_DIR", str(tmp_weights))

    # Patch weight_merger path functions
    monkeypatch.setattr(wm, "get_weights_dir", lambda: tmp_weights)
    monkeypatch.setattr(wm, "get_active_dir", lambda: tmp_active)
    monkeypatch.setattr(wm, "get_runs_dir", lambda: tmp_runs)
    monkeypatch.setattr(wm, "get_merged_dir", lambda: tmp_merged)

    # Seed default files
    default_weights = {
        "Algo0": {"bias": 0.5, "w_modularity": 0.1, "w_log_nodes": 0.0}
    }
    (tmp_weights / "type_0").mkdir(parents=True, exist_ok=True)
    (tmp_weights / "type_0" / "weights.json").write_text(json.dumps(default_weights))
    (tmp_weights / "registry.json").write_text(json.dumps({
        "type_0": {"centroid": [0]*7, "graph_count": 1, "algorithms": ["Algo0"]}
    }))

    return tmp_weights, tmp_weights, tmp_runs, tmp_merged


def test_path_constants(results: ResultsTracker):
    """Test that path constants are correct."""
    print("\n1. Testing Path Constants")
    print("-" * 40)
    
    # WEIGHTS_DIR should be results/models/perceptron
    results.check(
        WEIGHTS_DIR.name == "perceptron" and WEIGHTS_DIR.parent.name == "models",
        f"WEIGHTS_DIR points to results/models/perceptron: {WEIGHTS_DIR}"
    )
    
    # ACTIVE_WEIGHTS_DIR should equal WEIGHTS_DIR (no more /active/ sublevel)
    results.check(
        ACTIVE_WEIGHTS_DIR == WEIGHTS_DIR,
        f"ACTIVE_WEIGHTS_DIR equals WEIGHTS_DIR: {ACTIVE_WEIGHTS_DIR}"
    )
    
    # DEFAULT_WEIGHTS_DIR should equal ACTIVE_WEIGHTS_DIR
    results.check(
        Path(DEFAULT_WEIGHTS_DIR) == ACTIVE_WEIGHTS_DIR,
        "DEFAULT_WEIGHTS_DIR equals ACTIVE_WEIGHTS_DIR"
    )
    
    # weight_merger functions should return correct paths
    results.check(
        get_weights_dir() == WEIGHTS_DIR,
        "get_weights_dir() returns WEIGHTS_DIR"
    )
    
    results.check(
        get_active_dir() == ACTIVE_WEIGHTS_DIR,
        "get_active_dir() returns ACTIVE_WEIGHTS_DIR"
    )
    
    results.check(
        get_runs_dir() == WEIGHTS_DIR / "runs",
        "get_runs_dir() returns results/models/perceptron/runs"
    )
    
    results.check(
        get_merged_dir() == WEIGHTS_DIR / "merged",
        "get_merged_dir() returns results/models/perceptron/merged"
    )


def test_directory_structure(results: ResultsTracker):
    """Test that directory structure exists."""
    print("\n2. Testing Directory Structure")
    print("-" * 40)
    
    results.check(
        WEIGHTS_DIR.exists(),
        "results/models/perceptron/ exists"
    )
    
    results.check(
        ACTIVE_WEIGHTS_DIR.exists(),
        "results/models/perceptron/ (active) exists"
    )
    
    # Check for type directories in active
    type_dirs = [d for d in ACTIVE_WEIGHTS_DIR.iterdir() if d.is_dir() and d.name.startswith('type_')]
    results.check(
        len(type_dirs) > 0,
        f"Found {len(type_dirs)} type directories in active/"
    )
    
    registry_file = ACTIVE_WEIGHTS_DIR / "registry.json"
    results.check(
        registry_file.exists(),
        "registry.json exists in active/"
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
    expected_path = ACTIVE_WEIGHTS_DIR / test_type / "weights.json"
    results.check(
        expected_path.exists(),
        f"save_type_weights wrote to active/{test_type}/weights.json"
    )
    
    # Load and verify content
    loaded = load_type_weights(test_type)
    results.check(
        loaded.get("TestAlgo", {}).get("bias") == 1.5,
        "load_type_weights reads from active/"
    )
    
    # Clean up
    if expected_path.exists():
        shutil.rmtree(expected_path.parent)
        print(f"  [cleanup] Removed {test_type}/")


def test_cpp_path_constants(results: ResultsTracker):
    """Test that C++ header has correct paths."""
    print("\n4. Testing C++ Path Constants")
    print("-" * 40)
    
    reorder_h = PROJECT_ROOT / "bench" / "include" / "graphbrew" / "reorder" / "reorder_types.h"
    
    results.check(
        reorder_h.exists(),
        "reorder_types.h exists"
    )
    
    if reorder_h.exists():
        content = reorder_h.read_text()
        
        # Check TYPE_WEIGHTS_DIR
        results.check(
            'TYPE_WEIGHTS_DIR = "results/models/perceptron/"' in content,
            "TYPE_WEIGHTS_DIR points to results/models/perceptron/"
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
    test_dir = ACTIVE_WEIGHTS_DIR / test_type
    test_dir.mkdir(parents=True, exist_ok=True)
    test_path = test_dir / "weights.json"
    with open(test_path, 'w') as f:
        json.dump(test_weights, f)
    
    # Create a minimal registry
    registry_path = ACTIVE_WEIGHTS_DIR / "registry.json"
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
        "save_current_run created run folder"
    )
    
    saved_type_file = run_path / test_type / "weights.json"
    results.check(
        saved_type_file.exists(),
        f"Run folder contains {test_type}/weights.json"
    )
    
    # List runs should include our test run
    runs = list_runs()
    test_run = next((r for r in runs if r.timestamp == "test_flow_run"), None)
    results.check(
        test_run is not None,
        "list_runs() includes test_flow_run"
    )
    
    # Clean up
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"  [cleanup] Removed {test_type}/ from active/")
    
    if run_path.exists():
        shutil.rmtree(run_path)
        print("  [cleanup] Removed test_flow_run from runs/")
    
    # Restore original registry
    if original_registry:
        with open(registry_path, 'w') as f:
            json.dump(original_registry, f, indent=2)
        print("  [cleanup] Restored original registry")


def test_existing_weights_valid(results: ResultsTracker):
    """Test that existing weights in active/ are valid."""
    print("\n6. Testing Existing Weights Validity")
    print("-" * 40)
    
    # Check type directories
    type_dirs = sorted(d for d in ACTIVE_WEIGHTS_DIR.iterdir() if d.is_dir() and d.name.startswith('type_'))
    results.check(
        len(type_dirs) > 0,
        f"Found {len(type_dirs)} type directories"
    )
    
    for type_dir in type_dirs:
        type_file = type_dir / "weights.json"
        if not type_file.exists():
            continue
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
                f"{type_dir.name}/weights.json: {len(algos)} algorithms, valid structure"
            )
        except Exception as e:
            results.check(False, f"{type_dir.name}/weights.json: Error - {e}")
    
    # Check registry
    registry_file = ACTIVE_WEIGHTS_DIR / "registry.json"
    if registry_file.exists():
        try:
            with open(registry_file) as f:
                registry = json.load(f)
            
            # Check it has type entries
            type_entries = [k for k in registry if k.startswith('type_')]
            results.check(
                len(type_entries) > 0,
                f"registry.json: {len(type_entries)} type entries"
            )
        except Exception as e:
            results.check(False, f"registry.json: Error - {e}")


# =========================================================================
# SSOT Variant Registry Tests
# =========================================================================

class TestVariantRegistrySSOT:
    """Verify the SSOT variant registry is consistent across all layers."""

    def test_variant_prefixes_are_tuples(self):
        """Variant lists should be tuples (immutable SSOT)."""
        assert isinstance(VARIANT_PREFIXES, tuple)
        assert isinstance(RABBITORDER_VARIANTS, tuple)
        assert isinstance(RCM_VARIANTS, tuple)
        assert isinstance(GRAPHBREW_VARIANTS, tuple)

    def test_variant_prefixes_match_registry(self):
        """VARIANT_PREFIXES should match _VARIANT_ALGO_REGISTRY prefixes."""
        from scripts.lib.utils import _VARIANT_ALGO_REGISTRY
        registry_prefixes = {pfx for pfx, _, _ in _VARIANT_ALGO_REGISTRY.values()}
        assert set(VARIANT_PREFIXES) == registry_prefixes

    def test_variant_algo_ids_complete(self):
        """Every algo with variants should be in VARIANT_ALGO_IDS."""
        assert 8 in VARIANT_ALGO_IDS   # RabbitOrder
        assert 11 in VARIANT_ALGO_IDS  # RCM
        assert 12 in VARIANT_ALGO_IDS  # GraphBrewOrder

    def test_get_all_variant_names_no_duplicates(self):
        """get_all_algorithm_variant_names() should return unique names."""
        names = get_all_algorithm_variant_names()
        assert len(names) == len(set(names)), f"Duplicates: {[n for n in names if names.count(n) > 1]}"

    def test_get_all_variant_names_includes_expected(self):
        """All canonical variant names should appear."""
        names = set(get_all_algorithm_variant_names())
        assert "RABBITORDER_csr" in names
        assert "RABBITORDER_boost" in names
        assert "RCM_default" in names
        assert "RCM_bnf" in names
        assert "GraphBrewOrder_leiden" in names
        assert "GraphBrewOrder_rabbit" in names
        assert "GraphBrewOrder_hubcluster" in names
        assert "LeidenOrder" in names

    def test_get_all_variant_names_excludes_meta(self):
        """MAP, AdaptiveOrder, ORIGINAL, RANDOM should not be in trainable variant names."""
        names = set(get_all_algorithm_variant_names())
        assert "MAP" not in names
        assert "AdaptiveOrder" not in names
        assert "ORIGINAL" not in names
        assert "RANDOM" not in names

    def test_resolve_canonical_base_names(self):
        """Base algorithm names resolve correctly."""
        assert resolve_canonical_name("ORIGINAL") == "ORIGINAL"
        assert resolve_canonical_name("Random") == "RANDOM"
        assert resolve_canonical_name("HubSort") == "HUBSORT"

    def test_resolve_canonical_variant_passthrough(self):
        """Variant-prefixed names pass through unchanged."""
        assert resolve_canonical_name("GraphBrewOrder_leiden") == "GraphBrewOrder_leiden"
        assert resolve_canonical_name("RABBITORDER_csr") == "RABBITORDER_csr"
        assert resolve_canonical_name("RCM_bnf") == "RCM_bnf"
        # New compound variants also pass through
        assert resolve_canonical_name("GraphBrewOrder_leiden_dfs") == "GraphBrewOrder_leiden_dfs"

    def test_resolve_canonical_default_variants(self):
        """Bare base names resolve to default variant."""
        assert resolve_canonical_name("RabbitOrder") == "RABBITORDER_csr"
        assert resolve_canonical_name("GraphBrewOrder") == "GraphBrewOrder_leiden"

    def test_is_variant_prefixed(self):
        """is_variant_prefixed correctly identifies variant names."""
        assert is_variant_prefixed("GraphBrewOrder_leiden")
        assert is_variant_prefixed("RABBITORDER_csr")
        assert is_variant_prefixed("RCM_bnf")
        assert not is_variant_prefixed("ORIGINAL")
        assert not is_variant_prefixed("DBG")
        assert not is_variant_prefixed("LeidenOrder")

    def test_display_to_canonical_complete(self):
        """DISPLAY_TO_CANONICAL should cover all C++ display names."""
        # Every non-variant algorithm should have an entry
        for algo_id, name in ALGORITHMS.items():
            if algo_id in VARIANT_ALGO_IDS or name in ("MAP", "AdaptiveOrder"):
                continue
            assert name in DISPLAY_TO_CANONICAL or name.upper() in DISPLAY_TO_CANONICAL, \
                f"Missing DISPLAY_TO_CANONICAL entry for {name}"

    # ---- Multi-layer tests ----

    def test_graphbrew_layers_structure(self):
        """GRAPHBREW_LAYERS should have all 6 layers."""
        from scripts.lib.utils import GRAPHBREW_LAYERS
        assert "preset" in GRAPHBREW_LAYERS
        assert "ordering" in GRAPHBREW_LAYERS
        assert "aggregation" in GRAPHBREW_LAYERS
        assert "features" in GRAPHBREW_LAYERS
        assert "graphbrew_dispatch" in GRAPHBREW_LAYERS
        assert "numeric" in GRAPHBREW_LAYERS

    def test_ordering_excludes_streaming(self):
        """'streaming' is an aggregation strategy, NOT an ordering strategy."""
        from scripts.lib.utils import GRAPHBREW_LAYERS
        assert "streaming" not in GRAPHBREW_LAYERS["ordering"]
        assert "streaming" in GRAPHBREW_LAYERS["aggregation"]

    def test_graphbrew_options_backward_compat(self):
        """GRAPHBREW_OPTIONS backward-compat alias matches GRAPHBREW_LAYERS."""
        from scripts.lib.utils import GRAPHBREW_OPTIONS, GRAPHBREW_LAYERS
        assert set(GRAPHBREW_OPTIONS["presets"].keys()) == set(GRAPHBREW_LAYERS["preset"].keys())
        assert GRAPHBREW_OPTIONS["ordering_strategies"] == list(GRAPHBREW_LAYERS["ordering"])
        assert GRAPHBREW_OPTIONS["aggregation"] == list(GRAPHBREW_LAYERS["aggregation"])
        assert GRAPHBREW_OPTIONS["features"] == list(GRAPHBREW_LAYERS["features"])

    def test_enumerate_multilayer_counts(self):
        """enumerate_graphbrew_multilayer() returns correct counts."""
        from scripts.lib.utils import enumerate_graphbrew_multilayer
        info = enumerate_graphbrew_multilayer()
        assert info["layers"]["presets"] == 3
        assert info["layers"]["orderings"] == 13
        assert info["layers"]["aggregations"] == 4
        assert info["layers"]["features"] == 11
        assert info["layers"]["feature_combos"] == 2048  # 2^11
        # 3 base presets + 3×13 compounds = 42
        assert len(info["compound_variants"]) == 3 + 3 * 13
        # Active trained ⊆ compound_variants
        for a in info["active_trained"]:
            assert a in info["compound_variants"]

    def test_enumerate_multilayer_no_overlap(self):
        """Active and untrained should be disjoint and cover all compounds."""
        from scripts.lib.utils import enumerate_graphbrew_multilayer
        info = enumerate_graphbrew_multilayer()
        active_set = set(info["active_trained"])
        untrained_set = set(info["untrained"])
        assert active_set & untrained_set == set(), "Overlap between active and untrained"
        assert active_set | untrained_set == set(info["compound_variants"])

    def test_get_algo_variants_variant_algos(self):
        """get_algo_variants() returns variant tuples for registered algorithms."""
        assert get_algo_variants(8) == RABBITORDER_VARIANTS
        assert get_algo_variants(11) == RCM_VARIANTS
        assert get_algo_variants(12) == GRAPHBREW_VARIANTS

    def test_get_algo_variants_non_variant(self):
        """get_algo_variants() returns None for non-variant algorithms."""
        assert get_algo_variants(0) is None
        assert get_algo_variants(2) is None
        assert get_algo_variants(9) is None   # GOrder: intentionally not in registry
        assert get_algo_variants(15) is None  # LeidenOrder


class TestChainedOrderingExclusion:
    """Chained orderings are included in weight training alongside single orderings."""

    def test_chain_separator_constant(self):
        """CHAIN_SEPARATOR should be '+'."""
        assert CHAIN_SEPARATOR == "+"

    def test_is_chained_ordering_name(self):
        """is_chained_ordering_name() should detect '+' in names."""
        assert is_chained_ordering_name("SORT+RABBITORDER_csr")
        assert is_chained_ordering_name("DBG+GraphBrewOrder_leiden")
        assert not is_chained_ordering_name("RABBITORDER_csr")
        assert not is_chained_ordering_name("ORIGINAL")
        assert not is_chained_ordering_name("GraphBrewOrder_leiden")

    def test_chained_orderings_all_have_separator(self):
        """Every CHAINED_ORDERINGS entry must contain CHAIN_SEPARATOR."""
        for canonical, _opts in CHAINED_ORDERINGS:
            assert CHAIN_SEPARATOR in canonical, f"{canonical} missing separator"

    def test_variant_names_include_chains(self):
        """get_all_algorithm_variant_names() must include chained names."""
        names = get_all_algorithm_variant_names()
        chained_names = [n for n in names if is_chained_ordering_name(n)]
        assert len(chained_names) == len(CHAINED_ORDERINGS), (
            f"Expected {len(CHAINED_ORDERINGS)} chained names in variant list, "
            f"got {len(chained_names)}: {chained_names}"
        )

    def test_chain_names_auto_generated(self):
        """CHAINED_ORDERINGS names must be auto-derived from _CHAINED_ORDERING_OPTS."""
        assert len(CHAINED_ORDERINGS) == len(_CHAINED_ORDERING_OPTS)
        for (canonical, opts), raw_opts in zip(CHAINED_ORDERINGS, _CHAINED_ORDERING_OPTS):
            assert opts == raw_opts, f"opts mismatch: {opts!r} != {raw_opts!r}"
            expected_name = chain_canonical_name(raw_opts)
            assert canonical == expected_name, (
                f"Auto name mismatch: {canonical!r} != {expected_name!r}"
            )


class TestCanonicalNameFromOpt:
    """canonical_name_from_converter_opt() derives names from -o arguments."""

    def test_simple_algorithms(self):
        """Non-variant algorithms return base name."""
        assert canonical_name_from_converter_opt("0") == "ORIGINAL"
        assert canonical_name_from_converter_opt("2") == "SORT"
        assert canonical_name_from_converter_opt("5") == "DBG"
        assert canonical_name_from_converter_opt("9") == "GORDER"

    def test_variant_default(self):
        """Variant algorithms without explicit variant use default."""
        assert canonical_name_from_converter_opt("8") == "RABBITORDER_csr"
        assert canonical_name_from_converter_opt("11") == "RCM_default"
        assert canonical_name_from_converter_opt("12") == "GraphBrewOrder_leiden"

    def test_variant_explicit(self):
        """Explicit variant in option string."""
        assert canonical_name_from_converter_opt("8:boost") == "RABBITORDER_boost"
        assert canonical_name_from_converter_opt("11:bnf") == "RCM_bnf"
        assert canonical_name_from_converter_opt("12:rabbit") == "GraphBrewOrder_rabbit"

    def test_multilayer_graphbrew(self):
        """Multi-layer GraphBrewOrder configs normalize colons to underscores."""
        assert canonical_name_from_converter_opt("12:leiden:hrab") == "GraphBrewOrder_leiden_hrab"
        assert canonical_name_from_converter_opt("12:leiden:hrab:gvecsr") == "GraphBrewOrder_leiden_hrab_gvecsr"
        assert canonical_name_from_converter_opt("12:leiden:hrab:gvecsr:merge:hubx") == (
            "GraphBrewOrder_leiden_hrab_gvecsr_merge_hubx"
        )

    def test_flat_token_stripped_from_canonical_name(self):
        """Runtime-only tokens (flat, norecurse, recursive) are stripped."""
        assert canonical_name_from_converter_opt("12:leiden:flat") == "GraphBrewOrder_leiden"
        assert canonical_name_from_converter_opt("12:leiden:hrab:flat") == "GraphBrewOrder_leiden_hrab"
        assert canonical_name_from_converter_opt("12:leiden:norecurse") == "GraphBrewOrder_leiden"
        assert canonical_name_from_converter_opt("12:leiden:recursive") == "GraphBrewOrder_leiden"

    def test_agrees_with_canonical_algo_key(self):
        """canonical_name_from_converter_opt() must agree with canonical_algo_key()."""
        for algo_id in ALGORITHMS:
            name_from_opt = canonical_name_from_converter_opt(str(algo_id))
            name_from_key = canonical_algo_key(algo_id)
            assert name_from_opt == name_from_key, (
                f"Disagreement for algo {algo_id}: opt={name_from_opt!r} key={name_from_key!r}"
            )

    def test_agrees_with_get_algorithm_name(self):
        """canonical_name_from_converter_opt and get_algorithm_name must agree."""
        test_opts = ["0", "2", "8", "8:boost", "12", "12:leiden", "12:rabbit", "11:bnf"]
        for opt in test_opts:
            from_opt = canonical_name_from_converter_opt(opt)
            from_name = get_algorithm_name(opt)
            assert from_opt == from_name, (
                f"Disagreement for {opt!r}: from_opt={from_opt!r} from_name={from_name!r}"
            )


class TestChainCanonicalName:
    """chain_canonical_name() auto-derives chain names from converter opts."""

    def test_basic_chains(self):
        assert chain_canonical_name("-o 2 -o 8:csr") == "SORT+RABBITORDER_csr"
        assert chain_canonical_name("-o 2 -o 8:boost") == "SORT+RABBITORDER_boost"
        assert chain_canonical_name("-o 7 -o 8:csr") == "HUBCLUSTERDBG+RABBITORDER_csr"
        assert chain_canonical_name("-o 2 -o 12:leiden") == "SORT+GraphBrewOrder_leiden"
        assert chain_canonical_name("-o 5 -o 12:leiden") == "DBG+GraphBrewOrder_leiden"

    def test_multilayer_in_chain(self):
        """Chains with multi-layer GraphBrewOrder configs."""
        assert chain_canonical_name("-o 2 -o 12:leiden:hrab") == "SORT+GraphBrewOrder_leiden_hrab"
        assert chain_canonical_name("-o 2 -o 12:leiden:hrab:gvecsr:merge") == (
            "SORT+GraphBrewOrder_leiden_hrab_gvecsr_merge"
        )

    def test_invalid_opts_raises(self):
        """Empty converter opts should raise ValueError."""
        import pytest
        with pytest.raises(ValueError):
            chain_canonical_name("no dash o here")


class TestMultiLayerVariantPropagation:
    """Multi-layer variant names propagate correctly through all APIs."""

    def test_canonical_algo_key_multilayer(self):
        """canonical_algo_key with colon-separated multi-layer variant."""
        assert canonical_algo_key(12, "leiden:hrab") == "GraphBrewOrder_leiden_hrab"
        assert canonical_algo_key(12, "leiden:hrab:gvecsr") == "GraphBrewOrder_leiden_hrab_gvecsr"
        assert canonical_algo_key(12, "leiden:hrab:gvecsr:merge:hubx") == (
            "GraphBrewOrder_leiden_hrab_gvecsr_merge_hubx"
        )

    def test_algo_converter_opt_multilayer(self):
        """algo_converter_opt preserves colons for CLI."""
        assert algo_converter_opt(12, "leiden:hrab") == "12:leiden:hrab"
        assert algo_converter_opt(12, "leiden:hrab:gvecsr:merge") == "12:leiden:hrab:gvecsr:merge"

    def test_roundtrip_key_to_opt(self):
        """canonical_algo_key and canonical_name_from_converter_opt roundtrip."""
        variants = ["leiden", "leiden:hrab", "leiden:hrab:gvecsr:merge:hubx"]
        for v in variants:
            key = canonical_algo_key(12, v)
            opt = algo_converter_opt(12, v)
            roundtrip_key = canonical_name_from_converter_opt(opt)
            assert roundtrip_key == key, (
                f"Roundtrip mismatch for variant {v!r}: key={key!r} roundtrip={roundtrip_key!r}"
            )


class TestCanonicalAlgoKey:
    """canonical_algo_key() is the single entry-point for algorithm naming."""

    def test_non_variant_algorithms(self):
        """Non-variant algorithms return the bare SSOT name."""
        assert canonical_algo_key(0) == "ORIGINAL"
        assert canonical_algo_key(1) == "RANDOM"
        assert canonical_algo_key(2) == "SORT"
        assert canonical_algo_key(5) == "DBG"
        assert canonical_algo_key(9) == "GORDER"
        assert canonical_algo_key(10) == "CORDER"
        assert canonical_algo_key(15) == "LeidenOrder"

    def test_variant_algorithms_default(self):
        """Variant algorithms always include default variant suffix."""
        assert canonical_algo_key(8) == "RABBITORDER_csr"
        assert canonical_algo_key(11) == "RCM_default"
        assert canonical_algo_key(12) == "GraphBrewOrder_leiden"

    def test_variant_algorithms_explicit(self):
        """Explicit variant is used when provided."""
        assert canonical_algo_key(8, "boost") == "RABBITORDER_boost"
        assert canonical_algo_key(11, "bnf") == "RCM_bnf"
        assert canonical_algo_key(12, "rabbit") == "GraphBrewOrder_rabbit"
        assert canonical_algo_key(12, "hubcluster") == "GraphBrewOrder_hubcluster"

    def test_matches_get_all_variant_names(self):
        """Every name from get_all_algorithm_variant_names() should be
        producible by canonical_algo_key()."""
        all_names = set(get_all_algorithm_variant_names())
        for algo_id, algo_name in ALGORITHMS.items():
            if algo_name in ("ORIGINAL", "RANDOM", "MAP", "AdaptiveOrder"):
                continue
            key = canonical_algo_key(algo_id)
            assert key in all_names, f"canonical_algo_key({algo_id}) = {key!r} not in SSOT list"

    def test_matches_get_algorithm_name_with_variant(self):
        """canonical_algo_key() and get_algorithm_name_with_variant() must agree."""
        from scripts.lib.reorder import get_algorithm_name_with_variant
        for algo_id in ALGORITHMS:
            assert canonical_algo_key(algo_id) == get_algorithm_name_with_variant(algo_id)

    def test_algo_converter_opt_non_variant(self):
        """Non-variant algorithms produce bare ID strings."""
        assert algo_converter_opt(0) == "0"
        assert algo_converter_opt(2) == "2"
        assert algo_converter_opt(5) == "5"

    def test_algo_converter_opt_variant_default(self):
        """Variant algorithms include default variant in converter opt."""
        assert algo_converter_opt(8) == "8:csr"
        assert algo_converter_opt(11) == "11:default"
        assert algo_converter_opt(12) == "12:leiden:flat"

    def test_algo_converter_opt_explicit(self):
        """Explicit variant is used in converter opt."""
        assert algo_converter_opt(8, "boost") == "8:boost"
        assert algo_converter_opt(12, "leiden") == "12:leiden:flat"
        assert algo_converter_opt(12, "rabbit") == "12:rabbit"

    def test_key_and_opt_are_paired(self):
        """canonical_algo_key() and algo_converter_opt() should cover the
        same algorithm IDs consistently."""
        for algo_id in ALGORITHMS:
            key = canonical_algo_key(algo_id)
            opt = algo_converter_opt(algo_id)
            # The opt should start with the algo_id
            assert opt.startswith(str(algo_id)), f"opt={opt!r} doesn't start with {algo_id}"
            # The key should not be empty
            assert key, f"Empty key for algo_id={algo_id}"

    def test_legacy_map_is_centralized(self):
        """LEGACY_ALGO_NAME_MAP should contain 'GraphBrewOrder' mapping."""
        assert "GraphBrewOrder" in LEGACY_ALGO_NAME_MAP
        assert LEGACY_ALGO_NAME_MAP["GraphBrewOrder"] == "GraphBrewOrder_leiden"


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
