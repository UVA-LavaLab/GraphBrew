"""Guard cross-language reordering identities."""

from pathlib import Path
import re

from scripts.lib.core.utils import ALGORITHMS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REORDER_TYPES = (
    PROJECT_ROOT / "bench/include/graphbrew/reorder/reorder_types.h"
)
REORDER_ADAPTIVE = (
    PROJECT_ROOT / "bench/include/graphbrew/reorder/reorder_adaptive.h"
)
REORDER_HEADER = PROJECT_ROOT / "bench/include/graphbrew/reorder/reorder.h"
ADAPTIVE_EMULATOR = PROJECT_ROOT / "scripts/lib/ml/adaptive_emulator.py"

def test_algorithm_ids_match_cpp_name_registry():
    source = REORDER_TYPES.read_text()
    id_block = source.split(
        "inline ReorderingAlgo getReorderingAlgo(int value)", 1
    )[1].split(
        "inline ReorderingAlgo getReorderingAlgo(const char* arg)", 1
    )[0]
    cpp_ids = {
        int(number): symbol
        for number, symbol in re.findall(
            r"case\s+(\d+):\s+return\s+(\w+);",
            id_block,
        )
    }

    name_block = source.split(
        "inline const std::map<std::string, ReorderingAlgo>& "
        "getAlgorithmNameMap()", 1
    )[1].split("return name_to_algo;", 1)[0]
    cpp_names = {
        name: symbol
        for name, symbol in re.findall(
            r'\{"([A-Z0-9_]+)",\s+(\w+)\}',
            name_block,
        )
    }

    assert sorted(cpp_ids) == list(range(17))
    for algorithm_id, name in ALGORITHMS.items():
        assert cpp_names[name.upper()] == cpp_ids[algorithm_id]


def test_retired_recursive_adaptive_path_has_no_don_lite_surface():
    assert not (
        PROJECT_ROOT
        / "bench/include/graphbrew/reorder/reorder_don_lite.h"
    ).exists()
    assert "reorder_don_lite.h" not in REORDER_HEADER.read_text()
    assert "DON_LITE" not in REORDER_ADAPTIVE.read_text()
    assert "DON_LITE" not in ADAPTIVE_EMULATOR.read_text()
    assert "Recursive AdaptiveOrder is retired" in (
        REORDER_ADAPTIVE.read_text()
    )
