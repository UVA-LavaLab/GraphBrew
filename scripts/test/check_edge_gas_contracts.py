#!/usr/bin/env python3
"""Validate canonical edge-centric and GAS algorithm contracts."""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "bench" / "contracts" / "edge_gas_algorithms.json"
MAKEFILE = ROOT / "Makefile"

CANONICAL = {
    "bfs", "bc", "cc", "cc_sv", "pr", "pr_spmv", "sssp", "tc",
}
GAS = {"cc", "pr", "sssp"}
DENSE_EDGE = {"cc", "cc_sv", "pr", "pr_spmv"}
FRONTIER_EDGE = {"bfs", "sssp"}
IRREGULAR_EDGE = {"bc", "tc"}
IMPLEMENTED_EDGE = DENSE_EDGE | FRONTIER_EDGE | IRREGULAR_EDGE
SPECIALIZED = {"bfs_p", "tc_p"}
GRAPH_DIRECTIONS = {
    "directed", "incoming_pull", "weakly_connected", "undirected_only",
}
WEIGHTING = {"unweighted", "edge_weighted"}
GRAPH_KINDS = {
    "directed_unweighted",
    "undirected_unweighted",
    "directed_weighted",
    "undirected_weighted",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def make_words(variable: str, text: str) -> set[str]:
    match = re.search(
        rf"^{re.escape(variable)}\s*=\s*(.+)$",
        text,
        flags=re.MULTILINE,
    )
    require(match is not None, f"Makefile lacks {variable}")
    return set(match.group(1).split())


def main() -> int:
    contract = json.loads(CONTRACT.read_text())
    require(
        contract.get("schema") == "graphbrew.edge-gas-algorithms",
        "edge/GAS contract schema differs",
    )
    require(
        contract.get("canonical_source_root") == "bench/src"
        and contract.get("edge_source_root") == "bench/src_edge"
        and contract.get("gas_source_root") == "bench/src_gas"
        and contract.get("simulator_source_root") == "bench/src_sim",
        "algorithm source roots differ",
    )
    canonical_root = ROOT / contract["canonical_source_root"]
    edge_root = ROOT / contract["edge_source_root"]
    gas_root = ROOT / contract["gas_source_root"]
    require(
        canonical_root.is_dir() and edge_root.is_dir() and gas_root.is_dir(),
        "algorithm source root missing",
    )
    require(
        not list(canonical_root.glob("*_edge.cc"))
        and not list(canonical_root.glob("*_gas.cc")),
        "canonical source root contains edge/GAS variants",
    )
    algorithms = contract.get("algorithms")
    require(isinstance(algorithms, list), "algorithms must be a list")
    by_name = {item.get("name"): item for item in algorithms}
    require(
        len(by_name) == len(algorithms),
        "algorithm names are missing or duplicated",
    )
    require(set(by_name) == CANONICAL, "canonical algorithm set differs")
    require(
        set(contract.get("edge_variants", [])) == CANONICAL,
        "edge variant set differs",
    )
    require(
        set(contract.get("implemented_edge_variants", [])) ==
        IMPLEMENTED_EDGE,
        "implemented edge variant set differs",
    )
    require(
        set(contract.get("gas_variants", [])) == GAS,
        "GAS variant set differs",
    )
    require(
        set(contract.get("implemented_gas_variants", [])) == GAS,
        "implemented GAS variant set differs",
    )
    require(
        contract.get("simulator_policy") ==
        "legacy_instrumented_forks_pending_shared_kernel_hooks",
        "simulator policy differs",
    )
    performance = contract.get("performance_contract")
    require(
        isinstance(performance, dict)
        and set(performance) == CANONICAL,
        "performance contract algorithm set differs",
    )
    for name, item in performance.items():
        for field in {"edge_work", "edge_balance", "ownership"}:
            require(
                isinstance(item.get(field), str) and item[field],
                f"{name}: performance {field} missing",
            )
        require(
            (isinstance(item.get("gas_work"), str) and item["gas_work"])
            if name in GAS
            else item.get("gas_work") is None,
            f"{name}: GAS work contract differs",
        )

    profiles = contract.get("test_profiles")
    require(isinstance(profiles, dict) and profiles, "test profiles missing")
    for name, profile in profiles.items():
        require(isinstance(profile, dict), f"{name}: profile must be an object")
        require(
            profile.get("graph_kind") in GRAPH_KINDS,
            f"{name}: invalid graph_kind",
        )
        path = profile.get("path")
        scale = profile.get("generator_scale")
        require(
            (path is None) != (scale is None),
            f"{name}: choose exactly one of path or generator_scale",
        )
        if path is not None:
            require(
                (ROOT / path).is_file(),
                f"{name}: missing test graph {path}",
            )
        else:
            require(
                type(scale) is int and scale > 0,
                f"{name}: generator_scale must be positive",
            )
        require(
            type(profile.get("symmetrize", False)) is bool,
            f"{name}: symmetrize must be boolean",
        )
    for name, item in sorted(by_name.items()):
        source = ROOT / item["source"]
        simulator = ROOT / item["simulator_source"]
        require(source.is_file(), f"{name}: missing canonical source")
        require(simulator.is_file(), f"{name}: missing simulator source")
        require(
            item.get("simulator_status") == "not_contract_authority",
            f"{name}: simulator status differs",
        )
        edge_binary = item.get("edge_binary")
        edge_source = item.get("edge_source")
        gas_binary = item.get("gas_binary")
        gas_source = item.get("gas_source")
        if name in IMPLEMENTED_EDGE:
            require(
                edge_binary == f"{name}_edge",
                f"{name}: edge binary differs",
            )
            require(
                isinstance(edge_source, str)
                and (ROOT / edge_source).is_file(),
                f"{name}: edge source missing",
            )
            require(
                (ROOT / edge_source).parent == edge_root,
                f"{name}: edge source is outside edge_source_root",
            )
        else:
            require(
                edge_binary is None and edge_source is None,
                f"{name}: unexpected implemented edge variant",
            )
        if name in GAS:
            require(
                gas_binary == f"{name}_gas",
                f"{name}: GAS binary differs",
            )
            require(
                isinstance(gas_source, str)
                and (ROOT / gas_source).is_file(),
                f"{name}: GAS source missing",
            )
            require(
                (ROOT / gas_source).parent == gas_root,
                f"{name}: GAS source is outside gas_source_root",
            )
            require(
                isinstance(item.get("gas_convergence"), str)
                and item["gas_convergence"],
                f"{name}: GAS convergence missing",
            )
            require(
                isinstance(item.get("gas_schedule"), str)
                and item["gas_schedule"],
                f"{name}: GAS schedule missing",
            )
        else:
            require(
                gas_binary is None and gas_source is None,
                f"{name}: unexpected implemented GAS variant",
            )
            require(
                item.get("gas_convergence") is None
                and item.get("gas_schedule") is None,
                f"{name}: unexpected GAS execution contract",
            )
        require(
            item.get("direction") in GRAPH_DIRECTIONS,
            f"{name}: invalid direction contract",
        )
        require(
            item.get("weighting") in WEIGHTING,
            f"{name}: invalid weighting contract",
        )
        require(
            isinstance(item.get("output"), str) and item["output"],
            f"{name}: output contract missing",
        )
        require(
            isinstance(item.get("equivalence"), str) and item["equivalence"],
            f"{name}: equivalence contract missing",
        )
        require(
            isinstance(item.get("convergence"), str) and item["convergence"],
            f"{name}: convergence contract missing",
        )
        require(
            isinstance(item.get("edge_schedule"), str)
            and item["edge_schedule"],
            f"{name}: edge schedule missing",
        )
        require(
            isinstance(item.get("gas"), str) and item["gas"],
            f"{name}: GAS classification missing",
        )
        requested_profiles = item.get("profiles")
        require(
            isinstance(requested_profiles, list) and requested_profiles,
            f"{name}: test profile list missing",
        )
        require(
            set(requested_profiles) <= set(profiles),
            f"{name}: unknown test profile",
        )
        require(
            item.get("smoke_profile") in requested_profiles,
            f"{name}: smoke_profile must be a declared profile",
        )
        profile_args = item.get("profile_args", {})
        require(
            isinstance(profile_args, dict)
            and set(profile_args) <= set(requested_profiles),
            f"{name}: profile_args differ from profiles",
        )
        require(
            all(
                isinstance(args, list)
                and all(isinstance(arg, str) and arg for arg in args)
                for args in profile_args.values()
            ),
            f"{name}: profile_args must contain string lists",
        )
        if item["weighting"] == "edge_weighted":
            require(
                all(
                    profiles[profile]["graph_kind"] in {
                        "directed_weighted",
                        "undirected_weighted",
                    }
                    for profile in requested_profiles
                ),
                f"{name}: weighted algorithm uses unweighted profile",
            )

        text = source.read_text()
        graph_type = item.get("graph_type")
        require(
            graph_type in {"Graph", "WGraph"},
            f"{name}: invalid graph_type",
        )
        verifier = item["verifier"]
        require(
            re.search(rf"\b{re.escape(verifier)}\s*\(", text) is not None,
            f"{name}: verifier {verifier} not found",
        )
        label_position = text.find(f'"{name}"')
        benchmark_position = text.rfind(
            "BenchmarkKernel", 0, label_position)
        require(
            label_position >= 0 and benchmark_position >= 0,
            f"{name}: BenchmarkKernel label differs",
        )
        benchmark_binding = text[benchmark_position:label_position]
        verifier_binding = item.get("verifier_binding")
        require(
            verifier_binding in {"bound", "direct"},
            f"{name}: invalid verifier_binding",
        )
        if verifier_binding == "bound":
            bound_position = text.rfind(
                "auto VerifierBound", 0, benchmark_position)
            require(
                bound_position >= 0,
                f"{name}: VerifierBound definition missing",
            )
            bound_body = text[bound_position:benchmark_position]
            require(
                re.search(
                    rf"\breturn\s+{re.escape(verifier)}\s*\(",
                    bound_body,
                ) is not None,
                f"{name}: VerifierBound does not call {verifier}",
            )
            require(
                "VerifierBound" in benchmark_binding,
                f"{name}: BenchmarkKernel does not use VerifierBound",
            )
        else:
            require(
                re.search(
                    rf"\b{re.escape(verifier)}\b",
                    benchmark_binding,
                ) is not None,
                f"{name}: BenchmarkKernel does not use {verifier}",
            )
        if item["source_picker_pair"]:
            picker_pattern = re.compile(
                rf"SourcePicker<{re.escape(graph_type)}>\s+(sp|vsp)"
                r"\(g,\s*cli\.start_vertex\(\),"
                r"\s*cli\.num_trials\(\)\);"
            )
            pickers = picker_pattern.findall(text)
            require(
                set(pickers) == {"sp", "vsp"},
                f"{name}: kernel/verifier SourcePicker pair missing",
            )
            source_pair_profile = item.get("source_pair_profile")
            trials = item.get("source_pair_trials")
            require(
                source_pair_profile in requested_profiles,
                f"{name}: source_pair_profile missing",
            )
            require(
                type(trials) is int and trials >= 2,
                f"{name}: source_pair_trials must be at least two",
            )
            pair_args = item.get("source_pair_args", [])
            require(
                isinstance(pair_args, list)
                and all(isinstance(arg, str) and arg for arg in pair_args),
                f"{name}: invalid source_pair_args",
            )

    specialized = contract.get("specialized_consumers")
    require(isinstance(specialized, list), "specialized consumers missing")
    specialized_by_name = {item.get("name"): item for item in specialized}
    require(
        set(specialized_by_name) == SPECIALIZED,
        "specialized consumer set differs",
    )
    for name, item in specialized_by_name.items():
        require(
            (ROOT / item["source"]).is_file(),
            f"{name}: specialized source missing",
        )
        require(
            item.get("canonical_semantics") in CANONICAL,
            f"{name}: canonical semantics missing",
        )

    makefile = MAKEFILE.read_text()
    require(
        CANONICAL | SPECIALIZED <= make_words("KERNELS", makefile),
        "Makefile KERNELS omits canonical or specialized algorithms",
    )
    require(
        {f"{name}_edge" for name in DENSE_EDGE} ==
        make_words("DENSE_EDGE_KERNELS", makefile),
        "Makefile dense edge kernels differ",
    )
    require(
        {f"{name}_edge" for name in FRONTIER_EDGE} ==
        make_words("FRONTIER_EDGE_KERNELS", makefile),
        "Makefile frontier edge kernels differ",
    )
    require(
        {f"{name}_edge" for name in IRREGULAR_EDGE} ==
        make_words("IRREGULAR_EDGE_KERNELS", makefile),
        "Makefile irregular edge kernels differ",
    )
    require(
        {f"{name}_gas" for name in GAS} ==
        make_words("GAS_KERNELS", makefile),
        "Makefile GAS kernels differ",
    )
    require(
        CANONICAL == make_words("KERNELS_SIM", makefile),
        "Makefile KERNELS_SIM must match canonical algorithms",
    )
    natural_gas = {
        name for name, item in by_name.items()
        if item["gas"].startswith("natural")
    }
    require(
        natural_gas == GAS,
        "per-algorithm GAS classifications differ from gas_variants",
    )

    print(
        "edge-gas-contract-check: PASS "
        "(8 planned edge algorithms; 8 implemented edge algorithms; "
        "3 natural GAS algorithms; "
        "2 specialized consumers)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
