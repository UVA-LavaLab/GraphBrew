#!/usr/bin/env python3
"""Shared GraphBrew reorder-config parsing and validation helpers."""

from __future__ import annotations

import json
import math
import struct
from typing import Optional

GRAPHBREW_EFFECTIVE_CONFIG_PREFIX = "GraphBrew Effective Config: "
GRAPHBREW_REALIZED_CONFIG_PREFIX = "GraphBrew Realized Config: "

_GRAPHBREW_CONFIG_PREFIX = GRAPHBREW_EFFECTIVE_CONFIG_PREFIX
_GRAPHBREW_REALIZED_PREFIX = GRAPHBREW_REALIZED_CONFIG_PREFIX
_SIZE_T_MAX = (1 << (8 * struct.calcsize("P"))) - 1


def parse_graphbrew_effective_configs(output: Optional[str]) -> list[dict]:
    """Parse every structured GraphBrew config emitted by a reorder chain."""
    configs: list[dict] = []
    if not output:
        return configs
    for line in output.splitlines():
        if not line.startswith(GRAPHBREW_EFFECTIVE_CONFIG_PREFIX):
            continue
        payload = line[len(GRAPHBREW_EFFECTIVE_CONFIG_PREFIX):]
        config = json.loads(payload)
        if config.get("schema") != "graphbrew_config/v3":
            raise RuntimeError(
                f"Unsupported GraphBrew config schema: {config.get('schema')}"
            )
        configs.append(config)
    return configs


def parse_graphbrew_realized_configs(output: Optional[str]) -> list[dict]:
    """Parse every structured GraphBrew execution record in a reorder chain."""
    configs: list[dict] = []
    if not output:
        return configs
    for line in output.splitlines():
        if not line.startswith(GRAPHBREW_REALIZED_CONFIG_PREFIX):
            continue
        payload = line[len(GRAPHBREW_REALIZED_CONFIG_PREFIX):]
        config = json.loads(payload)
        if config.get("schema") not in {
            "graphbrew_realized/v1",
            "graphbrew_realized/v2",
        }:
            raise RuntimeError(
                "Unsupported GraphBrew realized-config schema: "
                f"{config.get('schema')}"
            )
        if config["schema"] == "graphbrew_realized/v1":
            config.setdefault("capacity_l2_bytes", 0)
            config.setdefault("capacity_llc_bytes", 0)
            config.setdefault(
                "capacity_property_bytes_per_vertex", 0)
            config.setdefault("capacity_l2_runs", 0)
            config.setdefault("capacity_llc_runs", 0)
            for key in (
                "gorder_communities",
                "gorder_vertices",
                "gorder_max_community",
                "gorder_fallback_communities",
                "gorder_fallback_vertices",
            ):
                config.setdefault(key, 0)
        configs.append(config)
    return configs


def extract_graphbrew_order_specs(algo_flags: list[str]) -> list[str]:
    return [
        algo_flags[index + 1]
        for index, flag in enumerate(algo_flags[:-1])
        if flag == "-o" and algo_flags[index + 1].split(":", 1)[0] == "12"
    ]


_graphbrew_specs = extract_graphbrew_order_specs


def expected_graphbrew_config(spec: str) -> dict[str, object]:
    tokens = spec.split(":")[1:]
    first = tokens[0] if tokens else "leiden"
    rabbit = first in {"rabbit", "rabbitorder"}
    preset = first in {"leiden", "rabbit", "hubcluster"}

    def is_numeric(token: str) -> bool:
        try:
            value = float(token)
        except ValueError:
            return False
        return math.isfinite(value)

    def is_integer(token: str) -> bool:
        try:
            value = int(token)
        except ValueError:
            return False
        return str(value) == token

    def is_ascii_digits(token: str) -> bool:
        return token.isascii() and token.isdigit()

    named_tokens = [first]
    positional: dict[int, str] = {}
    for index, token in enumerate(tokens[1:], start=1):
        if not preset:
            named_tokens.append(token)
            continue
        numeric = is_numeric(token)
        if (
            index in {1, 3, 4, 5}
            and numeric
            and not is_integer(token)
        ):
            raise RuntimeError(
                f"Malformed positional GraphBrew integer in {spec}: {token}"
            )
        is_positional = (
            (index == 1 and is_integer(token))
            or (
                index == 2
                and (
                    numeric
                    or token == "auto"
                    or token == "dynamic"
                    or token.startswith("dynamic_")
                )
            )
            or (index in {3, 4} and is_integer(token))
            or (
                index == 5
                and (
                    is_integer(token)
                    or token in {"auto", "adaptive"}
                )
            )
        )
        if is_positional:
            positional[index] = token
        else:
            named_tokens.append(token)

    expected: dict[str, object] = {
        "algorithm": "leiden",
        "community_mode": "full-leiden",
        "aggregation": "leiden-csr",
        "ordering": "layer",
        "super_graph": "none",
        "community_order": "size-desc",
        "intra_community_order": "bfs",
        "refinement_pass": "none",
        "resolution": "__positive_float__",
        "super_graph_resolution": 0.1,
        "max_iterations": 10,
        "max_passes": 10,
        "refinement_depth": -1,
        "m_computation": "half-edges",
        "deterministic_community_detection": True,
        "supergraph_move_batch": 1,
        "gorder_window": 5,
        "gorder_fallback": 0,
        "capacity_l2_bytes": 0,
        "capacity_llc_bytes": 0,
        "capacity_property_bytes_per_vertex": 0,
        "final_algo_id": 8,
        "recursive_depth": -1,
        "sub_algo_id": 8,
        "rabbit_degree_sort_preprocess": False,
        "use_refinement": True,
        "use_lazy_updates": False,
        "verify_topology": False,
        "dynamic_resolution": False,
        "degree_sorting": False,
        "community_merging": False,
        "hub_extraction": False,
        "hub_extraction_pct": 0.001,
        "gorder_intra": False,
        "hub_sort": False,
        "rcm_super": False,
        "rcm_intra": False,
        "small_community_merging": True,
        "has_explicit_ordering": False,
    }

    if first == "leiden":
        expected.update({
            "aggregation": "gve-csr",
            "m_computation": "total-edges",
            "refinement_depth": 0,
            "ordering": "layer",
            "has_explicit_ordering": True,
        })
    elif rabbit:
        expected.update({
            "algorithm": "rabbit",
            "ordering": "rabbit-native-dfs",
            "resolution": None,
            "final_algo_id": -1,
            "small_community_merging": False,
            "rabbit_degree_sort_preprocess": True,
        })
    elif first == "hubcluster":
        expected.update({
            "ordering": "hubcluster",
            "final_algo_id": -1,
            "small_community_merging": False,
            "has_explicit_ordering": True,
        })
    elif first in {"hrab", "tqr", "hcache", "hlr"}:
        expected.update({
            "ordering": first,
            "final_algo_id": -1,
            "small_community_merging": False,
            "has_explicit_ordering": True,
        })
        if first == "hrab":
            expected["rcm_intra"] = True
    elif first == "streaming":
        expected["aggregation"] = "streaming"

    ordering_tokens = {
        "compose": "compose",
        "pluggable": "compose",
        "dbg": "dbg",
        "corder": "corder",
        "hubcluster": "hubcluster",
        "hrab": "hrab",
        "tqr": "tqr",
        "hcache": "hcache",
        "hlr": "hlr",
        "dfs": "dendrogram-dfs",
        "bfs": "dendrogram-bfs",
        "community": "community-sort",
    }
    for token in named_tokens:
        if token in ordering_tokens:
            expected["ordering"] = ordering_tokens[token]
            expected["has_explicit_ordering"] = True
        elif token in {"graphbrew", "gb", "flat", "norecurse"}:
            expected["ordering"] = "layer"
            expected["small_community_merging"] = True
            expected["final_algo_id"] = (
                8 if expected["final_algo_id"] == -1
                else expected["final_algo_id"]
            )
            expected["has_explicit_ordering"] = True
            if token in {"flat", "norecurse"}:
                expected["recursive_depth"] = 0
        elif token.startswith("depth"):
            depth = token.removeprefix("depth").lstrip(":")
            if not depth.isdigit():
                raise RuntimeError(
                    f"Unparseable GraphBrew depth token in {spec}: {token}"
                )
            expected["ordering"] = "layer"
            expected["recursive_depth"] = int(depth)
            expected["small_community_merging"] = True
            expected["has_explicit_ordering"] = True
        elif token in {"bfs_intra", "bfsintra", "no_rcm_intra"}:
            expected["rcm_intra"] = False
        elif token == "rcm":
            expected["rcm_super"] = True
            expected["rcm_intra"] = True
        elif token in {"rcm_super", "rcmsuper"}:
            expected["rcm_super"] = True
        elif token in {"rcm_intra", "rcmintra"}:
            expected["rcm_intra"] = True
        elif token in {"hubx", "hub-extract", "hubextract"}:
            expected["hub_extraction"] = True
        elif token.startswith("hubx") and token[4:]:
            expected["hub_extraction"] = True
            expected["hub_extraction_pct"] = float(token[4:]) / 100.0
        elif token in {"gord", "gorder"}:
            expected["gorder_intra"] = True
        elif token.startswith("gord") and token[4:].isdigit():
            expected["gorder_intra"] = True
            expected["gorder_window"] = int(token[4:])
        elif token.startswith("gordf") and token[5:].isdigit():
            expected["gorder_intra"] = True
            expected["gorder_fallback"] = int(token[5:])
        elif token in {"hsort", "hubsort"}:
            expected["hub_sort"] = True
        elif token in {"merge", "coarsen"}:
            expected["community_merging"] = True
        elif token == "norefine":
            expected["use_refinement"] = False
        elif token in {"lazyupdate", "lazyupdates"}:
            expected["use_lazy_updates"] = True
        elif token == "verify":
            expected["verify_topology"] = True
        elif token == "dynamic":
            expected["dynamic_resolution"] = True
        elif token == "streaming":
            expected["aggregation"] = "streaming"
        elif token == "hybrid":
            expected["aggregation"] = "hybrid"
        elif token in {"gvecsr", "gve-csr"}:
            expected["aggregation"] = "gve-csr"
        elif token in {"totalm", "gvem"}:
            expected["m_computation"] = "total-edges"
        elif token == "halfm":
            expected["m_computation"] = "half-edges"
        elif token.startswith("refine") and token[6:].isdigit():
            expected["refinement_depth"] = int(token[6:])
        elif token in {
            "cd_parallel", "cd:parallel", "community_parallel",
        }:
            expected["deterministic_community_detection"] = False
        elif token in {
            "cd_serial", "cd:serial", "community_serial",
        }:
            expected["deterministic_community_detection"] = True
        elif token.startswith("sgmb") and token[4:].isdigit():
            expected["supergraph_move_batch"] = int(token[4:])
        elif (
            token.startswith("capl2k")
            and is_ascii_digits(token[6:])
        ):
            value = int(token[6:])
            if value <= 0 or value > _SIZE_T_MAX // 1024:
                raise RuntimeError(
                    f"Invalid capacity L2 token in {spec}: {token}")
            expected["capacity_l2_bytes"] = value * 1024
        elif token.startswith("capl2k"):
            raise RuntimeError(
                f"Malformed capacity L2 token in {spec}: {token}")
        elif (
            token.startswith("capllck")
            and is_ascii_digits(token[7:])
        ):
            value = int(token[7:])
            if value <= 0 or value > _SIZE_T_MAX // 1024:
                raise RuntimeError(
                    f"Invalid capacity LLC token in {spec}: {token}")
            expected["capacity_llc_bytes"] = value * 1024
        elif token.startswith("capllck"):
            raise RuntimeError(
                f"Malformed capacity LLC token in {spec}: {token}")
        elif (
            token.startswith("capv")
            and is_ascii_digits(token[4:])
        ):
            value = int(token[4:])
            if value <= 0 or value > _SIZE_T_MAX:
                raise RuntimeError(
                    f"Invalid capacity property token in {spec}: {token}")
            expected["capacity_property_bytes_per_vertex"] = value
        elif token.startswith("capv"):
            raise RuntimeError(
                f"Malformed capacity property token in {spec}: {token}")
        elif token.startswith("sgres") or token.startswith("gamma"):
            expected["super_graph_resolution"] = float(token[5:])
        elif token.startswith("gw") and token[2:].isdigit():
            expected["gorder_window"] = int(token[2:])
        elif is_numeric(token):
            value = float(token)
            if (
                0.0 < value <= 3.0
                and ("." in token or value < 1.0)
            ):
                expected["resolution"] = value
            elif 1.0 <= value <= 100.0:
                integer = int(value)
                if expected["max_iterations"] == 10:
                    expected["max_iterations"] = integer
                else:
                    expected["max_passes"] = integer
            elif 0.0 < value <= 3.0:
                expected["resolution"] = value

    super_graph_tokens = {
        "s1_none": "none",
        "s1none": "none",
        "sg_none": "none",
        "sgnone": "none",
        "s1_super_rabbit": "super-rabbit",
        "s1srabbit": "super-rabbit",
        "sg_super_rabbit": "super-rabbit",
        "sgsrabbit": "super-rabbit",
        "s1_super_rcm": "super-rcm",
        "s1srcm": "super-rcm",
        "sg_super_rcm": "super-rcm",
        "sgsrcm": "super-rcm",
        "s1_tile_rabbit": "tile-rabbit",
        "s1tilerabbit": "tile-rabbit",
        "sg_tile_rabbit": "tile-rabbit",
        "sgtilerabbit": "tile-rabbit",
        "s1_hilbert": "hilbert",
        "s1hilbert": "hilbert",
        "sg_hilbert": "hilbert",
        "sghilbert": "hilbert",
        "hilbert": "hilbert",
    }
    community_tokens = {
        "s2_identity": "identity",
        "s2identity": "identity",
        "comm_identity": "identity",
        "commidentity": "identity",
        "s2_size": "size-desc",
        "s2size": "size-desc",
        "comm_size": "size-desc",
        "commsize": "size-desc",
        "comm_size_desc": "size-desc",
        "commsizedesc": "size-desc",
        "s2_size_asc": "size-asc",
        "s2sizeasc": "size-asc",
        "comm_size_asc": "size-asc",
        "commsizeasc": "size-asc",
        "s2_degree": "degree-desc",
        "s2degree": "degree-desc",
        "comm_degree": "degree-desc",
        "commdegree": "degree-desc",
        "comm_degree_desc": "degree-desc",
        "commdegreedesc": "degree-desc",
        "s2_degree_asc": "degree-asc",
        "s2degreeasc": "degree-asc",
        "comm_degree_asc": "degree-asc",
        "commdegreeasc": "degree-asc",
        "s2_capacity": "capacity-runs",
        "comm_capacity_runs": "capacity-runs",
        "comm_cache_fit": "capacity-runs",
        "capacity_runs": "capacity-runs",
        "cache_fit": "capacity-runs",
        "s2_cut_min": "cut-min",
        "s2cutmin": "cut-min",
        "comm_cut_min": "cut-min",
        "commcutmin": "cut-min",
        "cut_min": "cut-min",
        "cutmin": "cut-min",
    }
    intra_tokens = {
        "s3_bfs": "bfs",
        "s3bfs": "bfs",
        "intra_bfs": "bfs",
        "intrabfs": "bfs",
        "s3_rcm": "rcm",
        "s3rcm": "rcm",
        "intra_rcm": "rcm",
        "intrarcm": "rcm",
        "s3_rcmpp": "rcmpp",
        "s3rcmpp": "rcmpp",
        "intra_rcmpp": "rcmpp",
        "intrarcmpp": "rcmpp",
        "rcmpp": "rcmpp",
        "rcm++": "rcmpp",
        "s3_dendrogram": "dendrogram",
        "s3dendrogram": "dendrogram",
        "intra_dendrogram": "dendrogram",
        "intradendrogram": "dendrogram",
        "s3_dend": "dendrogram",
        "intra_dend": "dendrogram",
        "s3_gorder": "gorder",
        "s3gorder": "gorder",
        "intra_gorder": "gorder",
        "intragorder": "gorder",
        "s3_gord": "gorder",
        "intra_gord": "gorder",
        "s3_gorder_faithful": "gorder-faithful",
        "intra_gorder_faithful": "gorder-faithful",
        "intra_gorder2": "gorder-faithful",
        "s3_hubsort": "hubsort",
        "s3hubsort": "hubsort",
        "intra_hubsort": "hubsort",
        "intrahubsort": "hubsort",
        "intra_hub": "hubsort",
        "s3_deg_asc": "degree-asc",
        "s3degasc": "degree-asc",
        "intra_deg_asc": "degree-asc",
        "intra_degasc": "degree-asc",
        "intra_degree_asc": "degree-asc",
        "deg_asc": "degree-asc",
        "degasc": "degree-asc",
        "s3_hub2": "hub2",
        "s3hub2": "hub2",
        "intra_hub2": "hub2",
        "intrahub2": "hub2",
        "hub2": "hub2",
        "s3_alternate": "alternate",
        "s3alt": "alternate",
        "intra_alternate": "alternate",
        "intra_alt": "alternate",
        "alternate": "alternate",
        "alt": "alternate",
        "s3_random": "random",
        "s3rand": "random",
        "intra_random": "random",
        "intra_rand": "random",
        "random": "random",
        "rand": "random",
        "s3_boundary_last": "boundary-last",
        "s3bndlast": "boundary-last",
        "intra_boundary_last": "boundary-last",
        "intra_bndlast": "boundary-last",
        "boundary_last": "boundary-last",
        "bndlast": "boundary-last",
        "boundarylast": "boundary-last",
        "s3_core": "core",
        "s3core": "core",
        "intra_core": "core",
        "intracore": "core",
        "core_order": "core",
        "coreorder": "core",
        "core": "core",
    }
    explicit_super_graph = any(
        token in super_graph_tokens for token in tokens
    )
    explicit_community_order = any(
        token in community_tokens for token in tokens
    )
    for token in tokens:
        if token in super_graph_tokens:
            expected["super_graph"] = super_graph_tokens[token]
        elif token in community_tokens:
            expected["community_order"] = community_tokens[token]
        elif token in intra_tokens:
            expected["intra_community_order"] = intra_tokens[token]
        elif token == "refine_2swap":
            expected["refinement_pass"] = "two-swap"

    if expected["ordering"] == "compose":
        if explicit_super_graph and not explicit_community_order:
            raise RuntimeError(
                "COMPOSE super-graph order requires an explicit "
                f"community-order token: {spec}"
            )

    if expected["community_order"] == "capacity-runs":
        if expected["ordering"] != "compose":
            raise RuntimeError(
                f"Capacity-run ordering requires COMPOSE: {spec}"
            )
        for key in (
            "capacity_l2_bytes",
            "capacity_llc_bytes",
            "capacity_property_bytes_per_vertex",
        ):
            if expected[key] == 0:
                raise RuntimeError(
                    "Capacity-run ordering requires pinned capl2k, "
                    f"capllck, and capv tokens: {spec}"
                )
        if expected["super_graph"] != "none":
            raise RuntimeError(
                f"Capacity-run ordering requires sg_none: {spec}"
            )
    elif any(
        expected[key] > 0
        for key in (
            "capacity_l2_bytes",
            "capacity_llc_bytes",
            "capacity_property_bytes_per_vertex",
        )
    ):
        raise RuntimeError(
            f"Capacity geometry requires capacity-run ordering: {spec}"
        )
    if (
        expected["intra_community_order"] == "gorder-faithful"
        and expected["ordering"] != "compose"
    ):
        raise RuntimeError(
            f"Faithful local Gorder requires COMPOSE: {spec}"
        )
    if (
        isinstance(expected["capacity_l2_bytes"], int)
        and isinstance(expected["capacity_llc_bytes"], int)
        and expected["capacity_l2_bytes"] > 0
        and expected["capacity_llc_bytes"]
            < expected["capacity_l2_bytes"]
    ):
        raise RuntimeError(
            f"Capacity LLC is smaller than L2 in {spec}")

    if 1 in positional:
        final_algo = int(positional[1])
        if not 0 <= final_algo <= 11:
            raise RuntimeError(
                f"Invalid GraphBrew final algorithm in {spec}: {final_algo}"
            )
        expected["final_algo_id"] = final_algo
    if 2 in positional:
        resolution = positional[2]
        if resolution.startswith("dynamic"):
            expected["dynamic_resolution"] = True
            if resolution.startswith("dynamic_"):
                expected["resolution"] = float(resolution[8:])
        elif resolution not in {"auto", "0"}:
            expected["resolution"] = float(resolution)
    if 3 in positional:
        expected["max_passes"] = int(positional[3])
    if 4 in positional:
        expected["recursive_depth"] = int(positional[4])
    if 5 in positional:
        sub_algo = positional[5]
        expected["sub_algo_id"] = (
            -1 if sub_algo in {"auto", "adaptive"}
            else int(sub_algo)
        )

    return expected


_expected_graphbrew_config = expected_graphbrew_config


def validate_graphbrew_effective_configs(
    algo_flags: list[str],
    configs: list[dict],
) -> None:
    specs = extract_graphbrew_order_specs(algo_flags)
    if len(configs) != len(specs):
        raise RuntimeError(
            "GraphBrew effective-config count mismatch: "
            f"expected {len(specs)}, got {len(configs)}"
        )
    for spec, actual in zip(specs, configs):
        expected = expected_graphbrew_config(spec)
        mismatches = {}
        for key, value in expected.items():
            actual_value = actual.get(key)
            if value == "__positive_float__":
                if (
                    not isinstance(actual_value, (int, float))
                    or actual_value <= 0
                ):
                    mismatches[key] = (actual_value, "positive float")
            elif value == "__positive_int__":
                if (
                    not isinstance(actual_value, int)
                    or actual_value <= 0
                ):
                    mismatches[key] = (actual_value, "positive integer")
            elif actual_value != value:
                mismatches[key] = (actual_value, value)
        unexpected = set(actual) - {"schema"} - set(expected)
        if unexpected:
            mismatches["unexpected_fields"] = (
                sorted(unexpected), "none",
            )
        if mismatches:
            raise RuntimeError(
                f"GraphBrew effective config mismatch for {spec}: {mismatches}"
            )


def validate_graphbrew_realized_configs(
    algo_flags: list[str],
    effective_configs: list[dict],
    realized_configs: list[dict],
) -> None:
    """Fail closed when runtime behavior differs from the requested config."""
    specs = extract_graphbrew_order_specs(algo_flags)
    if len(effective_configs) != len(specs):
        raise RuntimeError(
            "GraphBrew effective-config count mismatch before realized "
            f"validation: expected {len(specs)}, got {len(effective_configs)}"
        )
    if len(realized_configs) != len(specs):
        raise RuntimeError(
            "GraphBrew realized-config count mismatch: "
            f"expected {len(specs)}, got {len(realized_configs)}"
        )

    allowed_fallbacks = {
        (
            "ordering",
            "hcache",
            "hierarchical",
            "hcache-requires-two-passes",
        ),
        (
            "ordering",
            "hlr",
            "hrab",
            "hierarchical-rabbit-requires-two-passes",
        ),
        (
            "community_order",
            "cut-min",
            "degree-desc",
            "cut-min-community-limit",
        ),
        (
            "intra_community_order",
            "dendrogram",
            "bfs",
            "rabbit-dendrogram-unavailable",
        ),
    }
    required_fields = {
        "schema",
        "algorithm",
        "aggregation",
        "ordering",
        "super_graph",
        "community_order",
        "intra_community_order",
        "refinement_pass",
        "resolution",
        "recursive_depth",
        "schedule_sensitive",
        "gorder_window",
        "gorder_fallback",
        "gorder_communities",
        "gorder_vertices",
        "gorder_max_community",
        "gorder_fallback_communities",
        "gorder_fallback_vertices",
        "capacity_l2_bytes",
        "capacity_llc_bytes",
        "capacity_property_bytes_per_vertex",
        "capacity_l2_runs",
        "capacity_llc_runs",
        "final_algo_id",
        "sub_algo_id",
        "num_passes",
        "num_communities",
        "fallbacks",
        "block_algorithms",
    }

    for spec, effective, realized in zip(
        specs, effective_configs, realized_configs,
    ):
        if set(realized) != required_fields:
            raise RuntimeError(
                f"GraphBrew realized config fields mismatch for {spec}: "
                f"got {sorted(realized)}, expected {sorted(required_fields)}"
            )
        expected_aggregation = (
            "rabbit-incremental"
            if effective["algorithm"] == "rabbit"
            else effective["aggregation"]
        )
        expected_schedule_sensitive = (
            effective["algorithm"] == "rabbit"
            or not effective["deterministic_community_detection"]
            or effective["ordering"] in {"hrab", "hlr", "tqr"}
            or (
                effective["ordering"] == "layer"
                and effective["final_algo_id"] == 8
            )
            or effective["super_graph"] in {
                "super-rabbit", "tile-rabbit",
            }
        )
        exact = {
            "algorithm": effective["algorithm"],
            "aggregation": expected_aggregation,
            "super_graph": effective["super_graph"],
            "refinement_pass": effective["refinement_pass"],
            "resolution": effective["resolution"],
            "schedule_sensitive": expected_schedule_sensitive,
            "gorder_window": effective["gorder_window"],
            "gorder_fallback": effective["gorder_fallback"],
            "final_algo_id": effective["final_algo_id"],
            "sub_algo_id": effective["sub_algo_id"],
        }
        if effective["community_order"] != "capacity-runs":
            exact.update({
                "capacity_l2_bytes":
                    effective["capacity_l2_bytes"],
                "capacity_llc_bytes":
                    effective["capacity_llc_bytes"],
                "capacity_property_bytes_per_vertex":
                    effective["capacity_property_bytes_per_vertex"],
                "capacity_l2_runs": 0,
                "capacity_llc_runs": 0,
            })
        if effective["intra_community_order"] not in {
            "gorder", "gorder-faithful",
        }:
            exact.update({
                "gorder_communities": 0,
                "gorder_vertices": 0,
                "gorder_max_community": 0,
                "gorder_fallback_communities": 0,
                "gorder_fallback_vertices": 0,
            })
        mismatches = {
            key: (realized.get(key), value)
            for key, value in exact.items()
            if realized.get(key) != value
        }

        fallback_records = realized["fallbacks"]
        if not isinstance(fallback_records, list):
            mismatches["fallbacks"] = (
                type(fallback_records).__name__, "list",
            )
            fallback_records = []
        fallback_axes: set[tuple[str, str, str, str]] = set()
        for fallback in fallback_records:
            if not isinstance(fallback, dict) or set(fallback) != {
                "reason", "requested", "realized",
            }:
                mismatches["fallback_record"] = (
                    fallback, "reason/requested/realized object",
                )
                continue
            matches = {
                item
                for item in allowed_fallbacks
                if item[1:] == (
                    fallback["requested"],
                    fallback["realized"],
                    fallback["reason"],
                )
            }
            if len(matches) != 1:
                mismatches["unsupported_fallback"] = (
                    fallback, "allow-listed fallback",
                )
                continue
            fallback_axes.update(matches)

        for axis in (
            "ordering",
            "community_order",
            "intra_community_order",
        ):
            requested = effective[axis]
            actual = realized[axis]
            if actual == requested:
                if any(item[0] == axis for item in fallback_axes):
                    mismatches[f"unused_{axis}_fallback"] = (
                        actual, "runtime axis change",
                    )
                continue
            if (
                axis,
                requested,
                actual,
                next(
                    (
                        item[3] for item in fallback_axes
                        if item[:3] == (axis, requested, actual)
                    ),
                    "",
                ),
            ) not in allowed_fallbacks:
                mismatches[axis] = (actual, requested)

        requested_depth = effective["recursive_depth"]
        actual_depth = realized["recursive_depth"]
        if effective["ordering"] == "layer":
            if (
                not isinstance(actual_depth, int)
                or actual_depth < 0
                or (requested_depth >= 0 and actual_depth != requested_depth)
            ):
                mismatches["recursive_depth"] = (
                    actual_depth,
                    requested_depth if requested_depth >= 0 else "resolved >= 0",
                )
        elif actual_depth is not None:
            mismatches["recursive_depth"] = (actual_depth, None)

        if effective["community_order"] == "capacity-runs":
            for key in (
                "capacity_l2_bytes",
                "capacity_llc_bytes",
                "capacity_property_bytes_per_vertex",
            ):
                if (
                    not isinstance(realized[key], int)
                    or realized[key] < 1
                ):
                    mismatches[key] = (
                        realized[key], "positive integer",
                    )
            for key in ("capacity_l2_runs", "capacity_llc_runs"):
                if (
                    not isinstance(realized[key], int)
                    or realized[key] < 0
                ):
                    mismatches[key] = (
                        realized[key], "nonnegative integer",
                    )
            for key in (
                "capacity_l2_bytes",
                "capacity_llc_bytes",
                "capacity_property_bytes_per_vertex",
            ):
                requested = effective[key]
                if requested > 0 and realized[key] != requested:
                    mismatches[key] = (realized[key], requested)
            if (
                realized["capacity_llc_bytes"]
                < realized["capacity_l2_bytes"]
            ):
                mismatches["capacity_llc_bytes"] = (
                    realized["capacity_llc_bytes"],
                    ">= capacity_l2_bytes",
                )
            if (
                realized["capacity_l2_runs"]
                < realized["capacity_llc_runs"]
            ):
                mismatches["capacity_l2_runs"] = (
                    realized["capacity_l2_runs"],
                    ">= capacity_llc_runs",
                )

        if effective["intra_community_order"] in {
            "gorder", "gorder-faithful",
        }:
            for key in (
                "gorder_communities",
                "gorder_vertices",
                "gorder_max_community",
                "gorder_fallback_communities",
                "gorder_fallback_vertices",
            ):
                if (
                    not isinstance(realized[key], int)
                    or realized[key] < 0
                ):
                    mismatches[key] = (
                        realized[key], "nonnegative integer",
                    )
            if (
                realized["gorder_communities"] == 0
                and realized["gorder_max_community"] != 0
            ):
                mismatches["gorder_max_community"] = (
                    realized["gorder_max_community"],
                    0,
                )
            if (
                realized["gorder_fallback_communities"] == 0
                and realized["gorder_fallback_vertices"] != 0
            ):
                mismatches["gorder_fallback_vertices"] = (
                    realized["gorder_fallback_vertices"],
                    0,
                )
            if (
                realized["gorder_communities"]
                + realized["gorder_fallback_communities"]
                > realized["num_communities"]
            ):
                mismatches["gorder_communities"] = (
                    realized["gorder_communities"],
                    "<= num_communities with fallbacks",
                )

        if (
            not isinstance(realized["num_passes"], int)
            or realized["num_passes"] < 1
        ):
            mismatches["num_passes"] = (
                realized["num_passes"], "positive integer",
            )
        if (
            not isinstance(realized["num_communities"], int)
            or realized["num_communities"] < 1
        ):
            mismatches["num_communities"] = (
                realized["num_communities"], "positive integer",
            )
        block_algorithms = realized["block_algorithms"]
        if (
            not isinstance(block_algorithms, dict)
            or any(
                not isinstance(name, str)
                or not isinstance(count, int)
                or count < 1
                for name, count in (
                    block_algorithms.items()
                    if isinstance(block_algorithms, dict) else []
                )
            )
        ):
            mismatches["block_algorithms"] = (
                block_algorithms, "string-to-positive-integer object",
            )
        if effective["ordering"] == "layer" and not block_algorithms:
            mismatches["block_algorithms"] = (
                block_algorithms, "non-empty for layer execution",
            )

        if mismatches:
            raise RuntimeError(
                f"GraphBrew realized config mismatch for {spec}: {mismatches}"
            )
