#ifndef GRAPHBREW_ANALYSIS_ADAPTIVE_SOURCE_POLICY_H_
#define GRAPHBREW_ANALYSIS_ADAPTIVE_SOURCE_POLICY_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <parallel/algorithm>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../external/nlohmann_json.hpp"

namespace graphbrew {
namespace analysis {

#define GRAPHBREW_ADAPTIVE_SOURCE_POLICY(id, count, seed, reachability) \
    inline constexpr const char* ADAPTIVE_SOURCE_POLICY_ID = id; \
    inline constexpr size_t ADAPTIVE_SOURCE_COUNT = count; \
    inline constexpr uint64_t ADAPTIVE_SOURCE_SEED = seed; \
    inline constexpr double ADAPTIVE_SOURCE_MIN_REACHABILITY = reachability;
#include "graphbrew/reorder/adaptive_source_policy.def"
#undef GRAPHBREW_ADAPTIVE_SOURCE_POLICY

struct AdaptiveSourceRecord {
    int64_t original = -1;
    int64_t internal = -1;
    int64_t out_degree = 0;
    size_t source_index = 0;
    size_t requested_octile = 0;
    size_t realized_octile = 0;
    size_t rank = 0;
    size_t octile_start = 0;
    size_t octile_end = 0;
    size_t candidate_offset = 0;
    uint64_t reachable_vertices = 0;
    double reachable_fraction = 0.0;
    std::vector<std::string> replacement_path;
};

struct AdaptiveSourceSelection {
    uint64_t nodes = 0;
    uint64_t directed_edges = 0;
    int64_t largest_component_label = -1;
    uint64_t largest_component_size = 0;
    int64_t largest_component_min_original = -1;
    uint64_t second_largest_component_size = 0;
    uint64_t minimum_reachable_vertices = 0;
    std::vector<AdaptiveSourceRecord> sources;
};

inline uint64_t MixAdaptiveSourceSeed(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

inline std::vector<size_t> AdaptiveOctileFallbackOrder(
    size_t requested) {
    std::vector<size_t> order{requested};
    for (size_t distance = 1; distance < ADAPTIVE_SOURCE_COUNT; ++distance) {
        if (requested + distance < ADAPTIVE_SOURCE_COUNT)
            order.push_back(requested + distance);
        if (requested >= distance)
            order.push_back(requested - distance);
    }
    return order;
}

template <typename GraphT, typename ComponentRange>
AdaptiveSourceSelection SelectAdaptiveSources(
    const GraphT& graph,
    const ComponentRange& components) {
    using NodeID = decltype(graph.get_internal_id(0));
    const size_t nodes = static_cast<size_t>(graph.num_nodes());
    if (graph.directed())
        throw std::invalid_argument(
            "Adaptive source policy requires an undirected graph");
    if (components.size() != nodes)
        throw std::invalid_argument(
            "Adaptive source policy component vector has wrong size");
    if (nodes < ADAPTIVE_SOURCE_COUNT)
        throw std::runtime_error(
            "Adaptive source policy requires at least eight vertices");

    std::vector<uint32_t> component_sizes(nodes, 0);
    std::vector<int32_t> component_min_original(
        nodes, std::numeric_limits<int32_t>::max());
    for (size_t node = 0; node < nodes; ++node) {
        const int64_t label = static_cast<int64_t>(components[node]);
        if (label < 0 || static_cast<size_t>(label) >= nodes)
            throw std::runtime_error(
                "Adaptive source policy saw an invalid component label");
        ++component_sizes[static_cast<size_t>(label)];
        component_min_original[static_cast<size_t>(label)] = std::min(
            component_min_original[static_cast<size_t>(label)],
            static_cast<int32_t>(
                graph.get_org_id(static_cast<NodeID>(node))));
    }

    size_t largest_label = 0;
    uint32_t largest_size = 0;
    int32_t largest_min_original = std::numeric_limits<int32_t>::max();
    for (size_t label = 0; label < nodes; ++label) {
        if (
            component_sizes[label] > largest_size
            || (
                component_sizes[label] == largest_size
                && component_min_original[label] < largest_min_original
            )
        ) {
            largest_size = component_sizes[label];
            largest_label = label;
            largest_min_original = component_min_original[label];
        }
    }
    uint32_t second_largest_size = 0;
    for (size_t label = 0; label < nodes; ++label) {
        if (label != largest_label)
            second_largest_size = std::max(
                second_largest_size, component_sizes[label]);
    }
    const uint64_t minimum_reachable = static_cast<uint64_t>(
        std::ceil(
            ADAPTIVE_SOURCE_MIN_REACHABILITY
            * static_cast<double>(nodes)));
    if (largest_size < minimum_reachable) {
        throw std::runtime_error(
            "Largest component fails adaptive reachability threshold");
    }

    std::vector<NodeID> ranked;
    ranked.reserve(largest_size);
    for (size_t node = 0; node < nodes; ++node) {
        if (
            static_cast<size_t>(components[node]) == largest_label
            && graph.out_degree(static_cast<NodeID>(node)) > 0
        ) {
            ranked.push_back(static_cast<NodeID>(node));
        }
    }
    if (ranked.size() < ADAPTIVE_SOURCE_COUNT)
        throw std::runtime_error(
            "Largest component has too few non-isolated vertices");

    __gnu_parallel::sort(
        ranked.begin(),
        ranked.end(),
        [&graph](NodeID left, NodeID right) {
            const auto left_degree = graph.out_degree(left);
            const auto right_degree = graph.out_degree(right);
            if (left_degree != right_degree)
                return left_degree < right_degree;
            return graph.get_org_id(left) < graph.get_org_id(right);
        });

    AdaptiveSourceSelection selection;
    selection.nodes = nodes;
    selection.directed_edges =
        static_cast<uint64_t>(graph.num_edges_directed());
    selection.largest_component_label =
        static_cast<int64_t>(largest_label);
    selection.largest_component_size = largest_size;
    selection.largest_component_min_original = largest_min_original;
    selection.second_largest_component_size = second_largest_size;
    selection.minimum_reachable_vertices = minimum_reachable;
    std::unordered_set<int64_t> selected_originals;

    for (
        size_t requested = 0;
        requested < ADAPTIVE_SOURCE_COUNT;
        ++requested
    ) {
        bool found = false;
        AdaptiveSourceRecord record;
        record.source_index = requested;
        record.requested_octile = requested;
        for (size_t realized : AdaptiveOctileFallbackOrder(requested)) {
            const size_t begin =
                realized * ranked.size() / ADAPTIVE_SOURCE_COUNT;
            const size_t end =
                (realized + 1) * ranked.size()
                / ADAPTIVE_SOURCE_COUNT;
            if (begin >= end)
                continue;
            const size_t size = end - begin;
            const size_t offset = static_cast<size_t>(
                MixAdaptiveSourceSeed(
                    ADAPTIVE_SOURCE_SEED
                    ^ (requested << 16)
                    ^ realized)
                % size);
            for (size_t attempt = 0; attempt < size; ++attempt) {
                const size_t candidate_offset =
                    (offset + attempt) % size;
                const size_t rank = begin + candidate_offset;
                const NodeID internal = ranked[rank];
                const int64_t original =
                    static_cast<int64_t>(graph.get_org_id(internal));
                record.replacement_path.push_back(
                    "octile=" + std::to_string(realized)
                    + ",offset=" + std::to_string(candidate_offset));
                if (!selected_originals.insert(original).second)
                    continue;
                record.original = original;
                record.internal = static_cast<int64_t>(internal);
                record.out_degree =
                    static_cast<int64_t>(graph.out_degree(internal));
                record.realized_octile = realized;
                record.rank = rank;
                record.octile_start = begin;
                record.octile_end = end;
                record.candidate_offset = candidate_offset;
                record.reachable_vertices = largest_size;
                record.reachable_fraction = (
                    static_cast<double>(largest_size) / nodes);
                found = true;
                break;
            }
            if (found)
                break;
        }
        if (!found)
            throw std::runtime_error(
                "Adaptive source policy exhausted all octiles");
        selection.sources.push_back(std::move(record));
    }
    return selection;
}

template <typename GraphT, typename ComponentRange>
void WriteAdaptiveSourceManifest(
    const GraphT& graph,
    const ComponentRange& components,
    const std::string& graph_name,
    const std::string& graph_path,
    const std::string& output_path,
    bool component_verified,
    const std::string& component_verifier,
    const nlohmann::json& labeling_features) {
    if (!component_verified || component_verifier.empty())
        throw std::invalid_argument(
            "Adaptive source manifest requires component verification");
    const auto selection = SelectAdaptiveSources(graph, components);
    nlohmann::json payload = {
        {"schema", "adaptive-source-manifest/v1"},
        {"policy_id", ADAPTIVE_SOURCE_POLICY_ID},
        {"seed", ADAPTIVE_SOURCE_SEED},
        {"source_count", ADAPTIVE_SOURCE_COUNT},
        {"minimum_reachability_fraction",
         ADAPTIVE_SOURCE_MIN_REACHABILITY},
        {"candidate_order", "seeded-cyclic-octile-scan/v1"},
        {"graph", graph_name},
        {"graph_path", graph_path},
        {"nodes", selection.nodes},
        {"directed_edges", selection.directed_edges},
        {"undirected_edges", selection.directed_edges / 2},
        {"largest_component_label",
         selection.largest_component_label},
        {"largest_component_size",
         selection.largest_component_size},
        {"largest_component_min_original",
         selection.largest_component_min_original},
        {"second_largest_component_size",
         selection.second_largest_component_size},
        {"minimum_reachable_vertices",
         selection.minimum_reachable_vertices},
        {"component_verification", "pass"},
        {"component_verifier", component_verifier},
        {"labeling_features", labeling_features},
        {"sources", nlohmann::json::array()},
    };
    for (const auto& source : selection.sources) {
        payload["sources"].push_back({
            {"source_index", source.source_index},
            {"source_id", source.original},
            {"source_internal", source.internal},
            {"source_out_degree", source.out_degree},
            {"requested_octile", source.requested_octile},
            {"realized_octile", source.realized_octile},
            {"rank", source.rank},
            {"octile_start", source.octile_start},
            {"octile_end", source.octile_end},
            {"candidate_offset", source.candidate_offset},
            {"reachable_vertices", source.reachable_vertices},
            {"reachable_fraction", source.reachable_fraction},
            {"replacement_path", source.replacement_path},
        });
    }

    const std::string temporary = output_path + ".tmp";
    {
        std::ofstream output(temporary);
        if (!output)
            throw std::runtime_error(
                "Cannot open adaptive source manifest: " + temporary);
        output << payload.dump(2) << "\n";
        if (!output)
            throw std::runtime_error(
                "Failed to write adaptive source manifest");
    }
    if (std::rename(temporary.c_str(), output_path.c_str()) != 0) {
        std::remove(temporary.c_str());
        throw std::runtime_error(
            "Failed to publish adaptive source manifest");
    }
}

}  // namespace analysis
}  // namespace graphbrew

#endif  // GRAPHBREW_ANALYSIS_ADAPTIVE_SOURCE_POLICY_H_
