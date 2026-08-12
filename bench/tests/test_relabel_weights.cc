#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "graphbrew/reorder/reorder_types.h"

namespace
{

using Edge = std::tuple<NodeID, NodeID, WeightT>;

void Require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

WGraph BuildUndirectedGraph()
{
    pvector<SGOffset> offsets(5);
    offsets[0] = 0;
    offsets[1] = 2;
    offsets[2] = 4;
    offsets[3] = 6;
    offsets[4] = 8;
    auto *neighbors = new WNode[8]{
        WNode(1, 5), WNode(2, 7),
        WNode(0, 5), WNode(3, 11),
        WNode(0, 7), WNode(3, 13),
        WNode(1, 11), WNode(2, 13),
    };
    auto **index = WGraph::GenIndex(offsets, neighbors);
    return WGraph(4, index, neighbors);
}

WGraph BuildDirectedGraph()
{
    pvector<SGOffset> out_offsets(5);
    out_offsets[0] = 0;
    out_offsets[1] = 2;
    out_offsets[2] = 2;
    out_offsets[3] = 3;
    out_offsets[4] = 4;
    auto *out_neighbors = new WNode[4]{
        WNode(1, 5), WNode(2, 7), WNode(1, 9), WNode(0, 11),
    };
    auto **out_index = WGraph::GenIndex(out_offsets, out_neighbors);

    pvector<SGOffset> in_offsets(5);
    in_offsets[0] = 0;
    in_offsets[1] = 1;
    in_offsets[2] = 3;
    in_offsets[3] = 4;
    in_offsets[4] = 4;
    auto *in_neighbors = new WNode[4]{
        WNode(3, 11), WNode(0, 5), WNode(2, 9), WNode(0, 7),
    };
    auto **in_index = WGraph::GenIndex(in_offsets, in_neighbors);
    return WGraph(
        4, out_index, out_neighbors, in_index, in_neighbors);
}

std::vector<Edge> CollectEdges(const WGraph &graph)
{
    std::vector<Edge> edges;
    for (NodeID source = 0; source < graph.num_nodes(); ++source)
    {
        for (WNode destination : graph.out_neigh(source))
            edges.emplace_back(source, destination.v, destination.w);
    }
    std::sort(edges.begin(), edges.end());
    return edges;
}

std::vector<Edge> RemapEdges(
    const std::vector<Edge> &edges,
    const pvector<NodeID> &mapping)
{
    std::vector<Edge> remapped;
    remapped.reserve(edges.size());
    for (const auto &[source, destination, weight] : edges)
        remapped.emplace_back(
            mapping[source], mapping[destination], weight);
    std::sort(remapped.begin(), remapped.end());
    return remapped;
}

pvector<NodeID> TestMapping()
{
    pvector<NodeID> mapping(4);
    mapping[0] = 2;
    mapping[1] = 0;
    mapping[2] = 3;
    mapping[3] = 1;
    return mapping;
}

void TestBuilderRelabelUndirected()
{
    WGraph graph = BuildUndirectedGraph();
    const auto original = CollectEdges(graph);
    pvector<NodeID> mapping = TestMapping();
    WGraph relabeled = WeightedBuilder::RelabelByMapping(graph, mapping);
    Require(
        CollectEdges(relabeled) == RemapEdges(original, mapping),
        "builder undirected relabel changed weighted edges");
}

void TestBuilderRelabelDirected()
{
    WGraph graph = BuildDirectedGraph();
    const auto original = CollectEdges(graph);
    pvector<NodeID> mapping = TestMapping();
    WGraph relabeled = WeightedBuilder::RelabelByMapping(graph, mapping);
    Require(
        CollectEdges(relabeled) == RemapEdges(original, mapping),
        "builder directed relabel changed weighted edges");
}

void TestBuilderRelabelV2()
{
    WGraph graph = BuildUndirectedGraph();
    const auto original = CollectEdges(graph);
    pvector<NodeID> mapping = TestMapping();
    WGraph relabeled = WeightedBuilder::RelabelByMapping_v2(graph, mapping);
    Require(
        CollectEdges(relabeled) == RemapEdges(original, mapping),
        "builder v2 relabel changed weighted edges");
}

void TestStandaloneRelabel()
{
    WGraph graph = BuildUndirectedGraph();
    const auto original = CollectEdges(graph);
    pvector<NodeID> mapping = TestMapping();
    WGraph relabeled =
        RelabelByMappingStandalone<NodeID, WNode, true>(graph, mapping);
    Require(
        CollectEdges(relabeled) == RemapEdges(original, mapping),
        "standalone relabel changed weighted edges");
}

void TestDegreeRelabelPreservesWeights()
{
    WGraph graph = BuildUndirectedGraph();
    auto original = CollectEdges(graph);
    WGraph relabeled = WeightedBuilder::RelabelByDegree(graph);
    auto changed = CollectEdges(relabeled);
    std::vector<WeightT> original_weights;
    std::vector<WeightT> changed_weights;
    for (const auto &[source, destination, weight] : original)
        original_weights.push_back(weight);
    for (const auto &[source, destination, weight] : changed)
        changed_weights.push_back(weight);
    std::sort(original_weights.begin(), original_weights.end());
    std::sort(changed_weights.begin(), changed_weights.end());
    Require(
        original_weights == changed_weights,
        "degree relabel changed edge weights");
}

void TestSourcePickerUsesOriginalIds()
{
    WGraph graph = BuildUndirectedGraph();
    pvector<NodeID> mapping = TestMapping();
    WGraph relabeled = WeightedBuilder::RelabelByMapping(graph, mapping);

    SourcePicker<WGraph> explicit_source(relabeled, 0, 1);
    NodeID internal = explicit_source.PickNext();
    Require(internal == mapping[0], "explicit source was not resolved in O(1) map");
    Require(
        explicit_source.last_original_source() == 0,
        "explicit source lost its original ID");
    Require(
        relabeled.get_org_id(internal) == 0,
        "resolved source does not map back to the requested original ID");
}

void TestSourceSequenceIsOrderingInvariant()
{
    WGraph original = BuildUndirectedGraph();
    WGraph source = BuildUndirectedGraph();
    pvector<NodeID> mapping = TestMapping();
    WGraph relabeled = WeightedBuilder::RelabelByMapping(source, mapping);
    SourcePicker<WGraph> original_picker(original, -1, 4);
    SourcePicker<WGraph> relabeled_picker(relabeled, -1, 4);

    for (int trial = 0; trial < 4; ++trial)
    {
        NodeID original_internal = original_picker.PickNext();
        NodeID relabeled_internal = relabeled_picker.PickNext();
        Require(
            original_picker.last_original_source() ==
                relabeled_picker.last_original_source(),
            "source sequence changed after reordering");
        Require(
            original.get_org_id(original_internal) ==
                relabeled.get_org_id(relabeled_internal),
            "picked internal sources represent different original vertices");
    }
}

} // namespace

int main()
{
    try
    {
        TestBuilderRelabelUndirected();
        TestBuilderRelabelDirected();
        TestBuilderRelabelV2();
        TestStandaloneRelabel();
        TestDegreeRelabelPreservesWeights();
        TestSourcePickerUsesOriginalIds();
        TestSourceSequenceIsOrderingInvariant();
    }
    catch (const std::exception &error)
    {
        std::cerr << "weighted relabel test failed: "
                  << error.what() << std::endl;
        return 1;
    }
    std::cout << "weighted relabel tests passed" << std::endl;
    return 0;
}
