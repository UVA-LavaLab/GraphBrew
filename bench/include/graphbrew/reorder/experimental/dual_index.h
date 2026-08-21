#ifndef GRAPHBREW_REORDER_EXPERIMENTAL_DUAL_INDEX_H_
#define GRAPHBREW_REORDER_EXPERIMENTAL_DUAL_INDEX_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "common.h"

namespace graphbrew::experimental {

template <typename K>
struct DualIndexLayout {
    // *ToOriginal maps a domain ID to the original vertex ID.
    // originalTo* maps an original vertex ID to its domain ID.
    std::vector<K> sourceToOriginal;
    std::vector<K> originalToSource;
    std::vector<K> destinationToOriginal;
    std::vector<K> originalToDestination;
    std::vector<size_t> outOffsets;
    std::vector<K> outDestinationIds;
    std::vector<size_t> inOffsets;
    std::vector<K> inSourceIds;

    size_t nodeCount() const
    {
        return sourceToOriginal.size();
    }

    size_t directedEdgeCount() const
    {
        return outDestinationIds.size();
    }

    uint64_t modeledBytes() const
    {
        // Logical bytes owned by this layout only. This excludes vector
        // metadata/capacity, construction temporaries, the original graph,
        // and any composite source<->destination traversal translation.
        uint64_t bytes = 0;
        uint64_t mapEntries = 0;
        for (size_t entries : {
                 sourceToOriginal.size(),
                 originalToSource.size(),
                 destinationToOriginal.size(),
                 originalToDestination.size(),
             }) {
            mapEntries = checkedUint64Add(
                mapEntries,
                entries,
                "Dual-index map-entry count overflowed");
        }
        bytes = checkedUint64Add(
            bytes,
            checkedUint64Multiply(
                mapEntries,
                sizeof(K),
                "Dual-index map-byte count overflowed"),
            "Dual-index byte count overflowed");
        const uint64_t offsetEntries = checkedUint64Add(
            outOffsets.size(),
            inOffsets.size(),
            "Dual-index offset-entry count overflowed");
        bytes = checkedUint64Add(
            bytes,
            checkedUint64Multiply(
                offsetEntries,
                sizeof(size_t),
                "Dual-index offset-byte count overflowed"),
            "Dual-index byte count overflowed");
        const uint64_t edgeEntries = checkedUint64Add(
            outDestinationIds.size(),
            inSourceIds.size(),
            "Dual-index edge-entry count overflowed");
        return checkedUint64Add(
            bytes,
            checkedUint64Multiply(
                edgeEntries,
                sizeof(K),
                "Dual-index edge-byte count overflowed"),
            "Dual-index byte count overflowed");
    }
};

template <typename K>
DualIndexLayout<K> buildDualIndexLayout(
    const std::vector<std::vector<K>>& outgoing,
    const std::vector<K>& sourceIdToOriginal,
    const std::vector<K>& destinationIdToOriginal)
{
    const size_t nodeCount = outgoing.size();
    if (
        sourceIdToOriginal.size() != nodeCount
        || destinationIdToOriginal.size() != nodeCount
    ) {
        throw std::invalid_argument(
            "Dual-index orders must cover every vertex");
    }

    DualIndexLayout<K> layout;
    layout.sourceToOriginal = sourceIdToOriginal;
    layout.originalToSource = invertOrder(sourceIdToOriginal);
    layout.destinationToOriginal = destinationIdToOriginal;
    layout.originalToDestination =
        invertOrder(destinationIdToOriginal);

    std::vector<size_t> outDegrees(nodeCount, 0);
    std::vector<size_t> inDegrees(nodeCount, 0);
    size_t edgeCount = 0;
    for (size_t originalSource = 0;
         originalSource < nodeCount;
         ++originalSource) {
        const size_t sourceId = static_cast<size_t>(
            layout.originalToSource[originalSource]);
        outDegrees[sourceId] = outgoing[originalSource].size();
        edgeCount = checkedSizeAdd(
            edgeCount,
            outgoing[originalSource].size(),
            "Dual-index edge count overflowed");
        for (K originalDestination : outgoing[originalSource]) {
            const size_t destination = checkedIndex(
                originalDestination,
                nodeCount,
                "Dual-index edge target is out of range");
            const size_t destinationId = static_cast<size_t>(
                layout.originalToDestination[destination]);
            inDegrees[destinationId] = checkedSizeAdd(
                inDegrees[destinationId],
                1,
                "Dual-index indegree overflowed");
        }
    }

    const size_t offsetCount = checkedSizeAdd(
        nodeCount,
        1,
        "Dual-index offset count overflowed");
    layout.outOffsets.assign(offsetCount, 0);
    layout.inOffsets.assign(offsetCount, 0);
    for (size_t id = 0; id < nodeCount; ++id) {
        layout.outOffsets[id + 1] =
            checkedSizeAdd(
                layout.outOffsets[id],
                outDegrees[id],
                "Dual-index outgoing offsets overflowed");
        layout.inOffsets[id + 1] =
            checkedSizeAdd(
                layout.inOffsets[id],
                inDegrees[id],
                "Dual-index incoming offsets overflowed");
    }
    layout.outDestinationIds.resize(edgeCount);
    layout.inSourceIds.resize(edgeCount);
    std::vector<size_t> outCursor = layout.outOffsets;
    std::vector<size_t> inCursor = layout.inOffsets;

    for (size_t originalSource = 0;
         originalSource < nodeCount;
         ++originalSource) {
        const K sourceId =
            layout.originalToSource[originalSource];
        for (K originalDestination : outgoing[originalSource]) {
            const size_t destination = checkedIndex(
                originalDestination,
                nodeCount,
                "Dual-index edge target is out of range");
            const K destinationId =
                layout.originalToDestination[destination];
            layout.outDestinationIds[
                outCursor[static_cast<size_t>(sourceId)]++
            ] = destinationId;
            layout.inSourceIds[
                inCursor[static_cast<size_t>(destinationId)]++
            ] = sourceId;
        }
    }

    for (size_t id = 0; id < nodeCount; ++id) {
        std::sort(
            layout.outDestinationIds.begin() + layout.outOffsets[id],
            layout.outDestinationIds.begin() + layout.outOffsets[id + 1]);
        std::sort(
            layout.inSourceIds.begin() + layout.inOffsets[id],
            layout.inSourceIds.begin() + layout.inOffsets[id + 1]);
    }
    return layout;
}

template <typename K>
bool validateDualIndexLayout(
    const DualIndexLayout<K>& layout,
    const std::vector<std::vector<K>>& outgoing)
{
    try {
        const size_t nodeCount = outgoing.size();
        const size_t offsetCount = checkedSizeAdd(
            nodeCount,
            1,
            "Dual-index offset count overflowed");
        if (
            layout.sourceToOriginal.size() != nodeCount
            || layout.originalToSource.size() != nodeCount
            || layout.destinationToOriginal.size() != nodeCount
            || layout.originalToDestination.size() != nodeCount
            || layout.outOffsets.size() != offsetCount
            || layout.inOffsets.size() != offsetCount
            || layout.outDestinationIds.size()
                != layout.inSourceIds.size()
            || layout.outOffsets.front() != 0
            || layout.inOffsets.front() != 0
            || layout.outOffsets.back()
                != layout.outDestinationIds.size()
            || layout.inOffsets.back()
                != layout.inSourceIds.size()
        ) {
            return false;
        }

        if (
            invertOrder(layout.sourceToOriginal)
                != layout.originalToSource
            || invertOrder(layout.destinationToOriginal)
                != layout.originalToDestination
        ) {
            return false;
        }

        for (size_t id = 0; id < nodeCount; ++id) {
            if (
                layout.outOffsets[id] > layout.outOffsets[id + 1]
                || layout.inOffsets[id] > layout.inOffsets[id + 1]
                || layout.outOffsets[id + 1]
                    > layout.outDestinationIds.size()
                || layout.inOffsets[id + 1]
                    > layout.inSourceIds.size()
            ) {
                return false;
            }
        }
        for (K destinationId : layout.outDestinationIds) {
            (void)checkedIndex(
                destinationId,
                nodeCount,
                "Dual-index outgoing neighbor is out of range");
        }
        for (K sourceId : layout.inSourceIds) {
            (void)checkedIndex(
                sourceId,
                nodeCount,
                "Dual-index incoming neighbor is out of range");
        }
        for (size_t id = 0; id < nodeCount; ++id) {
            if (
                !std::is_sorted(
                    layout.outDestinationIds.begin()
                        + layout.outOffsets[id],
                    layout.outDestinationIds.begin()
                        + layout.outOffsets[id + 1])
                || !std::is_sorted(
                    layout.inSourceIds.begin()
                        + layout.inOffsets[id],
                    layout.inSourceIds.begin()
                        + layout.inOffsets[id + 1])
            ) {
                return false;
            }
        }

        std::vector<std::pair<K, K>> expected;
        std::vector<std::pair<K, K>> decodedOut;
        std::vector<std::pair<K, K>> decodedIn;
        for (size_t source = 0; source < nodeCount; ++source) {
            for (K destination : outgoing[source]) {
                (void)checkedIndex(
                    destination,
                    nodeCount,
                    "Dual-index edge target is out of range");
                expected.push_back({
                    static_cast<K>(source), destination});
            }
        }
        for (size_t sourceId = 0;
             sourceId < nodeCount;
             ++sourceId) {
            const K originalSource =
                layout.sourceToOriginal[sourceId];
            for (size_t edge = layout.outOffsets[sourceId];
                 edge < layout.outOffsets[sourceId + 1];
                 ++edge) {
                const size_t destinationId = checkedIndex(
                    layout.outDestinationIds[edge],
                    nodeCount,
                    "Dual-index outgoing neighbor is out of range");
                decodedOut.push_back({
                    originalSource,
                    layout.destinationToOriginal[destinationId]});
            }
        }
        for (size_t destinationId = 0;
             destinationId < nodeCount;
             ++destinationId) {
            const K originalDestination =
                layout.destinationToOriginal[destinationId];
            for (size_t edge = layout.inOffsets[destinationId];
                 edge < layout.inOffsets[destinationId + 1];
                 ++edge) {
                const size_t sourceId = checkedIndex(
                    layout.inSourceIds[edge],
                    nodeCount,
                    "Dual-index incoming neighbor is out of range");
                decodedIn.push_back({
                    layout.sourceToOriginal[sourceId],
                    originalDestination});
            }
        }

        std::sort(expected.begin(), expected.end());
        std::sort(decodedOut.begin(), decodedOut.end());
        std::sort(decodedIn.begin(), decodedIn.end());
        return expected == decodedOut && expected == decodedIn;
    } catch (...) {
        return false;
    }
}

}  // namespace graphbrew::experimental

#endif  // GRAPHBREW_REORDER_EXPERIMENTAL_DUAL_INDEX_H_
