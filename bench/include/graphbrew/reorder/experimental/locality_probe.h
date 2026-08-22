#ifndef GRAPHBREW_REORDER_EXPERIMENTAL_LOCALITY_PROBE_H_
#define GRAPHBREW_REORDER_EXPERIMENTAL_LOCALITY_PROBE_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "common.h"

namespace graphbrew::experimental {

inline constexpr size_t kStructuralProbeMaxVertices = 256;
inline constexpr size_t kStructuralProbeMaxDirectedEdges = 65536;
inline constexpr size_t kStructuralProbeMaxWindow = 64;
inline constexpr size_t kStructuralProbeMaxPredecessorComparisons =
    10000000;

struct StructuralProbeConfig {
    size_t window = 5;
    // Relative score delta: 300 means expensive >= cheap * 1.03.
    uint32_t minimumScoreGainBasisPoints = 300;
    // Absolute ratio cap: 10000 means expensive <= cheap * 1.00.
    uint32_t maximumSpanRatioBasisPoints = 10000;
};

struct StructuralProbeResult {
    uint64_t cheapWindowScore = 0;
    uint64_t expensiveWindowScore = 0;
    uint64_t cheapDirectedEdgeSpan = 0;
    uint64_t expensiveDirectedEdgeSpan = 0;
    double relativeScoreGain = 0.0;
    size_t inputArcVisits = 0;
    size_t normalizedDirectedEdges = 0;
    size_t edgeSpanVisits = 0;
    size_t windowPairEvaluations = 0;
    size_t directLinkSearches = 0;
    size_t predecessorComparisons = 0;
    bool chooseExpensive = false;
};

namespace locality_probe_detail {

template <typename K>
struct OrderMetrics {
    uint64_t windowScore = 0;
    uint64_t directedEdgeSpan = 0;
    size_t pairEvaluations = 0;
    size_t predecessorComparisons = 0;
};

template <typename K>
OrderMetrics<K> measureOrder(
    const std::vector<std::vector<K>>& sortedOut,
    const std::vector<std::vector<K>>& sortedIn,
    const std::vector<K>& order,
    size_t window)
{
    const std::vector<K> position = invertOrder(order);
    OrderMetrics<K> metrics;
    for (size_t source = 0; source < sortedOut.size(); ++source) {
        for (K target : sortedOut[source]) {
            const uint64_t left = position[source];
            const uint64_t right = position[checkedIndex(
                target,
                sortedOut.size(),
                "Structural probe edge target is out of range")];
            metrics.directedEdgeSpan = checkedUint64Add(
                metrics.directedEdgeSpan,
                left > right ? left - right : right - left,
                "Structural probe edge span overflowed");
        }
    }

    for (size_t rightPosition = 0;
         rightPosition < order.size();
         ++rightPosition) {
        const size_t begin =
            rightPosition > window ? rightPosition - window : 0;
        const K rightVertex = order[rightPosition];
        const size_t rightIndex = checkedIndex(
            rightVertex,
            order.size(),
            "Structural probe order contains an invalid vertex");
        for (size_t leftPosition = begin;
             leftPosition < rightPosition;
             ++leftPosition) {
            metrics.pairEvaluations = checkedSizeAdd(
                metrics.pairEvaluations,
                1,
                "Structural probe pair count overflowed");
            const K leftVertex = order[leftPosition];
            const size_t leftIndex = checkedIndex(
                leftVertex,
                order.size(),
                "Structural probe order contains an invalid vertex");
            if (
                std::binary_search(
                    sortedOut[leftIndex].begin(),
                    sortedOut[leftIndex].end(),
                    rightVertex)
            ) {
                metrics.windowScore = checkedUint64Add(
                    metrics.windowScore,
                    1,
                    "Structural probe score overflowed");
            }
            if (
                std::binary_search(
                    sortedOut[rightIndex].begin(),
                    sortedOut[rightIndex].end(),
                    leftVertex)
            ) {
                metrics.windowScore = checkedUint64Add(
                    metrics.windowScore,
                    1,
                    "Structural probe score overflowed");
            }
            auto leftIn = sortedIn[leftIndex].begin();
            auto rightIn = sortedIn[rightIndex].begin();
            while (
                leftIn != sortedIn[leftIndex].end()
                && rightIn != sortedIn[rightIndex].end()
            ) {
                metrics.predecessorComparisons = checkedSizeAdd(
                    metrics.predecessorComparisons,
                    1,
                    "Structural probe predecessor count overflowed");
                if (
                    metrics.predecessorComparisons
                    > kStructuralProbeMaxPredecessorComparisons
                ) {
                    throw std::invalid_argument(
                        "Structural probe exceeds its comparison limit");
                }
                if (*leftIn < *rightIn) {
                    ++leftIn;
                } else if (*rightIn < *leftIn) {
                    ++rightIn;
                } else {
                    metrics.windowScore = checkedUint64Add(
                        metrics.windowScore,
                        1,
                        "Structural probe score overflowed");
                    ++leftIn;
                    ++rightIn;
                }
            }
        }
    }
    return metrics;
}

}  // namespace locality_probe_detail

inline bool shouldChooseExpensive(
    uint64_t cheapScore,
    uint64_t expensiveScore,
    uint64_t cheapSpan,
    uint64_t expensiveSpan,
    uint32_t minimumScoreGainBasisPoints,
    uint32_t maximumSpanRatioBasisPoints)
{
    if (expensiveScore <= cheapScore) return false;
#if !defined(__SIZEOF_INT128__)
#error "Structural locality probe requires unsigned 128-bit integers"
#endif
    using Wide = unsigned __int128;
    const bool gainPasses =
        cheapScore == 0
        || static_cast<Wide>(expensiveScore - cheapScore) * 10000
            >= static_cast<Wide>(cheapScore)
                * minimumScoreGainBasisPoints;
    const bool spanPasses =
        static_cast<Wide>(expensiveSpan) * 10000
        <= static_cast<Wide>(cheapSpan)
            * maximumSpanRatioBasisPoints;
    return gainPasses && spanPasses;
}

template <typename K>
StructuralProbeResult compareLocalOrders(
    const std::vector<std::vector<K>>& outgoing,
    const std::vector<K>& cheapOrder,
    const std::vector<K>& expensiveOrder,
    const StructuralProbeConfig& config = {})
{
    if (outgoing.size() > kStructuralProbeMaxVertices) {
        throw std::invalid_argument(
            "Structural probe exceeds its vertex limit");
    }
    // The window contains preceding positions only. Each unordered pair in
    // that window earns one point per directed link and one point per
    // distinct common predecessor. Input duplicate arcs are collapsed.
    // Self-loops remain and can therefore contribute predecessor overlap.
    if (config.window == 0) {
        throw std::invalid_argument(
            "Structural probe window must be positive");
    }
    if (config.window > kStructuralProbeMaxWindow) {
        throw std::invalid_argument(
            "Structural probe exceeds its window limit");
    }
    if (
        cheapOrder.size() != outgoing.size()
        || expensiveOrder.size() != outgoing.size()
    ) {
        throw std::invalid_argument(
            "Structural probe orders must cover every vertex");
    }
    (void)invertOrder(cheapOrder);
    (void)invertOrder(expensiveOrder);

    std::vector<std::vector<K>> sortedOut = outgoing;
    std::vector<std::vector<K>> sortedIn(outgoing.size());
    size_t directedEdges = 0;
    size_t inputArcVisits = 0;
    for (size_t source = 0; source < sortedOut.size(); ++source) {
        auto& neighbors = sortedOut[source];
        inputArcVisits = checkedSizeAdd(
            inputArcVisits,
            neighbors.size(),
            "Structural probe input-arc count overflowed");
        if (inputArcVisits > kStructuralProbeMaxDirectedEdges) {
            throw std::invalid_argument(
                "Structural probe exceeds its edge limit");
        }
        for (K target : neighbors) {
            (void)checkedIndex(
                target,
                outgoing.size(),
                "Structural probe edge target is out of range");
        }
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(
            std::unique(neighbors.begin(), neighbors.end()),
            neighbors.end());
        directedEdges = checkedSizeAdd(
            directedEdges,
            neighbors.size(),
            "Structural probe edge count overflowed");
        for (K target : neighbors)
            sortedIn[checkedIndex(
                target,
                outgoing.size(),
                "Structural probe edge target is out of range")]
                .push_back(static_cast<K>(source));
    }
    for (auto& neighbors : sortedIn)
        std::sort(neighbors.begin(), neighbors.end());

    const auto cheap =
        locality_probe_detail::measureOrder(
            sortedOut, sortedIn, cheapOrder, config.window);
    const auto expensive =
        locality_probe_detail::measureOrder(
            sortedOut, sortedIn, expensiveOrder, config.window);

    StructuralProbeResult result;
    result.cheapWindowScore = cheap.windowScore;
    result.expensiveWindowScore = expensive.windowScore;
    result.cheapDirectedEdgeSpan = cheap.directedEdgeSpan;
    result.expensiveDirectedEdgeSpan =
        expensive.directedEdgeSpan;
    result.relativeScoreGain =
        (
            static_cast<double>(expensive.windowScore)
            - static_cast<double>(cheap.windowScore)
        ) / static_cast<double>(
            std::max<uint64_t>(1, cheap.windowScore));
    result.inputArcVisits = inputArcVisits;
    result.normalizedDirectedEdges = directedEdges;
    result.edgeSpanVisits = checkedSizeAdd(
        directedEdges,
        directedEdges,
        "Structural probe span-visit count overflowed");
    result.windowPairEvaluations = checkedSizeAdd(
        cheap.pairEvaluations,
        expensive.pairEvaluations,
        "Structural probe pair count overflowed");
    result.directLinkSearches = checkedSizeAdd(
        result.windowPairEvaluations,
        result.windowPairEvaluations,
        "Structural probe search count overflowed");
    result.predecessorComparisons = checkedSizeAdd(
        cheap.predecessorComparisons,
        expensive.predecessorComparisons,
        "Structural probe predecessor count overflowed");
    result.chooseExpensive = shouldChooseExpensive(
        cheap.windowScore,
        expensive.windowScore,
        cheap.directedEdgeSpan,
        expensive.directedEdgeSpan,
        config.minimumScoreGainBasisPoints,
        config.maximumSpanRatioBasisPoints);
    return result;
}

}  // namespace graphbrew::experimental

#endif  // GRAPHBREW_REORDER_EXPERIMENTAL_LOCALITY_PROBE_H_
