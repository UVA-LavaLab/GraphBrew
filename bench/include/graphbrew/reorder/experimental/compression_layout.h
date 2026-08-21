#ifndef GRAPHBREW_REORDER_EXPERIMENTAL_COMPRESSION_LAYOUT_H_
#define GRAPHBREW_REORDER_EXPERIMENTAL_COMPRESSION_LAYOUT_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "common.h"

namespace graphbrew::experimental {

inline constexpr size_t kCompressionMaxVertices = 256;
inline constexpr size_t kCompressionMaxPasses = 32;
inline constexpr size_t kCompressionMaxDirectedEdges = 65536;

struct GapEncodingMetrics {
    // Sum of integer bit widths: a deterministic lower-bound proxy, not a
    // self-delimiting encoded size.
    uint64_t modeledGapBits = 0;
    // Realizable unsigned-varint payload bytes for the modeled integers.
    uint64_t modeledVarintBytes = 0;
    uint64_t fixedNeighborBytes = 0;
    uint64_t fixedOffsetBytes = 0;
    uint64_t weightBytes = 0;
    size_t directedEdges = 0;
};

template <typename K>
struct CompressionRefinementResult {
    std::vector<K> order;
    GapEncodingMetrics before;
    GapEncodingMetrics after;
    // Refinement optimizes modeledGapBits only. Varint bytes may increase.
    size_t acceptedSwaps = 0;
};

namespace compression_detail {

inline uint64_t zigZag(int64_t value)
{
    if (value >= 0)
        return static_cast<uint64_t>(value) * 2;
    return static_cast<uint64_t>(-(value + 1)) * 2 + 1;
}

inline uint64_t modeledBits(uint64_t value)
{
    uint64_t bits = 1;
    while (value > 1) {
        value >>= 1;
        ++bits;
    }
    return bits;
}

inline uint64_t varintBytes(uint64_t value)
{
    uint64_t bytes = 1;
    while (value >= 128) {
        value >>= 7;
        ++bytes;
    }
    return bytes;
}

}  // namespace compression_detail

template <typename K>
GapEncodingMetrics measureGapEncoding(
    const std::vector<std::vector<K>>& outgoing,
    const std::vector<K>& order,
    size_t weightBytesPerEdge = 0)
{
    if (order.size() != outgoing.size()) {
        throw std::invalid_argument(
            "Compression order must cover every vertex");
    }
    if (
        !order.empty()
        && order.size() - 1
            > static_cast<size_t>(
                std::numeric_limits<int64_t>::max())
    ) {
        throw std::overflow_error(
            "Compression positions exceed signed-delta range");
    }
    const std::vector<K> position = invertOrder(order);

    GapEncodingMetrics metrics;
    metrics.fixedOffsetBytes = checkedUint64Multiply(
        checkedSizeAdd(
            outgoing.size(),
            1,
            "Compression offset count overflowed"),
        sizeof(size_t),
        "Compression offset bytes overflowed");
    for (size_t originalSource = 0;
         originalSource < outgoing.size();
         ++originalSource) {
        std::vector<K> transformed;
        transformed.reserve(outgoing[originalSource].size());
        for (K originalTarget : outgoing[originalSource]) {
            const size_t target = checkedIndex(
                originalTarget,
                outgoing.size(),
                "Compression edge target is out of range");
            transformed.push_back(position[target]);
        }
        std::sort(transformed.begin(), transformed.end());
        metrics.directedEdges = checkedSizeAdd(
            metrics.directedEdges,
            transformed.size(),
            "Compression edge count overflowed");
        if (transformed.empty()) continue;

        const int64_t source =
            static_cast<int64_t>(position[originalSource]);
        const int64_t first =
            static_cast<int64_t>(transformed[0]);
        uint64_t encoded =
            compression_detail::zigZag(first - source);
        metrics.modeledGapBits = checkedUint64Add(
            metrics.modeledGapBits,
            compression_detail::modeledBits(encoded),
            "Compression modeled-bit count overflowed");
        metrics.modeledVarintBytes = checkedUint64Add(
            metrics.modeledVarintBytes,
            compression_detail::varintBytes(encoded),
            "Compression modeled-byte count overflowed");
        for (size_t edge = 1; edge < transformed.size(); ++edge) {
            encoded = static_cast<uint64_t>(
                transformed[edge] - transformed[edge - 1]);
            metrics.modeledGapBits = checkedUint64Add(
                metrics.modeledGapBits,
                compression_detail::modeledBits(encoded),
                "Compression modeled-bit count overflowed");
            metrics.modeledVarintBytes = checkedUint64Add(
                metrics.modeledVarintBytes,
                compression_detail::varintBytes(encoded),
                "Compression modeled-byte count overflowed");
        }
    }
    metrics.fixedNeighborBytes = checkedUint64Multiply(
        metrics.directedEdges,
        sizeof(K),
        "Compression fixed-neighbor bytes overflowed");
    metrics.weightBytes = checkedUint64Multiply(
        metrics.directedEdges,
        weightBytesPerEdge,
        "Compression weight bytes overflowed");
    return metrics;
}

template <typename K>
CompressionRefinementResult<K> refineCompressionOrder(
    const std::vector<std::vector<K>>& outgoing,
    const std::vector<K>& baseOrder,
    size_t maxPasses = 4,
    size_t weightBytesPerEdge = 0)
{
    if (baseOrder.size() != outgoing.size()) {
        throw std::invalid_argument(
            "Compression order must cover every vertex");
    }
    if (outgoing.size() > kCompressionMaxVertices) {
        throw std::invalid_argument(
            "Compression refinement exceeds its validity size limit");
    }
    if (maxPasses > kCompressionMaxPasses) {
        throw std::invalid_argument(
            "Compression refinement exceeds its pass limit");
    }
    size_t directedEdges = 0;
    for (const auto& neighbors : outgoing) {
        directedEdges = checkedSizeAdd(
            directedEdges,
            neighbors.size(),
            "Compression refinement edge count overflowed");
    }
    if (directedEdges > kCompressionMaxDirectedEdges) {
        throw std::invalid_argument(
            "Compression refinement exceeds its edge limit");
    }

    CompressionRefinementResult<K> result;
    result.order = baseOrder;
    result.before = measureGapEncoding(
        outgoing, result.order, weightBytesPerEdge);
    result.after = result.before;

    for (size_t pass = 0; pass < maxPasses; ++pass) {
        bool changed = false;
        for (size_t position = 0;
             position + 1 < result.order.size();
             ++position) {
            std::swap(
                result.order[position],
                result.order[position + 1]);
            const auto candidate = measureGapEncoding(
                outgoing, result.order, weightBytesPerEdge);
            if (
                candidate.modeledGapBits
                < result.after.modeledGapBits
            ) {
                result.after = candidate;
                ++result.acceptedSwaps;
                changed = true;
            } else {
                std::swap(
                    result.order[position],
                    result.order[position + 1]);
            }
        }
        if (!changed) break;
    }
    return result;
}

}  // namespace graphbrew::experimental

#endif  // GRAPHBREW_REORDER_EXPERIMENTAL_COMPRESSION_LAYOUT_H_
