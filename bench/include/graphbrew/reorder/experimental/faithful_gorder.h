#ifndef GRAPHBREW_REORDER_EXPERIMENTAL_FAITHFUL_GORDER_H_
#define GRAPHBREW_REORDER_EXPERIMENTAL_FAITHFUL_GORDER_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../reorder_gorder.h"

namespace graphbrew::experimental {

inline bool faithfulGorderScoreRangeSafe(
    size_t vertexCount,
    int window)
{
    if (window <= 0) return false;
    const uint64_t window64 = static_cast<uint64_t>(window);
    const uint64_t maximum = static_cast<uint64_t>(
        std::numeric_limits<int>::max());
    if (2 * window64 > maximum) return false;
    const uint64_t vertexLimit =
        (maximum - 2 * window64) / (window64 + 1);
    return vertexCount <= vertexLimit;
}

template <typename K>
void faithfulGorderLocalOrder(
    const std::vector<std::vector<size_t>>& outNeighbors,
    const std::vector<std::vector<size_t>>& inNeighbors,
    size_t vertexCount,
    int window,
    std::vector<K>& order)
{
    const size_t size = vertexCount;
    if (
        outNeighbors.size() < size
        || inNeighbors.size() < size
    ) {
        throw std::invalid_argument(
            "Faithful local Gorder received mismatched adjacency");
    }
    if (window <= 0) {
        throw std::invalid_argument(
            "Faithful local Gorder requires a positive window");
    }

    order.clear();
    order.reserve(size);
    if (size == 0) return;
    if (size == 1) {
        order.push_back(0);
        return;
    }
    if (!faithfulGorderScoreRangeSafe(size, window)) {
        throw std::overflow_error(
            "Faithful local Gorder community exceeds score range");
    }
    for (size_t local = 0; local < size; ++local) {
        if (!std::is_sorted(
                outNeighbors[local].begin(),
                outNeighbors[local].end())) {
            throw std::invalid_argument(
                "Faithful local Gorder requires sorted out-neighbors");
        }
        for (size_t target : outNeighbors[local]) {
            if (target >= size) {
                throw std::invalid_argument(
                    "Faithful local Gorder out-neighbor is out of range");
            }
        }
        for (size_t source : inNeighbors[local]) {
            if (source >= size) {
                throw std::invalid_argument(
                    "Faithful local Gorder in-neighbor is out of range");
            }
        }
    }

    int seed = 0;
    int maxInDegree = -1;
    std::vector<K> zeroDegree;
    for (size_t local = 0; local < size; ++local) {
        const int inDegree =
            static_cast<int>(inNeighbors[local].size());
        const int outDegree =
            static_cast<int>(outNeighbors[local].size());
        if (inDegree > maxInDegree) {
            seed = static_cast<int>(local);
            maxInDegree = inDegree;
        } else if (inDegree + outDegree == 0) {
            zeroDegree.push_back(static_cast<K>(local));
        }
    }

    const int localCount = static_cast<int>(size);
    gorder_csr_detail::UnitHeap heap(localCount);
    for (int local = 0; local < localCount; ++local) {
        const int inDegree = static_cast<int>(
            inNeighbors[static_cast<size_t>(local)].size());
        heap.list[local].key = inDegree;
        heap.update[local] = -inDegree;
    }
    heap.ReConstruct();

    std::vector<char> active(size, 1);
    for (K local : zeroDegree) {
        active[local] = 0;
        heap.update[local] = std::numeric_limits<int>::max() / 2;
        heap.DeleteElement(static_cast<int>(local));
    }
    active[static_cast<size_t>(seed)] = 0;
    heap.update[seed] = std::numeric_limits<int>::max() / 2;
    heap.DeleteElement(seed);
    order.push_back(static_cast<K>(seed));

    auto scoreIncrement = [&](size_t local) {
        if (!active[local]) return;
        if (heap.update[local] == 0)
            heap.IncrementKey(static_cast<int>(local));
        else
            ++heap.update[local];
    };
    auto scoreDecrement = [&](size_t local) {
        if (active[local]) --heap.update[local];
    };
    const size_t hugeVertex = static_cast<size_t>(
        std::sqrt(static_cast<double>(size)));

    for (size_t in : inNeighbors[static_cast<size_t>(seed)]) {
        if (outNeighbors[in].size() > hugeVertex) continue;
        scoreIncrement(in);
        if (outNeighbors[in].size() > 1) {
            for (size_t sibling : outNeighbors[in])
                scoreIncrement(sibling);
        }
    }
    if (outNeighbors[static_cast<size_t>(seed)].size() <= hugeVertex) {
        for (size_t out : outNeighbors[static_cast<size_t>(seed)])
            scoreIncrement(out);
    }

    const size_t activeAfterSeed =
        size - zeroDegree.size() - 1;
    std::vector<char> popVertexExists(size, 0);
    for (size_t count = 1; count <= activeAfterSeed; ++count) {
        const int current = heap.ExtractMax();
        active[static_cast<size_t>(current)] = 0;
        heap.update[current] = std::numeric_limits<int>::max() / 2;
        order.push_back(static_cast<K>(current));

        const int popIndex = static_cast<int>(count) - window;
        if (popIndex >= 0) {
            const size_t oldLocal = static_cast<size_t>(
                order[static_cast<size_t>(popIndex)]);
            if (outNeighbors[oldLocal].size() <= hugeVertex) {
                for (size_t out : outNeighbors[oldLocal])
                    scoreDecrement(out);
            }
            for (size_t in : inNeighbors[oldLocal]) {
                if (outNeighbors[in].size() > hugeVertex)
                    continue;
                scoreDecrement(in);
                if (outNeighbors[in].size() <= 1)
                    continue;
                if (
                    std::binary_search(
                        outNeighbors[in].begin(),
                        outNeighbors[in].end(),
                        static_cast<size_t>(current))
                ) {
                    popVertexExists[in] = 1;
                } else {
                    for (size_t sibling : outNeighbors[in])
                        scoreDecrement(sibling);
                }
            }
        }

        const size_t currentLocal = static_cast<size_t>(current);
        if (outNeighbors[currentLocal].size() <= hugeVertex) {
            for (size_t out : outNeighbors[currentLocal])
                scoreIncrement(out);
        }
        for (size_t in : inNeighbors[currentLocal]) {
            if (outNeighbors[in].size() > hugeVertex)
                continue;
            scoreIncrement(in);
            if (popVertexExists[in]) {
                popVertexExists[in] = 0;
            } else if (outNeighbors[in].size() > 1) {
                for (size_t sibling : outNeighbors[in])
                    scoreIncrement(sibling);
            }
        }
    }

    if (!zeroDegree.empty()) {
        order.insert(
            order.end() - 1,
            zeroDegree.begin(),
            zeroDegree.end());
    }
}

template <typename K>
std::vector<K> faithfulGorderInducedSimpleOrder(
    const std::vector<std::vector<K>>& outgoing,
    int window)
{
    std::vector<std::vector<size_t>> localOut(outgoing.size());
    std::vector<std::vector<size_t>> localIn(outgoing.size());
    for (size_t source = 0; source < outgoing.size(); ++source) {
        auto& neighbors = localOut[source];
        neighbors.reserve(outgoing[source].size());
        for (K target : outgoing[source]) {
            if constexpr (std::is_signed<K>::value) {
                if (target < 0) {
                    throw std::invalid_argument(
                        "Faithful local Gorder target is negative");
                }
            }
            const size_t targetIndex = static_cast<size_t>(target);
            if (targetIndex >= outgoing.size()) {
                throw std::invalid_argument(
                    "Faithful local Gorder target is out of range");
            }
            neighbors.push_back(targetIndex);
        }
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(
            std::unique(neighbors.begin(), neighbors.end()),
            neighbors.end());
        for (size_t target : neighbors)
            localIn[target].push_back(source);
    }
    std::vector<K> order;
    faithfulGorderLocalOrder<K>(
        localOut, localIn, outgoing.size(), window, order);
    return order;
}

}  // namespace graphbrew::experimental

#endif  // GRAPHBREW_REORDER_EXPERIMENTAL_FAITHFUL_GORDER_H_
