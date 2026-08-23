#ifndef GRAPHBREW_REORDER_EXPERIMENTAL_CAPACITY_RUNS_H_
#define GRAPHBREW_REORDER_EXPERIMENTAL_CAPACITY_RUNS_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <graph.h>

#include "common.h"

namespace graphbrew::experimental {

template <typename K>
struct CapacityRunOrderResult {
    std::vector<K> order;
    std::vector<size_t> l2RunEnds;
    std::vector<size_t> llcRunEnds;
};

struct CapacityGeometry {
    size_t l2Bytes = 0;
    size_t llcBytes = 0;
    size_t propertyBytesPerVertex = 0;
    size_t l2TargetVertices = 0;
    size_t llcTargetVertices = 0;
};

inline CapacityGeometry resolvePinnedCapacityGeometry(
    size_t l2Bytes,
    size_t llcBytes,
    size_t propertyBytesPerVertex)
{
    if (
        l2Bytes == 0
        || llcBytes == 0
        || propertyBytesPerVertex == 0
    ) {
        throw std::invalid_argument(
            "Capacity runs require pinned L2, LLC, and property bytes");
    }
    if (llcBytes < l2Bytes) {
        throw std::invalid_argument(
            "Capacity-run LLC bytes must be at least L2 bytes");
    }
    CapacityGeometry geometry;
    geometry.l2Bytes = l2Bytes;
    geometry.llcBytes = llcBytes;
    geometry.propertyBytesPerVertex = propertyBytesPerVertex;
    geometry.l2TargetVertices = std::max<size_t>(
        1, l2Bytes / propertyBytesPerVertex);
    geometry.llcTargetVertices = std::max(
        geometry.l2TargetVertices,
        llcBytes / propertyBytesPerVertex);
    return geometry;
}

template <typename K, typename NodeID_T, typename DestID_T>
std::vector<std::vector<std::pair<K, uint64_t>>>
buildCapacityCommunityAdjacency(
    const std::vector<K>& membership,
    const CSRGraph<NodeID_T, DestID_T, true>& graph,
    size_t nodeCount,
    K communityCount)
{
    if (graph.directed()) {
        throw std::invalid_argument(
            "Capacity-run adjacency requires an undirected graph");
    }
    if (membership.size() < nodeCount) {
        throw std::invalid_argument(
            "Capacity-run membership does not cover every vertex");
    }
    if (nodeCount > static_cast<size_t>(graph.num_nodes())) {
        throw std::invalid_argument(
            "Capacity-run node count exceeds the graph");
    }
    for (size_t source = 0; source < nodeCount; ++source) {
        if (membership[source] >= communityCount) {
            throw std::invalid_argument(
                "Capacity-run membership is out of range");
        }
        for (auto neighbor : graph.out_neigh(
                 static_cast<NodeID_T>(source))) {
            NodeID_T target;
            if constexpr (std::is_same_v<DestID_T, NodeID_T>)
                target = neighbor;
            else
                target = neighbor.v;
            const size_t targetIndex = checkedIndex(
                target,
                nodeCount,
                "Capacity-run neighbor is out of range");
            if (membership[targetIndex] >= communityCount) {
                throw std::invalid_argument(
                    "Capacity-run membership is out of range");
            }
        }
    }
    static_assert(
        std::is_unsigned<K>::value,
        "Capacity-run community IDs must be unsigned");
    static_assert(
        sizeof(K) <= sizeof(uint32_t),
        "Capacity-run community IDs must fit in 32 bits");
    int threadCount = 1;
    #ifdef _OPENMP
    threadCount = omp_get_max_threads();
    #endif
    std::vector<std::vector<uint64_t>> localKeys(threadCount);

    #pragma omp parallel
    {
        int thread = 0;
        #ifdef _OPENMP
        thread = omp_get_thread_num();
        #endif
        auto& keys = localKeys[thread];
        #pragma omp for schedule(dynamic, 1024)
        for (size_t source = 0; source < nodeCount; ++source) {
            const K sourceCommunity = membership[source];
            for (auto neighbor : graph.out_neigh(
                     static_cast<NodeID_T>(source))) {
                NodeID_T target;
                if constexpr (std::is_same_v<DestID_T, NodeID_T>)
                    target = neighbor;
                else
                    target = neighbor.v;
                const size_t targetIndex =
                    static_cast<size_t>(target);
                const K targetCommunity = membership[targetIndex];
                if (sourceCommunity == targetCommunity) continue;
                const K low = std::min(
                    sourceCommunity, targetCommunity);
                const K high = std::max(
                    sourceCommunity, targetCommunity);
                keys.push_back(
                    (static_cast<uint64_t>(low) << 32)
                    | static_cast<uint64_t>(high));
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int thread = 0; thread < threadCount; ++thread)
        std::sort(localKeys[thread].begin(), localKeys[thread].end());

    struct Cursor {
        uint64_t key;
        int thread;
        size_t index;
    };
    struct CursorGreater {
        bool operator()(const Cursor& left, const Cursor& right) const {
            if (left.key != right.key) return left.key > right.key;
            return left.thread > right.thread;
        }
    };
    std::priority_queue<
        Cursor,
        std::vector<Cursor>,
        CursorGreater> cursors;
    for (int thread = 0; thread < threadCount; ++thread) {
        if (!localKeys[thread].empty())
            cursors.push({localKeys[thread][0], thread, 0});
    }

    std::vector<std::vector<std::pair<K, uint64_t>>> adjacency(
        communityCount);
    while (!cursors.empty()) {
        const uint64_t key = cursors.top().key;
        uint64_t count = 0;
        while (!cursors.empty() && cursors.top().key == key) {
            Cursor cursor = cursors.top();
            cursors.pop();
            const auto& keys = localKeys[cursor.thread];
            size_t next = cursor.index;
            while (next < keys.size() && keys[next] == key) {
                ++count;
                ++next;
            }
            if (next < keys.size())
                cursors.push({keys[next], cursor.thread, next});
        }
        const K low = static_cast<K>(key >> 32);
        const K high =
            static_cast<K>(key & UINT64_C(0xffffffff));
        adjacency[low].push_back({high, count});
        adjacency[high].push_back({low, count});
    }
    for (auto& neighbors : adjacency) {
        std::sort(
            neighbors.begin(), neighbors.end(),
            [](const auto& left, const auto& right) {
                return left.first < right.first;
            });
    }
    return adjacency;
}

template <typename K>
CapacityRunOrderResult<K> buildCapacityRunCommunityOrder(
    const std::vector<size_t>& communitySizes,
    const std::vector<std::vector<std::pair<K, uint64_t>>>& adjacency,
    const std::vector<K>& baseOrder,
    size_t l2TargetVertices,
    size_t llcTargetVertices)
{
    const size_t communityCount = communitySizes.size();
    CapacityRunOrderResult<K> result;
    result.order.reserve(communityCount);
    if (communityCount == 0) return result;
    if (adjacency.size() != communityCount) {
        throw std::invalid_argument(
            "Capacity-run adjacency size does not match communities");
    }
    for (size_t community = 0;
         community < communityCount;
         ++community) {
        for (const auto& [neighbor, weight] : adjacency[community]) {
            (void)weight;
            if (static_cast<size_t>(neighbor) >= communityCount) {
                throw std::invalid_argument(
                    "Capacity-run adjacency neighbor is out of range");
            }
        }
    }

    l2TargetVertices = std::max<size_t>(1, l2TargetVertices);
    llcTargetVertices = std::max(
        l2TargetVertices, llcTargetVertices);

    std::vector<size_t> baseRank(
        communityCount, communityCount);
    for (size_t index = 0; index < baseOrder.size(); ++index) {
        const size_t community =
            static_cast<size_t>(baseOrder[index]);
        if (community < communityCount)
            baseRank[community] = index;
    }
    for (size_t community = 0;
         community < communityCount;
         ++community) {
        if (baseRank[community] == communityCount)
            baseRank[community] = community;
    }

    struct Entry {
        uint64_t score;
        size_t rank;
        K community;
    };
    struct EntryLess {
        bool operator()(const Entry& left, const Entry& right) const {
            if (left.score != right.score)
                return left.score < right.score;
            if (left.rank != right.rank)
                return left.rank > right.rank;
            return left.community > right.community;
        }
    };
    using Queue = std::priority_queue<
        Entry, std::vector<Entry>, EntryLess>;

    std::vector<char> placed(communityCount, 0);
    std::map<size_t, std::set<std::pair<size_t, K>>>
        remainingBySize;
    std::vector<K> emptyCommunities;
    for (K community = 0;
         community < static_cast<K>(communityCount);
         ++community) {
        if (communitySizes[community] == 0) {
            placed[community] = 1;
            emptyCommunities.push_back(community);
        } else {
            remainingBySize[communitySizes[community]].insert(
                {baseRank[community], community});
        }
    }
    std::sort(
        emptyCommunities.begin(), emptyCommunities.end(),
        [&](K left, K right) {
            return baseRank[left] < baseRank[right];
        });
    const size_t activeCount =
        communityCount - emptyCommunities.size();

    auto nextLargestFitting = [&](size_t capacity) -> K {
        auto sizeIt = remainingBySize.upper_bound(capacity);
        if (sizeIt == remainingBySize.begin())
            return static_cast<K>(-1);
        --sizeIt;
        return sizeIt->second.begin()->second;
    };
    auto nextLargest = [&]() -> K {
        if (remainingBySize.empty()) return static_cast<K>(-1);
        return std::prev(
            remainingBySize.end())->second.begin()->second;
    };
    auto removeRemaining = [&](K community) {
        auto sizeIt = remainingBySize.find(
            communitySizes[community]);
        if (sizeIt == remainingBySize.end()) return;
        sizeIt->second.erase({baseRank[community], community});
        if (sizeIt->second.empty()) remainingBySize.erase(sizeIt);
    };

    std::vector<uint64_t> l2Score(communityCount, 0);
    std::vector<uint64_t> llcScore(communityCount, 0);
    std::vector<K> l2Touched;
    std::vector<K> llcTouched;
    Queue l2Queue;
    Queue llcQueue;

    auto resetFrontier = [](
        std::vector<uint64_t>& score,
        std::vector<K>& touched,
        Queue& queue) {
        for (K community : touched) score[community] = 0;
        touched.clear();
        queue = Queue();
    };
    auto updateFrontier = [&](
        K source,
        std::vector<uint64_t>& score,
        std::vector<K>& touched,
        Queue& queue) {
        for (const auto& [neighbor, weight] : adjacency[source]) {
            if (placed[neighbor]) continue;
            if (score[neighbor] == 0)
                touched.push_back(neighbor);
            score[neighbor] += weight;
            queue.push({
                score[neighbor],
                baseRank[neighbor],
                neighbor,
            });
        }
    };
    auto peekFrontierFitting = [&](
        Queue& queue,
        const std::vector<uint64_t>& score,
        size_t capacity) -> K {
        while (!queue.empty()) {
            const Entry& entry = queue.top();
            if (
                placed[entry.community]
                || entry.score != score[entry.community]
                || communitySizes[entry.community] > capacity
            ) {
                queue.pop();
                continue;
            }
            return entry.community;
        }
        return static_cast<K>(-1);
    };

    while (result.order.size() < activeCount) {
        resetFrontier(llcScore, llcTouched, llcQueue);
        resetFrontier(l2Score, l2Touched, l2Queue);
        K seed = nextLargest();
        if (seed == static_cast<K>(-1)) break;

        size_t llcUsed = 0;
        bool closeLLC = false;
        while (seed != static_cast<K>(-1) && !closeLLC) {
            resetFrontier(l2Score, l2Touched, l2Queue);
            size_t l2Used = 0;
            K current = seed;

            while (current != static_cast<K>(-1)) {
                const size_t size = communitySizes[current];
                if (
                    l2Used > 0
                    && (
                        l2Used >= l2TargetVertices
                        || size > l2TargetVertices - l2Used
                    )
                ) break;
                if (
                    llcUsed > 0
                    && (
                        llcUsed >= llcTargetVertices
                        || size > llcTargetVertices - llcUsed
                    )
                ) {
                    closeLLC = true;
                    break;
                }

                placed[current] = 1;
                removeRemaining(current);
                result.order.push_back(current);
                l2Used += size;
                llcUsed += size;
                updateFrontier(
                    current, l2Score, l2Touched, l2Queue);
                updateFrontier(
                    current, llcScore, llcTouched, llcQueue);

                const size_t l2Remaining =
                    l2Used < l2TargetVertices
                        ? l2TargetVertices - l2Used
                        : 0;
                const size_t llcRemaining =
                    llcUsed < llcTargetVertices
                        ? llcTargetVertices - llcUsed
                        : 0;
                const size_t remaining =
                    std::min(l2Remaining, llcRemaining);
                current = peekFrontierFitting(
                    l2Queue, l2Score, remaining);
                if (current == static_cast<K>(-1))
                    current = nextLargestFitting(remaining);
            }

            result.l2RunEnds.push_back(result.order.size());
            if (
                closeLLC
                || result.order.size() == activeCount
                || llcUsed >= llcTargetVertices
            ) break;

            const size_t llcRemaining =
                llcUsed < llcTargetVertices
                    ? llcTargetVertices - llcUsed
                    : 0;
            seed = peekFrontierFitting(
                llcQueue, llcScore, llcRemaining);
            if (seed == static_cast<K>(-1))
                seed = nextLargestFitting(llcRemaining);
            if (
                seed == static_cast<K>(-1)
                || llcUsed >= llcTargetVertices
                || communitySizes[seed]
                    > llcTargetVertices - llcUsed
            ) break;
        }
        result.llcRunEnds.push_back(result.order.size());
    }

    result.order.insert(
        result.order.end(),
        emptyCommunities.begin(),
        emptyCommunities.end());
    return result;
}

}  // namespace graphbrew::experimental

#endif  // GRAPHBREW_REORDER_EXPERIMENTAL_CAPACITY_RUNS_H_
