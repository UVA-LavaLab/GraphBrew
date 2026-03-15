// ============================================================================
// DROPLET Indirect Graph Prefetcher for gem5 — Implementation
// ============================================================================
// Reference: Basak et al., HPCA 2019 + research/caching/droplet.md
// ============================================================================

#include "mem/cache/prefetch/droplet.hh"

#include <algorithm>
#include <cassert>

namespace gem5 {
namespace prefetch {

GraphDropletPrefetcher::GraphDropletPrefetcher(const Params &p)
    : Queued(p),
      prefetchDegree(p.prefetch_degree),
      indirectDegree(p.indirect_degree),
      strideTableSize(p.stride_table_size)
{
    strideTable.resize(strideTableSize);
}

void
GraphDropletPrefetcher::calculatePrefetch(
    const PrefetchInfo &pfi,
    std::vector<AddrPriority> &addresses,
    const CacheAccessor &cache)
{
    if (!pfi.hasPC()) return;

    Addr addr = pfi.getAddr();
    bool isMiss = pfi.isCacheMiss();

    // ── Engine 1: Edge-list stride prefetcher ──
    // Detect sequential CSR edge array traversal and prefetch ahead
    if (isEdgeArrayAccess(addr)) {
        std::vector<uint64_t> stridePredictions;
        updateStrideDetector(addr, stridePredictions);

        for (const auto& predAddr : stridePredictions) {
            if (isEdgeArrayAccess(predAddr) && shouldPrefetch(predAddr)) {
                addresses.push_back(AddrPriority(predAddr, 0));
            }
        }

        // ── Engine 2: Indirect property prefetcher ──
        // From the current edge data, extract neighbor IDs and prefetch
        // their property data. This is the key DROPLET innovation:
        // the indirect chain edge_list[i] → neighbor_id → property[neighbor_id]
        if (isMiss) {
            issueIndirectPrefetches(addr, addresses);
        }
    }

    // Property data misses: no action needed (covered by indirect prefetch)
    // Index array accesses: very short reuse, fits in L1 (no prefetch needed)
}

void
GraphDropletPrefetcher::updateStrideDetector(
    uint64_t addr, std::vector<uint64_t>& predictions)
{
    // Simple stride detector: track last N accesses to edge array,
    // detect stride pattern, predict next addresses

    // Find matching entry or create new one
    StrideEntry* bestEntry = nullptr;
    for (auto& entry : strideTable) {
        if (!entry.valid) {
            if (!bestEntry) bestEntry = &entry;
            continue;
        }
        int64_t delta = static_cast<int64_t>(addr) -
                        static_cast<int64_t>(entry.lastAddr);
        if (delta == entry.stride && delta != 0) {
            entry.confidence = std::min(uint8_t(3), uint8_t(entry.confidence + 1));
            entry.lastAddr = addr;
            bestEntry = &entry;
            break;
        } else if (std::abs(delta) <= 512 && delta != 0) {
            // New stride detected
            entry.stride = delta;
            entry.confidence = 1;
            entry.lastAddr = addr;
            bestEntry = &entry;
            break;
        }
    }

    if (!bestEntry) {
        // Evict oldest entry (front of deque)
        bestEntry = &strideTable.front();
    }

    if (!bestEntry->valid) {
        bestEntry->lastAddr = addr;
        bestEntry->stride = 64; // Default: one cache line ahead
        bestEntry->confidence = 0;
        bestEntry->valid = true;
        return;
    }

    // Generate predictions if confidence is high enough
    if (bestEntry->confidence >= 2 && bestEntry->stride != 0) {
        for (int d = 1; d <= prefetchDegree; d++) {
            uint64_t predAddr = addr +
                static_cast<uint64_t>(bestEntry->stride * d);
            predictions.push_back(predAddr);
        }
    }
}

void
GraphDropletPrefetcher::issueIndirectPrefetches(
    uint64_t edgeAddr, std::vector<AddrPriority>& addresses)
{
    if (!propertyConfigured) return;

    // In a real hardware implementation, DROPLET would read the prefetched
    // edge data to extract neighbor vertex IDs. In gem5 simulation, we
    // approximate this by computing likely neighbor IDs from the edge
    // address position in the CSR array.
    //
    // The edge array stores uint32_t vertex IDs (4 bytes each).
    // From the edge address, we can estimate which edges are ahead
    // and prefetch properties for vertices at those positions.

    uint32_t elementsPerLine = 64 / sizeof(uint32_t); // 16 vertex IDs per line
    uint64_t baseLineAddr = edgeAddr & ~uint64_t(63);

    for (int i = 0; i < indirectDegree; i++) {
        // Prefetch property for vertices stored at positions ahead
        // We estimate vertex ID from position (this is approximate;
        // in real HW, the prefetcher reads the actual edge data)
        uint64_t targetEdgeAddr = baseLineAddr +
            static_cast<uint64_t>((i + 1) * sizeof(uint32_t));

        if (!isEdgeArrayAccess(targetEdgeAddr)) break;

        // Compute vertex ID from edge array offset
        uint32_t edgeOffset = static_cast<uint32_t>(
            (targetEdgeAddr - edgeArrayBase) / sizeof(uint32_t));

        // Issue property prefetch for this vertex
        uint64_t propAddr = propertyAddrForVertex(edgeOffset);
        if (isPropertyAccess(propAddr) && shouldPrefetch(propAddr)) {
            // Lower priority than edge prefetches (indirect chain)
            addresses.push_back(AddrPriority(propAddr, -1));
        }
    }
}

bool
GraphDropletPrefetcher::shouldPrefetch(uint64_t addr)
{
    uint64_t lineAddr = addr & ~uint64_t(63);

    if (recentPrefetches.count(lineAddr)) return false;

    recentPrefetches.insert(lineAddr);
    if (recentPrefetches.size() > MAX_RECENT) {
        // Simple eviction: clear half the set
        auto it = recentPrefetches.begin();
        size_t toRemove = recentPrefetches.size() / 2;
        while (toRemove > 0 && it != recentPrefetches.end()) {
            it = recentPrefetches.erase(it);
            toRemove--;
        }
    }
    return true;
}

} // namespace prefetch
} // namespace gem5
