#include "graph_cache_context_sniper.h"
#include "ecg_victim_policy.h"  // SSOT: shared GRASP insertion-tier classifier

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>

namespace graphbrew {
namespace sniper {

namespace {

// Tier A sideband-registration sanity log.  Mirrors the cache_sim / gem5
// variants. Suppress with GRAPHBREW_SIDEBAND_LOG=0.
bool graphCtxRegistrationLogEnabled()
{
    static int enabled = []() {
        const char* value = std::getenv("GRAPHBREW_SIDEBAND_LOG");
        if (!value || !value[0]) return 1;
        return (std::strcmp(value, "0") == 0) ? 0 : 1;
    }();
    return enabled != 0;
}

void logGraphCtxRegistration(const char* source,
                             const char* name,
                             uint64_t base,
                             uint64_t upper,
                             uint32_t hot_pct,
                             bool grasp_region)
{
    if (!graphCtxRegistrationLogEnabled()) return;
    std::fprintf(stderr,
                 "[graphctx] register region source=%s name=%s base=0x%lx "
                 "upper=0x%lx hot_pct=%u grasp_region=%d\n",
                 source ? source : "?",
                 (name && name[0]) ? name : "(unnamed)",
                 static_cast<unsigned long>(base),
                 static_cast<unsigned long>(upper),
                 hot_pct,
                 grasp_region ? 1 : 0);
}

std::array<std::atomic<uint32_t>, MAX_TRACKED_CORES>& vertexStorage()
{
    static std::array<std::atomic<uint32_t>, MAX_TRACKED_CORES> storage{};
    return storage;
}

std::array<std::atomic<bool>, MAX_TRACKED_CORES>& vertexValidStorage()
{
    static std::array<std::atomic<bool>, MAX_TRACKED_CORES> storage{};
    return storage;
}

std::array<std::atomic<uint32_t>, MAX_TRACKED_CORES>& prefetchTargetStorage()
{
    static std::array<std::atomic<uint32_t>, MAX_TRACKED_CORES> storage{};
    return storage;
}

std::array<std::atomic<bool>, MAX_TRACKED_CORES>& prefetchTargetValidStorage()
{
    static std::array<std::atomic<bool>, MAX_TRACKED_CORES> storage{};
    return storage;
}

uint32_t clampVertex(uint64_t vertex)
{
    return vertex > std::numeric_limits<uint32_t>::max()
        ? std::numeric_limits<uint32_t>::max()
        : static_cast<uint32_t>(vertex);
}

uint64_t parseJsonUint(const std::string& json, const std::string& key)
{
    size_t pos = json.find(key);
    if (pos == std::string::npos) return 0;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return 0;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    return std::strtoull(json.c_str() + pos, nullptr, 10);
}

std::string parseJsonString(const std::string& json, const std::string& key)
{
    size_t pos = json.find(key);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    size_t end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
}

bool parseJsonBool(const std::string& json, const std::string& key)
{
    size_t pos = json.find(key);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    return json.compare(pos, 4, "true") == 0;
}

}  // namespace

void setCurrentVertexHint(uint32_t core_id, uint64_t vertex)
{
    if (core_id >= MAX_TRACKED_CORES) return;
    vertexStorage()[core_id].store(clampVertex(vertex), std::memory_order_release);
    vertexValidStorage()[core_id].store(true, std::memory_order_release);
}

bool hasCurrentVertexHint(uint32_t core_id)
{
    return core_id < MAX_TRACKED_CORES &&
        vertexValidStorage()[core_id].load(std::memory_order_acquire);
}

bool hasAnyCurrentVertexHint()
{
    for (uint32_t core_id = 0; core_id < MAX_TRACKED_CORES; ++core_id) {
        if (vertexValidStorage()[core_id].load(std::memory_order_acquire))
            return true;
    }
    return false;
}

uint32_t getCurrentVertexHint(uint32_t core_id)
{
    if (core_id >= MAX_TRACKED_CORES) return 0;
    return vertexStorage()[core_id].load(std::memory_order_acquire);
}

void clearCurrentVertexHint(uint32_t core_id)
{
    if (core_id >= MAX_TRACKED_CORES) return;
    vertexValidStorage()[core_id].store(false, std::memory_order_release);
}

static uint32_t& currentNucaRequesterCoreStorage()
{
    static thread_local uint32_t requester_core = UINT32_MAX;
    return requester_core;
}

void setCurrentNucaRequesterCore(uint32_t core_id)
{
    currentNucaRequesterCoreStorage() = core_id;
}

uint32_t currentNucaRequesterCore()
{
    return currentNucaRequesterCoreStorage();
}

// === Per-core prefetch-target hint ring buffer (sprint 6f-6 fix) ===
//
// Previously used a single atomic<uint32_t> mailbox per core. Kernel
// emits thousands of hints per PR iteration; the L2 prefetcher only
// runs getNextAddress() on cache notification events. With a single
// slot, each new kernel hint OVERWROTE the prior unconsumed hint —
// ~99% of hints were lost on email-Eu-core (38 issued of ~2360
// emitted). Ring buffer of N entries lets the kernel queue up to N
// hints between prefetcher invocations.

static constexpr std::size_t kHintQueueSize = 256;

struct PerCoreHintQueue {
    std::array<std::atomic<uint32_t>, kHintQueueSize> entries{};
    std::atomic<std::size_t> head{0};
    std::atomic<std::size_t> tail{0};
};

std::array<PerCoreHintQueue, MAX_TRACKED_CORES>& prefetchTargetHintQueues()
{
    static std::array<PerCoreHintQueue, MAX_TRACKED_CORES> queues;
    return queues;
}

void setPrefetchTargetHint(uint32_t core_id, uint64_t vertex)
{
    if (core_id >= MAX_TRACKED_CORES) return;
    auto& q = prefetchTargetHintQueues()[core_id];
    std::size_t t = q.tail.load(std::memory_order_relaxed);
    std::size_t next = (t + 1) % kHintQueueSize;
    if (next == q.head.load(std::memory_order_acquire)) {
        // Queue full: drop oldest by advancing head one slot.
        q.head.store((q.head.load(std::memory_order_relaxed) + 1) % kHintQueueSize,
                     std::memory_order_release);
    }
    q.entries[t].store(clampVertex(vertex), std::memory_order_relaxed);
    q.tail.store(next, std::memory_order_release);
    // Keep the legacy "valid" flag alive for has/getPrefetchTargetHint
    // callers that may exist outside the consume path. It now means
    // "queue is non-empty."
    prefetchTargetValidStorage()[core_id].store(true, std::memory_order_release);
}

bool hasPrefetchTargetHint(uint32_t core_id)
{
    if (core_id >= MAX_TRACKED_CORES) return false;
    auto& q = prefetchTargetHintQueues()[core_id];
    return q.head.load(std::memory_order_acquire) !=
           q.tail.load(std::memory_order_acquire);
}

uint32_t getPrefetchTargetHint(uint32_t core_id)
{
    // Returns the oldest entry without consuming it. Used for
    // diagnostics; callers wanting to consume should call
    // consumePrefetchTargetHint instead.
    if (core_id >= MAX_TRACKED_CORES) return 0;
    auto& q = prefetchTargetHintQueues()[core_id];
    std::size_t h = q.head.load(std::memory_order_acquire);
    if (h == q.tail.load(std::memory_order_acquire)) return 0;
    return q.entries[h].load(std::memory_order_acquire);
}

bool consumePrefetchTargetHint(uint32_t core_id, uint32_t& vertex)
{
    if (core_id >= MAX_TRACKED_CORES) return false;
    auto& q = prefetchTargetHintQueues()[core_id];
    std::size_t h = q.head.load(std::memory_order_relaxed);
    if (h == q.tail.load(std::memory_order_acquire)) {
        prefetchTargetValidStorage()[core_id].store(false, std::memory_order_release);
        return false;
    }
    vertex = q.entries[h].load(std::memory_order_relaxed);
    std::size_t next = (h + 1) % kHintQueueSize;
    q.head.store(next, std::memory_order_release);
    if (next == q.tail.load(std::memory_order_acquire)) {
        prefetchTargetValidStorage()[core_id].store(false, std::memory_order_release);
    }
    return true;
}

void clearPrefetchTargetHint(uint32_t core_id)
{
    if (core_id >= MAX_TRACKED_CORES) return;
    auto& q = prefetchTargetHintQueues()[core_id];
    q.head.store(0, std::memory_order_release);
    q.tail.store(0, std::memory_order_release);
    prefetchTargetValidStorage()[core_id].store(false, std::memory_order_release);
}

// === SNIPER_ECG_EXTRACT per-core epoch map (direct-mapped by cache line) ===
// The builder delivers line-min epochs, so keying the bounded map by line avoids
// a stale fallback to another vertex after a direct-mapped collision.
static constexpr std::size_t kEcgEpochMapSize = 8192;

struct PerCoreEpochMap {
    std::array<std::atomic<uint32_t>, kEcgEpochMapSize> line_plus1{};
    std::array<std::atomic<uint16_t>, kEcgEpochMapSize> epoch{};
    std::array<std::atomic<uint16_t>, kEcgEpochMapSize> epoch2{};
    std::array<std::atomic<uint8_t>, kEcgEpochMapSize> count{};
    // Seqlock word: (global_delivery_sequence << 1) | write_in_progress.
    std::array<std::atomic<uint64_t>, kEcgEpochMapSize> version{};
};

static std::array<PerCoreEpochMap, MAX_TRACKED_CORES>& ecgEpochMaps()
{
    static std::array<PerCoreEpochMap, MAX_TRACKED_CORES> maps;
    return maps;
}

static std::atomic<uint64_t>& ecgEpochGlobalSequence()
{
    static std::atomic<uint64_t> sequence{0};
    return sequence;
}

static uint32_t ecgVerticesPerLine()
{
    static const uint32_t value = []() {
        const char* raw = std::getenv("SNIPER_ECG_VERTICES_PER_LINE");
        int parsed = raw ? std::atoi(raw) : 16;
        if (parsed < 1) parsed = 1;
        if (parsed > 1024) parsed = 1024;
        return static_cast<uint32_t>(parsed);
    }();
    return value;
}

void recordEcgEpoch(uint32_t core_id, uint32_t vertex, uint16_t epoch)
{
    if (core_id >= MAX_TRACKED_CORES) return;
    auto& m = ecgEpochMaps()[core_id];
    const uint32_t line = vertex / ecgVerticesPerLine();
    std::size_t i = line % kEcgEpochMapSize;
    uint64_t sequence =
        ecgEpochGlobalSequence().fetch_add(1, std::memory_order_relaxed) + 1;
    m.version[i].exchange(
        (sequence << 1) | 1u, std::memory_order_acq_rel);
    m.epoch[i].store(epoch, std::memory_order_relaxed);
    m.epoch2[i].store(epoch, std::memory_order_relaxed);
    m.count[i].store(1, std::memory_order_relaxed);
    m.line_plus1[i].store(line + 1u, std::memory_order_relaxed);
    m.version[i].store(sequence << 1, std::memory_order_release);
}

void recordEcgEpochPair(uint32_t core_id, uint32_t vertex,
                        uint16_t first, uint16_t second)
{
    if (core_id >= MAX_TRACKED_CORES) return;
    static std::atomic<uint64_t> trace_sequence{0};
    static const uint64_t trace_limit = []() {
        const char* value = std::getenv("ECG_K2_DELIVERY_TRACE");
        return value ? static_cast<uint64_t>(std::strtoull(value, nullptr, 10)) : 0;
    }();
    const uint64_t sequence_index =
        trace_sequence.fetch_add(1, std::memory_order_relaxed);
    if (sequence_index < trace_limit) {
        std::fprintf(stderr,
            "[ECG-K2-RECV sim=sniper seq=%llu dest=%u epoch1=%u epoch2=%u]\n",
            (unsigned long long)sequence_index, vertex,
            static_cast<unsigned>(first), static_cast<unsigned>(second));
    }
    static std::atomic<uint32_t> debug_count{0};
    static std::atomic<uint32_t> debug_nonzero_count{0};
    const char* debug = std::getenv("ECG_DEBUG");
    uint32_t debug_index = debug_count.fetch_add(1, std::memory_order_relaxed);
    uint32_t debug_nonzero_index = (first != 0 || second != 0)
        ? debug_nonzero_count.fetch_add(1, std::memory_order_relaxed)
        : UINT32_MAX;
    if (debug && debug[0] && std::strcmp(debug, "0") != 0 &&
        (debug_index < 4 ||
         ((first != 0 || second != 0) && debug_nonzero_index < 4))) {
        std::fprintf(stderr,
                     "[ECG-DELIVER2 sim=sniper core=%u vertex=%u "
                     "epoch1=%u epoch2=%u]\n",
                     core_id, vertex, static_cast<unsigned>(first),
                     static_cast<unsigned>(second));
    }
    auto& m = ecgEpochMaps()[core_id];
    const uint32_t line = vertex / ecgVerticesPerLine();
    std::size_t i = line % kEcgEpochMapSize;
    uint64_t sequence =
        ecgEpochGlobalSequence().fetch_add(1, std::memory_order_relaxed) + 1;
    m.version[i].exchange(
        (sequence << 1) | 1u, std::memory_order_acq_rel);
    m.epoch[i].store(first, std::memory_order_relaxed);
    m.epoch2[i].store(second, std::memory_order_relaxed);
    m.count[i].store(2, std::memory_order_relaxed);
    m.line_plus1[i].store(line + 1u, std::memory_order_relaxed);
    m.version[i].store(sequence << 1, std::memory_order_release);
}

bool lookupEcgEpoch(uint32_t core_id, uint32_t vertex,
                    uint16_t& epoch, uint64_t& sequence)
{
    uint16_t second = 0;
    uint8_t count = 0;
    return lookupEcgEpochPair(
        core_id, vertex, epoch, second, count, sequence);
}

bool lookupEcgEpochPair(uint32_t core_id, uint32_t vertex,
                        uint16_t& first, uint16_t& second,
                        uint8_t& count, uint64_t& sequence)
{
    if (core_id >= MAX_TRACKED_CORES) return false;
    auto& m = ecgEpochMaps()[core_id];
    const uint32_t line = vertex / ecgVerticesPerLine();
    std::size_t i = line % kEcgEpochMapSize;
    for (unsigned attempt = 0; attempt < 4; ++attempt) {
        uint64_t before = m.version[i].load(std::memory_order_acquire);
        if (before == 0 || (before & 1u)) continue;
        uint32_t stored_line =
            m.line_plus1[i].load(std::memory_order_relaxed);
        uint16_t stored_first = m.epoch[i].load(std::memory_order_relaxed);
        uint16_t stored_second = m.epoch2[i].load(std::memory_order_relaxed);
        uint8_t stored_count = m.count[i].load(std::memory_order_relaxed);
        uint64_t after = m.version[i].load(std::memory_order_acquire);
        if (before != after || (after & 1u)) continue;
        if (stored_line != line + 1u) return false;
        first = stored_first;
        second = stored_second;
        count = stored_count;
        sequence = after >> 1;
        return true;
    }
    return false;
}

ECGMode stringToECGMode(const std::string& text)
{
    if (text == "POPT_PRIMARY" || text == "popt_primary" || text == "popt") return ECGMode::POPT_PRIMARY;
    if (text == "DBG_ONLY" || text == "dbg_only" || text == "dbg") return ECGMode::DBG_ONLY;
    if (text == "ECG_EMBEDDED" || text == "ecg_embedded" || text == "embedded") return ECGMode::ECG_EMBEDDED;
    if (text == "ECG_COMBINED" || text == "ecg_combined" || text == "combined") return ECGMode::ECG_COMBINED;
    if (text == "ECG_GRASP_POPT" || text == "ecg_grasp_popt" || text == "grasp_popt") return ECGMode::ECG_GRASP_POPT;
    if (text.empty() || text == "DBG_PRIMARY" || text == "dbg_primary") return ECGMode::DBG_PRIMARY;
    // Fail fast instead of silently aliasing an unknown/typo'd mode to
    // DBG_PRIMARY (which would mislabel result rows). POPT_TIE and
    // ECG_EPOCH_EMBEDDED are cache_sim-only experimental modes.
    std::fprintf(stderr,
        "[graphctx] FATAL: unsupported ECG mode '%s' for Sniper. Supported: "
        "DBG_PRIMARY, POPT_PRIMARY, DBG_ONLY, ECG_EMBEDDED, ECG_COMBINED, "
        "ECG_GRASP_POPT (POPT_TIE / ECG_EPOCH_EMBEDDED are cache_sim-only).\n", text.c_str());
    std::abort();
}

std::string ecgModeToString(ECGMode mode)
{
    switch (mode) {
      case ECGMode::DBG_PRIMARY:  return "DBG_PRIMARY";
      case ECGMode::POPT_PRIMARY: return "POPT_PRIMARY";
      case ECGMode::DBG_ONLY:     return "DBG_ONLY";
      case ECGMode::ECG_EMBEDDED: return "ECG_EMBEDDED";
      case ECGMode::ECG_COMBINED: return "ECG_COMBINED";
      case ECGMode::ECG_GRASP_POPT: return "ECG_GRASP_POPT";
      default:                    return "UNKNOWN";
    }
}

bool PropertyRegion::contains(uint64_t addr) const
{
    return addr >= base_address && addr < upper_bound;
}

uint32_t PropertyRegion::classifyBucket(uint64_t addr) const
{
    if (!contains(addr) || num_buckets == 0) return num_buckets;
    for (uint32_t bucket = 0; bucket < num_buckets; ++bucket) {
        if (addr < bucket_bounds[bucket]) return bucket;
    }
    return num_buckets - 1;
}

bool EdgeRegion::contains(uint64_t addr) const
{
    return addr >= base_address && addr < upper_bound;
}

bool RereferenceMatrix::loadFromFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    file.read(reinterpret_cast<char*>(&num_epochs), sizeof(num_epochs));
    file.read(reinterpret_cast<char*>(&num_cache_lines), sizeof(num_cache_lines));
    file.read(reinterpret_cast<char*>(&epoch_size), sizeof(epoch_size));
    file.read(reinterpret_cast<char*>(&sub_epoch_size), sizeof(sub_epoch_size));
    if (!file || num_epochs == 0 || num_cache_lines == 0) {
        enabled = false;
        return false;
    }

    size_t matrix_size = static_cast<size_t>(num_epochs) * num_cache_lines;
    if (num_cache_lines != 0 && matrix_size / num_cache_lines != num_epochs) {
        enabled = false;
        return false;
    }
    data.assign(matrix_size, 0);
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(matrix_size));
    enabled = static_cast<size_t>(file.gcount()) == matrix_size;
    return enabled;
}

uint32_t RereferenceMatrix::findNextRef(uint32_t cline_id, uint32_t current_vertex) const
{
    if (!enabled || cline_id >= num_cache_lines) return 127;
    uint32_t epoch_id = epoch_size > 0 ? current_vertex / epoch_size : 0;
    if (epoch_id >= num_epochs) return 127;

    uint8_t entry = data[static_cast<size_t>(epoch_id) * num_cache_lines + cline_id];
    constexpr uint8_t OR_MASK = 0x80;
    constexpr uint8_t AND_MASK = 0x7F;

    if ((entry & OR_MASK) != 0) {
        return entry & AND_MASK;
    } else {
        uint8_t last_ref_sub_epoch = entry & AND_MASK;
        uint32_t current_sub_epoch = sub_epoch_size > 0
            ? ((current_vertex % epoch_size) / sub_epoch_size)
            : 0;
        if (current_sub_epoch <= last_ref_sub_epoch) return 0;
        if (epoch_id + 1 < num_epochs) {
            uint8_t next_entry = data[static_cast<size_t>(epoch_id + 1) * num_cache_lines + cline_id];
            if ((next_entry & OR_MASK) == 0) return 1;
            uint8_t reref = next_entry & AND_MASK;
            return reref < 127 ? reref + 1 : 127;
        }
        return 127;
    }
}

uint32_t RereferenceMatrix::findNextRefByAddr(uint64_t addr, uint32_t current_vertex) const
{
    if (!enabled || addr < base_address) return 127;
    uint32_t cline_id = static_cast<uint32_t>((addr - base_address) / cache_line_size);
    return findNextRef(cline_id, current_vertex);
}

void MaskConfig::computeShifts()
{
    prefetch_shift = 0;
    popt_shift = prefetch_bits;
    dbg_shift = prefetch_bits + popt_bits;
    prefetch_mask_val = prefetch_bits ? ((1U << prefetch_bits) - 1) : 0;
    popt_mask_val = popt_bits ? (((1U << popt_bits) - 1) << popt_shift) : 0;
    dbg_mask_val = dbg_bits ? (((1U << dbg_bits) - 1) << dbg_shift) : 0;
}

uint8_t MaskConfig::decodeDBG(uint32_t mask_entry) const
{
    return dbg_bits ? static_cast<uint8_t>((mask_entry & dbg_mask_val) >> dbg_shift) : 0;
}

uint8_t MaskConfig::decodePOPT(uint32_t mask_entry) const
{
    return popt_bits ? static_cast<uint8_t>((mask_entry & popt_mask_val) >> popt_shift) : 0;
}

uint8_t MaskConfig::dbgTierToRRPV(uint8_t dbg_tier) const
{
    float fraction = static_cast<float>(dbg_tier) / std::max<uint8_t>(1, num_buckets);
    uint8_t result = static_cast<uint8_t>(rrpv_max * fraction);
    if (result > rrpv_max) result = rrpv_max;
    if (result == 0 && fraction > 0.0f) result = 1;
    return result;
}

bool GraphCacheContext::loadFromSideband(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open()) return false;
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    topology.num_vertices = static_cast<uint32_t>(parseJsonUint(content, "\"num_vertices\""));
    topology.num_edges = parseJsonUint(content, "\"num_edges\"");
    stream_bypass_base = parseJsonUint(content, "\"stream_bypass_base\"");
    const uint64_t stream_bypass_size =
        parseJsonUint(content, "\"stream_bypass_size\"");
    stream_bypass_upper = stream_bypass_base + stream_bypass_size;
    topology.max_degree = static_cast<uint32_t>(parseJsonUint(content, "\"max_degree\""));
    topology.avg_degree = topology.num_vertices > 0
        ? static_cast<double>(topology.num_edges) / topology.num_vertices
        : 0.0;
    topology.enabled = topology.num_vertices > 0;

    // ne for SNIPER_ECG_EXTRACT circular distance — same source as the kernel's
    // ECG_EDGE_MASK_EPOCHS packing.
    if (const char* ne_env = std::getenv("ECG_EDGE_MASK_EPOCHS")) {
        uint32_t ne = static_cast<uint32_t>(std::strtoul(ne_env, nullptr, 10));
        if (ne < 2) ne = 2;
        if (ne > 65535) ne = 65535;  // match the kernel clamp + the 16-bit epoch field
        edge_epoch_count = ne;
    }

    num_regions = 0;
    size_t pos = content.find("\"property_regions\"");
    if (pos != std::string::npos) {
        size_t arr_start = content.find('[', pos);
        size_t arr_end = content.find(']', arr_start);
        if (arr_start != std::string::npos && arr_end != std::string::npos) {
            std::string array_text = content.substr(arr_start, arr_end - arr_start + 1);
            size_t obj_pos = 0;
            while ((obj_pos = array_text.find('{', obj_pos)) != std::string::npos &&
                   num_regions < MAX_PROPERTY_REGIONS) {
                size_t obj_end = array_text.find('}', obj_pos);
                if (obj_end == std::string::npos) break;
                std::string obj = array_text.substr(obj_pos, obj_end - obj_pos + 1);

                PropertyRegion& region = regions[num_regions];
                region.name = parseJsonString(obj, "\"name\"");
                region.base_address = parseJsonUint(obj, "\"base\"");
                uint64_t size = parseJsonUint(obj, "\"size\"");
                region.upper_bound = region.base_address + size;
                region.num_elements = static_cast<uint32_t>(parseJsonUint(obj, "\"count\""));
                region.elem_size = static_cast<uint32_t>(parseJsonUint(obj, "\"elem_size\""));
                region.region_id = num_regions;
                region.grasp_region = obj.find("\"grasp\"") == std::string::npos ||
                    parseJsonBool(obj, "\"grasp\"");
                region.num_buckets = 3;

                uint64_t third = ((size / 3) + rereference.cache_line_size - 1) & ~(rereference.cache_line_size - 1);
                region.bucket_bounds[0] = region.base_address + third;
                region.bucket_bounds[1] = region.base_address + 2 * third;
                region.bucket_bounds[2] = region.upper_bound;

                // GRASP hot region = frontier_frac as % of the VERTEX SPACE
                // (array-relative, GRASP-faithful). Actual classification reads
                // GRASP_HOT_FRACTION (default 0.15) in classifyGRASP(); this is
                // just the logged registration value.
                constexpr uint32_t kSidebandHotPct = 15;
                logGraphCtxRegistration("sniper", region.name.c_str(),
                                        region.base_address,
                                        region.upper_bound,
                                        kSidebandHotPct,
                                        region.grasp_region);

                num_regions++;
                obj_pos = obj_end + 1;
            }
        }
    }

    num_edge_regions = 0;
    pos = content.find("\"edge_regions\"");
    if (pos != std::string::npos) {
        size_t arr_start = content.find('[', pos);
        size_t arr_end = content.find(']', arr_start);
        if (arr_start != std::string::npos && arr_end != std::string::npos) {
            std::string array_text = content.substr(arr_start, arr_end - arr_start + 1);
            size_t obj_pos = 0;
            while ((obj_pos = array_text.find('{', obj_pos)) != std::string::npos &&
                   num_edge_regions < edge_regions.size()) {
                size_t obj_end = array_text.find('}', obj_pos);
                if (obj_end == std::string::npos) break;
                std::string obj = array_text.substr(obj_pos, obj_end - obj_pos + 1);
                EdgeRegion& region = edge_regions[num_edge_regions];
                region.name = parseJsonString(obj, "\"name\"");
                region.base_address = parseJsonUint(obj, "\"base\"");
                uint64_t size = parseJsonUint(obj, "\"size\"");
                region.upper_bound = region.base_address + size;
                region.elem_size = static_cast<uint32_t>(parseJsonUint(obj, "\"elem_size\""));
                region.preferred = parseJsonBool(obj, "\"preferred\"");
                region.data_path = parseJsonString(obj, "\"data_path\"");
                num_edge_regions++;
                obj_pos = obj_end + 1;
            }
        }
    }

    loaded = num_regions > 0;
    mask_config.computeShifts();
    return loaded;
}

bool GraphCacheContext::loadRereferenceMatrix(const std::string& path)
{
    return rereference.loadFromFile(path);
}

void GraphCacheContext::setCacheLineSize(uint64_t line_size)
{
    if (line_size == 0) return;
    rereference.cache_line_size = line_size;
}

uint32_t GraphCacheContext::currentVertexForPopt(uint32_t core_id) const
{
    if (hasCurrentVertexHint(core_id)) {
        uint32_t vertex = getCurrentVertexHint(core_id);
        current_dst_vertex = vertex;
        current_outer_vertex = vertex;
        return vertex;
    }
    return current_dst_vertex;
}

void GraphCacheContext::updateVertexFromAddr(uint64_t addr, uint32_t core_id) const
{
    if (num_regions == 0 || !regions[0].contains(addr) || regions[0].elem_size == 0) return;
    uint32_t vertex = static_cast<uint32_t>((addr - regions[0].base_address) / regions[0].elem_size);
    current_dst_vertex = vertex;
    if (!hasCurrentVertexHint(core_id)) current_outer_vertex = vertex;
}

uint32_t GraphCacheContext::vertexForAddress(uint64_t addr) const
{
    // Search ALL property regions (e.g. scores AND contrib) — the eviction-
    // protected array is region[1] (contrib) for PR, so a region[0]-only check
    // would silently never stamp it. Mirrors isPropertyData/findNextRef.
    for (uint32_t i = 0; i < num_regions; ++i) {
        if (regions[i].elem_size != 0 && regions[i].contains(addr))
            return static_cast<uint32_t>((addr - regions[i].base_address) / regions[i].elem_size);
    }
    return UINT32_MAX;
}

// elem_size of the property region owning addr (0 if none); used to size the
// per-line vertex scan in lookupLineEcgEpoch.
uint32_t GraphCacheContext::propertyElemSizeForAddress(uint64_t addr) const
{
    for (uint32_t i = 0; i < num_regions; ++i) {
        if (regions[i].elem_size != 0 && regions[i].contains(addr))
            return regions[i].elem_size;
    }
    return 0;
}

bool GraphCacheContext::isPropertyData(uint64_t addr) const
{
    for (uint32_t i = 0; i < num_regions; ++i) {
        if (regions[i].contains(addr)) return true;
    }
    return false;
}

bool GraphCacheContext::isEcgEpochData(uint64_t addr) const
{
    const char* requested = std::getenv("SNIPER_ECG_EPOCH_REGION");
    if (requested && requested[0]) {
        for (uint32_t i = 0; i < num_regions; ++i) {
            if (regions[i].name == requested)
                return regions[i].contains(addr);
        }
        return false;
    }
    if (num_regions == 1) return regions[0].contains(addr);
    static const char* defaults[] = {
        "contrib", "parent", "dist", "depth", "comp"
    };
    for (const char* name : defaults) {
        for (uint32_t i = 0; i < num_regions; ++i) {
            if (regions[i].name == name)
                return regions[i].contains(addr);
        }
    }
    return false;
}

bool GraphCacheContext::isStreamBypassData(uint64_t addr) const
{
    return stream_bypass_base < stream_bypass_upper &&
           addr >= stream_bypass_base && addr < stream_bypass_upper;
}

bool GraphCacheContext::isEdgeData(uint64_t addr) const
{
    for (uint32_t i = 0; i < num_edge_regions; ++i) {
        if (edge_regions[i].contains(addr)) return true;
    }
    return false;
}

uint32_t GraphCacheContext::classifyBucket(uint64_t addr) const
{
    for (uint32_t i = 0; i < num_regions; ++i) {
        if (regions[i].contains(addr)) return regions[i].classifyBucket(addr);
    }
    return mask_config.num_buckets;
}

uint32_t GraphCacheContext::findNextRef(uint64_t addr, uint32_t core_id) const
{
    if (!rereference.enabled) return 127;
    for (uint32_t i = 0; i < num_regions; ++i) {
        if (regions[i].contains(addr)) {
            uint32_t cline_id = static_cast<uint32_t>(
                (addr - regions[i].base_address) / rereference.cache_line_size);
            return rereference.findNextRef(cline_id, currentVertexForPopt(core_id));
        }
    }
    return 127;
}

uint32_t GraphCacheContext::classifyGRASP(uint64_t addr, uint64_t llc_size) const
{
    // GRASP-faithful (ligra.h add_region): the hot region is a fraction of the
    // VERTEX SPACE (frontier_frac x n) = a fraction of the property ARRAY, not of
    // the LLC. Auto-scales with graph size. Default ~0.15 (~Faldu's vertex-relative
    // "10%"). Override via GRASP_HOT_FRACTION (0<f<=1) for sensitivity sweeps.
    static const double hot_fraction = [](){
        const char* e = std::getenv("GRASP_HOT_FRACTION");
        double v = e ? std::atof(e) : 0.15;
        return (v > 0.0 && v <= 1.0) ? v : 0.15;
    }();
    (void)llc_size;
    for (uint32_t i = 0; i < num_regions; ++i) {
        if (!regions[i].grasp_region) continue;
        // SSOT: per-region tier math shared with cache_sim + gem5.
        uint32_t tier = ecg_policy::classifyGraspTier(
            addr, regions[i].base_address, regions[i].upper_bound, hot_fraction);
        if (tier != 0) return tier;
    }
    return 3;
}

uint8_t GraphCacheContext::getInsertRRPV(uint64_t addr) const
{
    uint32_t bucket = classifyBucket(addr);
    if (bucket >= mask_config.num_buckets) return mask_config.rrpv_max;
    return mask_config.dbgTierToRRPV(static_cast<uint8_t>(bucket));
}

GraphCacheContext& globalContext()
{
    static GraphCacheContext context;
    return context;
}

bool isEcgStreamBypassAddress(uint64_t addr)
{
    const char* enabled = std::getenv("ECG_STREAM_BYPASS");
    if (!enabled || std::strcmp(enabled, "0") == 0) return false;
    GraphCacheContext& context = globalContext();
    if (!context.loaded ||
        context.stream_bypass_base >= context.stream_bypass_upper) {
        const char* path = std::getenv("SNIPER_GRAPHBREW_CTX");
        if (!path || !path[0]) path = "/tmp/sniper_graphbrew_ctx.json";
        context.loaded = context.loadFromSideband(path);
    }
    const bool match = context.loaded && context.isStreamBypassData(addr);
    static uint64_t probes = 0;
    static const uint64_t limit = []() {
        const char* value = std::getenv("ECG_STREAM_BYPASS_TRACE");
        return value ? std::strtoull(value, nullptr, 10) : 0;
    }();
    if (probes++ < limit) {
        std::fprintf(stderr,
            "[ECG-STREAM-PROBE sim=sniper addr=%#llx base=%#llx "
            "upper=%#llx loaded=%d match=%d]\n",
            static_cast<unsigned long long>(addr),
            static_cast<unsigned long long>(context.stream_bypass_base),
            static_cast<unsigned long long>(context.stream_bypass_upper),
            context.loaded ? 1 : 0, match ? 1 : 0);
    }
    static uint64_t ranged_probes = 0;
    if (context.stream_bypass_base < context.stream_bypass_upper &&
        ranged_probes++ < limit) {
        std::fprintf(stderr,
            "[ECG-STREAM-RANGED sim=sniper addr=%#llx base=%#llx "
            "upper=%#llx match=%d]\n",
            static_cast<unsigned long long>(addr),
            static_cast<unsigned long long>(context.stream_bypass_base),
            static_cast<unsigned long long>(context.stream_bypass_upper),
            match ? 1 : 0);
    }
    return match;
}

}  // namespace sniper
}  // namespace graphbrew