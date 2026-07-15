#ifndef GRAPHBREW_PARTITION_SG_MMAP_VIEW_H_
#define GRAPHBREW_PARTITION_SG_MMAP_VIEW_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace graphbrew
{
namespace partition
{

// Read-only, zero-copy view over an unweighted GAP serialized graph (`.sg`).
//
// The on-disk layout is produced by WriterBase::WriteSerializedGraph and
// consumed by Reader::ReadSerializedGraph. For an unweighted graph it is, in
// native byte order and with no padding between regions:
//
//   [0]                     directed        : bool   (1 byte)
//   [1]                     num_edges       : int64  (num_edges_directed)
//   [9]                     num_nodes       : int64
//   [17]                    out_offsets     : int64 * (num_nodes + 1)
//   ...                     out_neighbors   : int32 * num_edges
//   (if directed) in_offsets               : int64 * (num_nodes + 1)
//   (if directed) in_neighbors             : int32 * num_edges
//   ...                     org_ids         : int32 * num_nodes
//
// Because the header is 17 bytes, every subsequent region is misaligned with
// respect to its element type. All field access therefore goes through
// std::memcpy, which is well-defined for unaligned source addresses and
// compiles to a plain load on x86. The view never materialises the full source
// CSR on the heap; the only O(N) allocation is an aligned copy of `org_ids`
// exposed through get_org_ids() (matching Reader/CSRGraph semantics).
template <typename NodeID_ = std::int32_t,
          typename SGOffset_ = std::int64_t>
class SerializedGraphView
{
    static_assert(std::is_integral<NodeID_>::value,
                  "SerializedGraphView requires an integral vertex id");
    static_assert(std::is_integral<SGOffset_>::value,
                  "SerializedGraphView requires an integral offset type");

public:
    using NodeID = NodeID_;
    using SGOffset = SGOffset_;

    class Neighborhood
    {
    public:
        class const_iterator
        {
        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = NodeID;
            using difference_type = std::ptrdiff_t;
            using pointer = void;
            using reference = NodeID;

            const_iterator(const std::uint8_t *neighbors, std::int64_t index)
                : neighbors_(neighbors), index_(index) {}

            NodeID operator*() const
            {
                NodeID value = 0;
                std::memcpy(
                    &value,
                    neighbors_ +
                        static_cast<std::size_t>(index_) * sizeof(NodeID),
                    sizeof(NodeID));
                return value;
            }

            const_iterator &operator++()
            {
                ++index_;
                return *this;
            }

            bool operator==(const const_iterator &other) const
            {
                return index_ == other.index_ &&
                       neighbors_ == other.neighbors_;
            }

            bool operator!=(const const_iterator &other) const
            {
                return !(*this == other);
            }

        private:
            const std::uint8_t *neighbors_;
            std::int64_t index_;
        };

        Neighborhood(const std::uint8_t *neighbors,
                     std::int64_t begin,
                     std::int64_t end)
            : neighbors_(neighbors), begin_(begin), end_(end) {}

        const_iterator begin() const
        {
            return const_iterator(neighbors_, begin_);
        }

        const_iterator end() const
        {
            return const_iterator(neighbors_, end_);
        }

        std::size_t size() const
        {
            return static_cast<std::size_t>(end_ - begin_);
        }

    private:
        const std::uint8_t *neighbors_;
        std::int64_t begin_;
        std::int64_t end_;
    };

    explicit SerializedGraphView(const std::string &path)
    {
        Open(path);
        try
        {
            Parse();
        }
        catch (...)
        {
            Release();
            throw;
        }
    }

    ~SerializedGraphView()
    {
        Release();
    }

    SerializedGraphView(const SerializedGraphView &) = delete;
    SerializedGraphView &operator=(const SerializedGraphView &) = delete;

    SerializedGraphView(SerializedGraphView &&other) noexcept
    {
        MoveFrom(std::move(other));
    }

    SerializedGraphView &operator=(SerializedGraphView &&other) noexcept
    {
        if (this != &other)
        {
            Release();
            MoveFrom(std::move(other));
        }
        return *this;
    }

    std::int64_t num_nodes() const
    {
        return num_nodes_;
    }

    // Directed edge count as stored in the file (== CSRGraph::num_edges_directed).
    std::int64_t num_edges_directed() const
    {
        return num_edges_directed_;
    }

    // Undirected graphs store each edge twice; expose the halved count to match
    // CSRGraph::num_edges().
    std::int64_t num_edges() const
    {
        return directed_ ? num_edges_directed_ : num_edges_directed_ / 2;
    }

    bool directed() const
    {
        return directed_;
    }

    bool is_weighted() const
    {
        return false;
    }

    std::int64_t out_degree(NodeID vertex) const
    {
        const std::int64_t v = CheckVertex(vertex);
        return OutOffset(v + 1) - OutOffset(v);
    }

    std::int64_t in_degree(NodeID vertex) const
    {
        const std::int64_t v = CheckVertex(vertex);
        if (!directed_)
            return OutOffset(v + 1) - OutOffset(v);
        return InOffset(v + 1) - InOffset(v);
    }

    Neighborhood out_neigh(NodeID vertex) const
    {
        const std::int64_t v = CheckVertex(vertex);
        return Neighborhood(out_neighbors_, OutOffset(v), OutOffset(v + 1));
    }

    // Symmetric graphs alias their incoming adjacency onto the outgoing arrays,
    // exactly as CSRGraph does when the reader shares the index/neighbor state.
    Neighborhood in_neigh(NodeID vertex) const
    {
        const std::int64_t v = CheckVertex(vertex);
        if (!directed_)
            return Neighborhood(out_neighbors_, OutOffset(v), OutOffset(v + 1));
        return Neighborhood(in_neighbors_, InOffset(v), InOffset(v + 1));
    }

    NodeID *get_org_ids() const
    {
        return const_cast<NodeID *>(org_ids_.data());
    }

private:
    void Open(const std::string &path)
    {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0)
            throw std::runtime_error("Cannot open serialized graph: " + path);
        struct stat status;
        if (::fstat(fd_, &status) != 0)
        {
            const int saved = fd_;
            fd_ = -1;
            ::close(saved);
            throw std::runtime_error(
                "Cannot stat serialized graph: " + path);
        }
        if (!S_ISREG(status.st_mode) || status.st_size <= 0)
        {
            const int saved = fd_;
            fd_ = -1;
            ::close(saved);
            throw std::runtime_error(
                "Serialized graph is not a regular non-empty file: " + path);
        }
        file_size_ = static_cast<std::size_t>(status.st_size);
        void *mapping = ::mmap(
            nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapping == MAP_FAILED)
        {
            const int saved = fd_;
            fd_ = -1;
            ::close(saved);
            throw std::runtime_error(
                "Cannot mmap serialized graph: " + path);
        }
        base_ = static_cast<const std::uint8_t *>(mapping);
    }

    void Release()
    {
        if (base_ != nullptr)
        {
            ::munmap(const_cast<std::uint8_t *>(base_), file_size_);
            base_ = nullptr;
        }
        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
        file_size_ = 0;
    }

    void MoveFrom(SerializedGraphView &&other) noexcept
    {
        fd_ = other.fd_;
        base_ = other.base_;
        file_size_ = other.file_size_;
        directed_ = other.directed_;
        num_nodes_ = other.num_nodes_;
        num_edges_directed_ = other.num_edges_directed_;
        out_offsets_ = other.out_offsets_;
        out_neighbors_ = other.out_neighbors_;
        in_offsets_ = other.in_offsets_;
        in_neighbors_ = other.in_neighbors_;
        org_ids_ = std::move(other.org_ids_);
        other.fd_ = -1;
        other.base_ = nullptr;
        other.file_size_ = 0;
        other.out_offsets_ = nullptr;
        other.out_neighbors_ = nullptr;
        other.in_offsets_ = nullptr;
        other.in_neighbors_ = nullptr;
    }

    template <typename T>
    T Load(std::size_t byte_offset) const
    {
        T value = 0;
        std::memcpy(&value, base_ + byte_offset, sizeof(T));
        return value;
    }

    static std::size_t CheckedMul(std::size_t lhs, std::size_t rhs)
    {
        if (lhs != 0 &&
            rhs > std::numeric_limits<std::size_t>::max() / lhs)
        {
            throw std::overflow_error(
                "Serialized graph region size overflow");
        }
        return lhs * rhs;
    }

    static std::size_t CheckedAdd(std::size_t lhs, std::size_t rhs)
    {
        if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
            throw std::overflow_error(
                "Serialized graph layout size overflow");
        return lhs + rhs;
    }

    void Parse()
    {
        const std::size_t header_bytes =
            sizeof(bool) + sizeof(SGOffset) + sizeof(SGOffset);
        if (file_size_ < header_bytes)
            throw std::runtime_error(
                "Serialized graph is smaller than its header");

        bool directed = false;
        std::memcpy(&directed, base_, sizeof(bool));
        directed_ = directed;
        const SGOffset raw_edges = Load<SGOffset>(sizeof(bool));
        const SGOffset raw_nodes =
            Load<SGOffset>(sizeof(bool) + sizeof(SGOffset));
        if (raw_nodes < 0 || raw_edges < 0)
            throw std::runtime_error(
                "Serialized graph has a negative node or edge count");
        if (static_cast<std::uint64_t>(raw_nodes) >
            static_cast<std::uint64_t>(
                std::numeric_limits<NodeID>::max()))
        {
            throw std::overflow_error(
                "Serialized graph node count exceeds the vertex id type");
        }
        num_nodes_ = raw_nodes;
        num_edges_directed_ = raw_edges;

        const std::size_t nodes = static_cast<std::size_t>(raw_nodes);
        const std::size_t edges = static_cast<std::size_t>(raw_edges);
        const std::size_t index_bytes =
            CheckedMul(CheckedAdd(nodes, 1), sizeof(SGOffset));
        const std::size_t neigh_bytes = CheckedMul(edges, sizeof(NodeID));
        const std::size_t ids_bytes = CheckedMul(nodes, sizeof(NodeID));

        std::size_t offset = header_bytes;
        const std::size_t out_offsets_at = offset;
        offset = CheckedAdd(offset, index_bytes);
        const std::size_t out_neighbors_at = offset;
        offset = CheckedAdd(offset, neigh_bytes);
        std::size_t in_offsets_at = 0;
        std::size_t in_neighbors_at = 0;
        if (directed_)
        {
            in_offsets_at = offset;
            offset = CheckedAdd(offset, index_bytes);
            in_neighbors_at = offset;
            offset = CheckedAdd(offset, neigh_bytes);
        }
        const std::size_t org_ids_at = offset;
        offset = CheckedAdd(offset, ids_bytes);

        if (offset != file_size_)
            throw std::runtime_error(
                "Serialized graph size does not match its declared layout");

        out_offsets_ = base_ + out_offsets_at;
        out_neighbors_ = base_ + out_neighbors_at;
        if (directed_)
        {
            in_offsets_ = base_ + in_offsets_at;
            in_neighbors_ = base_ + in_neighbors_at;
        }

        ValidateOffsets(out_offsets_, raw_edges);
        if (directed_)
            ValidateOffsets(in_offsets_, raw_edges);

        CopyOrgIds(org_ids_at, nodes);
    }

    void ValidateOffsets(const std::uint8_t *offsets, SGOffset edges) const
    {
        SGOffset previous = 0;
        std::memcpy(&previous, offsets, sizeof(SGOffset));
        if (previous != 0)
            throw std::runtime_error(
                "Serialized graph CSR must begin at offset zero");
        for (std::int64_t v = 1; v <= num_nodes_; ++v)
        {
            SGOffset current = 0;
            std::memcpy(
                &current,
                offsets + static_cast<std::size_t>(v) * sizeof(SGOffset),
                sizeof(SGOffset));
            if (current < previous)
                throw std::runtime_error(
                    "Serialized graph CSR offsets are not monotonic");
            previous = current;
        }
        if (previous != edges)
            throw std::runtime_error(
                "Serialized graph CSR edge count is inconsistent");
    }

    void CopyOrgIds(std::size_t byte_offset, std::size_t nodes)
    {
        org_ids_.resize(nodes);
        if (nodes != 0)
        {
            std::memcpy(
                org_ids_.data(), base_ + byte_offset, nodes * sizeof(NodeID));
        }
    }

    std::int64_t CheckVertex(NodeID vertex) const
    {
        if constexpr (std::is_signed<NodeID>::value)
        {
            if (vertex < 0)
                throw std::out_of_range(
                    "Serialized graph vertex is negative");
        }
        const std::int64_t v = static_cast<std::int64_t>(vertex);
        if (v >= num_nodes_)
            throw std::out_of_range(
                "Serialized graph vertex is out of range");
        return v;
    }

    std::int64_t OutOffset(std::int64_t vertex) const
    {
        return static_cast<std::int64_t>(Load<SGOffset>(
            OutOffsetsByte(vertex)));
    }

    std::int64_t InOffset(std::int64_t vertex) const
    {
        SGOffset value = 0;
        std::memcpy(
            &value,
            in_offsets_ +
                static_cast<std::size_t>(vertex) * sizeof(SGOffset),
            sizeof(SGOffset));
        return static_cast<std::int64_t>(value);
    }

    std::size_t OutOffsetsByte(std::int64_t vertex) const
    {
        return static_cast<std::size_t>(
            out_offsets_ - base_) +
            static_cast<std::size_t>(vertex) * sizeof(SGOffset);
    }

    int fd_ = -1;
    const std::uint8_t *base_ = nullptr;
    std::size_t file_size_ = 0;

    bool directed_ = false;
    std::int64_t num_nodes_ = 0;
    std::int64_t num_edges_directed_ = 0;

    const std::uint8_t *out_offsets_ = nullptr;
    const std::uint8_t *out_neighbors_ = nullptr;
    const std::uint8_t *in_offsets_ = nullptr;
    const std::uint8_t *in_neighbors_ = nullptr;
    std::vector<NodeID> org_ids_;
};

} // namespace partition
} // namespace graphbrew

#endif // GRAPHBREW_PARTITION_SG_MMAP_VIEW_H_
