#ifndef GRAPHBREW_REORDER_EXPERIMENTAL_COMMON_H_
#define GRAPHBREW_REORDER_EXPERIMENTAL_COMMON_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace graphbrew::experimental {

inline size_t checkedSizeAdd(
    size_t left,
    size_t right,
    const char* message)
{
    if (right > std::numeric_limits<size_t>::max() - left)
        throw std::overflow_error(message);
    return left + right;
}

inline size_t checkedSizeMultiply(
    size_t left,
    size_t right,
    const char* message)
{
    if (
        left != 0
        && right > std::numeric_limits<size_t>::max() / left
    ) {
        throw std::overflow_error(message);
    }
    return left * right;
}

inline uint64_t checkedUint64Add(
    uint64_t left,
    uint64_t right,
    const char* message)
{
    if (right > std::numeric_limits<uint64_t>::max() - left)
        throw std::overflow_error(message);
    return left + right;
}

inline uint64_t checkedUint64Multiply(
    uint64_t left,
    uint64_t right,
    const char* message)
{
    if (
        left != 0
        && right > std::numeric_limits<uint64_t>::max() / left
    ) {
        throw std::overflow_error(message);
    }
    return left * right;
}

template <typename K>
size_t checkedIndex(K value, size_t size, const char* message)
{
    static_assert(std::is_integral<K>::value);
    if constexpr (std::is_signed<K>::value) {
        if (value < 0) throw std::invalid_argument(message);
    }
    const uintmax_t converted = static_cast<uintmax_t>(value);
    if (
        converted >= static_cast<uintmax_t>(size)
        || converted
            > static_cast<uintmax_t>(
                std::numeric_limits<size_t>::max())
    ) {
        throw std::invalid_argument(message);
    }
    return static_cast<size_t>(converted);
}

template <typename K>
std::vector<K> invertOrder(const std::vector<K>& order)
{
    static_assert(std::is_integral<K>::value);
    if (
        !order.empty()
        && static_cast<uintmax_t>(order.size() - 1)
            > static_cast<uintmax_t>(
                std::numeric_limits<K>::max())
    ) {
        throw std::overflow_error(
            "Experimental order exceeds identifier range");
    }

    std::vector<K> inverse(order.size());
    std::vector<char> seen(order.size(), 0);
    for (size_t position = 0; position < order.size(); ++position) {
        const K vertex = order[position];
        const size_t index = checkedIndex(
            vertex,
            order.size(),
            "Experimental order contains an invalid identifier");
        if (seen[index]) {
            throw std::invalid_argument(
                "Experimental order is not a permutation");
        }
        seen[index] = 1;
        inverse[index] = static_cast<K>(position);
    }
    return inverse;
}

}  // namespace graphbrew::experimental

#endif  // GRAPHBREW_REORDER_EXPERIMENTAL_COMMON_H_
