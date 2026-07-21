#ifndef GRAPHBREW_EDGE_FRONTIER_H_
#define GRAPHBREW_EDGE_FRONTIER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace graphbrew::edge {

template <typename Node>
class FrontierBuilder;

template <typename Node>
class Frontier {
 public:
  explicit Frontier(const std::size_t vertices)
      : vertices_(vertices),
        dense_((vertices + kBitsPerWord - 1) / kBitsPerWord, 0) {}

  void Clear() {
    std::fill(dense_.begin(), dense_.end(), std::uint64_t{0});
    sparse_.clear();
  }

  void AssignSingleton(const Node vertex) {
    Clear();
    Push(vertex);
  }

  void Assign(std::vector<Node> vertices) {
    std::sort(vertices.begin(), vertices.end());
    vertices.erase(
        std::unique(vertices.begin(), vertices.end()),
        vertices.end());
    AssignSortedUnique(std::move(vertices));
  }

  void Push(const Node vertex) {
    // Use FrontierBuilder when producers run concurrently.
    Check(vertex);
    if (!Contains(vertex)) {
      SetDense(vertex);
      sparse_.push_back(vertex);
    }
  }

  bool Contains(const Node vertex) const {
    Check(vertex);
    const std::size_t index = static_cast<std::size_t>(vertex);
    return (
        dense_[index / kBitsPerWord] >>
        (index % kBitsPerWord)) & 1u;
  }

  bool empty() const { return sparse_.empty(); }
  std::size_t size() const { return sparse_.size(); }
  std::size_t capacity() const { return vertices_; }
  const std::vector<Node> &sparse() const { return sparse_; }

 private:
  friend class FrontierBuilder<Node>;

  void AssignSortedUnique(std::vector<Node> vertices) {
    Clear();
    for (const Node vertex : vertices)
      SetDense(vertex);
    sparse_ = std::move(vertices);
  }

  void Check(const Node vertex) const {
    if (vertex < 0 ||
        static_cast<std::size_t>(vertex) >= vertices_) {
      throw std::out_of_range("frontier vertex out of range");
    }
  }

  void SetDense(const Node vertex) {
    Check(vertex);
    const std::size_t index = static_cast<std::size_t>(vertex);
    dense_[index / kBitsPerWord] |=
        std::uint64_t{1} << (index % kBitsPerWord);
  }

  static constexpr std::size_t kBitsPerWord = 64;
  std::size_t vertices_;
  std::vector<std::uint64_t> dense_;
  std::vector<Node> sparse_;
};

template <typename Node>
class FrontierBuilder {
 public:
  explicit FrontierBuilder(const std::size_t vertices)
      : vertices_(vertices),
        seen_((vertices + kBitsPerWord - 1) / kBitsPerWord, 0),
        local_(ThreadCount()) {}

  void PrepareForParallel() {
    if (!overflow_.empty() ||
        std::any_of(
            local_.begin(), local_.end(),
            [](const auto &items) { return !items.empty(); })) {
      throw std::logic_error(
          "frontier builder has unfinished parallel output");
    }
    const std::size_t threads = ThreadCount();
    if (threads > local_.size())
      local_.resize(threads);
  }

  bool Push(const Node vertex) {
    Check(vertex);
    const std::size_t index = static_cast<std::size_t>(vertex);
    const std::size_t word = index / kBitsPerWord;
    const std::uint64_t mask =
        std::uint64_t{1} << (index % kBitsPerWord);
#ifdef _OPENMP
    const std::uint64_t previous = __atomic_fetch_or(
        &seen_[word], mask, __ATOMIC_RELAXED);
#else
    const std::uint64_t previous = seen_[word];
    seen_[word] |= mask;
#endif
    if ((previous & mask) != 0)
      return false;
    const std::size_t thread = ThreadId();
    if (thread < local_.size()) {
      local_[thread].push_back(vertex);
    } else {
#ifdef _OPENMP
#pragma omp critical(graphbrew_frontier_builder_overflow)
#endif
      overflow_.push_back(vertex);
    }
    return true;
  }

  Frontier<Node> Finish() {
    std::vector<Node> merged;
    std::size_t total = 0;
    for (const auto &items : local_)
      total += items.size();
    total += overflow_.size();
    merged.reserve(total);
    for (const auto &items : local_)
      merged.insert(merged.end(), items.begin(), items.end());
    merged.insert(
        merged.end(), overflow_.begin(), overflow_.end());

    std::size_t log_active = 1;
    for (std::size_t value = total; value > 1; value >>= 1)
      ++log_active;
    if (total == 0 || total <= vertices_ / log_active) {
      std::sort(merged.begin(), merged.end());
    } else {
      merged.clear();
      merged.reserve(total);
      for (std::size_t vertex = 0; vertex < vertices_; ++vertex) {
        const std::uint64_t mask =
            std::uint64_t{1} << (vertex % kBitsPerWord);
        if ((seen_[vertex / kBitsPerWord] & mask) != 0)
          merged.push_back(static_cast<Node>(vertex));
      }
    }

    Frontier<Node> frontier(vertices_);
    frontier.AssignSortedUnique(std::move(merged));
    for (const Node vertex : frontier.sparse_) {
      const std::size_t index = static_cast<std::size_t>(vertex);
      seen_[index / kBitsPerWord] &=
          ~(std::uint64_t{1} << (index % kBitsPerWord));
    }
    for (auto &items : local_)
      items.clear();
    overflow_.clear();
    return frontier;
  }

 private:
  void Check(const Node vertex) const {
    if (vertex < 0 ||
        static_cast<std::size_t>(vertex) >= vertices_) {
      throw std::out_of_range("frontier vertex out of range");
    }
  }

  static std::size_t ThreadCount() {
#ifdef _OPENMP
    return static_cast<std::size_t>(omp_get_max_threads());
#else
    return 1;
#endif
  }

  static std::size_t ThreadId() {
#ifdef _OPENMP
    return static_cast<std::size_t>(omp_get_thread_num());
#else
    return 0;
#endif
  }

  static constexpr std::size_t kBitsPerWord = 64;
  std::size_t vertices_;
  std::vector<std::uint64_t> seen_;
  std::vector<std::vector<Node>> local_;
  std::vector<Node> overflow_;
};

}  // namespace graphbrew::edge

#endif  // GRAPHBREW_EDGE_FRONTIER_H_
