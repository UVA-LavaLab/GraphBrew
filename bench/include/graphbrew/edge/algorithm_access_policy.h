#ifndef GRAPHBREW_EDGE_ALGORITHM_ACCESS_POLICY_H_
#define GRAPHBREW_EDGE_ALGORITHM_ACCESS_POLICY_H_

#include <cstddef>

namespace graphbrew::edge {

enum class AccessKind {
  kVertexRead,
  kVertexWrite,
  kEdgeRead,
  kFrontierRead,
  kFrontierWrite,
};

struct NoOpAccessPolicy {
  template <typename Node>
  void OnVertex(Node, AccessKind) const {}

  template <typename EdgeT>
  void OnEdge(const EdgeT &) const {}

  void OnBarrier(std::size_t) const {}
};

}  // namespace graphbrew::edge

#endif  // GRAPHBREW_EDGE_ALGORITHM_ACCESS_POLICY_H_
