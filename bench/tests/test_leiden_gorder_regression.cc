#include <cctype>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "leiden/_iostream.hxx"
#include "reorder/reorder_types.h"
#include "reorder/reorder_gorder.h"

namespace {

using Node = std::int32_t;
using Graph = CSRGraph<Node>;
using Mapping = pvector<Node>;

void Require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

bool HasTimestampShape(const std::string &text) {
  if (text.size() != 19)
    return false;
  for (std::size_t index = 0; index < text.size(); ++index) {
    const char ch = text[index];
    if (index == 4 || index == 7) {
      if (ch != '-')
        return false;
    }
    else if (index == 10) {
      if (ch != ' ')
        return false;
    }
    else if (index == 13 || index == 16) {
      if (ch != ':')
        return false;
    }
    else if (!std::isdigit(static_cast<unsigned char>(ch))) {
      return false;
    }
  }
  return true;
}

Graph MakeEmptyGraph() {
  auto **out_index = new Node *[1];
  auto *out_neighbors = new Node[0];
  auto **in_index = new Node *[1];
  auto *in_neighbors = new Node[0];
  out_index[0] = out_neighbors;
  in_index[0] = in_neighbors;
  return Graph(0, out_index, out_neighbors, in_index, in_neighbors);
}

void TestInt64StreamingIsUnambiguous() {
  const std::int64_t value =
      std::numeric_limits<std::int64_t>::max() - 7;
  std::ostringstream stream;
  stream << value;
  Require(
      stream.str() == std::to_string(value),
      "int64_t stream formatting changed");
}

void TestTimeFormattingRemainsExplicit() {
  const std::time_t timestamp = 0;
  std::ostringstream direct;
  std::ostringstream wrapped;
  writeTime(direct, timestamp);
  wrapped << formattedTime(timestamp);
  Require(
      HasTimestampShape(direct.str()),
      "writeTime no longer emits YYYY-MM-DD HH:MM:SS");
  Require(
      wrapped.str() == direct.str(),
      "formattedTime wrapper diverges from writeTime");
}

void TestGOrderDiagnosticsCompile() {
  Graph graph = MakeEmptyGraph();
  Mapping mapping;
  GenerateGOrderCSRMapping<Node, Node, Node, true>(
      graph, mapping, "unused");
  Require(mapping.size() == 0, "empty graph GOrder CSR mapping changed");
  GenerateGOrderFastMapping<Node, Node, Node, true>(
      graph, mapping, "unused");
  Require(mapping.size() == 0, "empty graph GOrder fast mapping changed");
}

}  // namespace

int main() {
  TestInt64StreamingIsUnambiguous();
  TestTimeFormattingRemainsExplicit();
  TestGOrderDiagnosticsCompile();
  std::cout << "test_leiden_gorder_regression: PASS\n";
  return 0;
}
