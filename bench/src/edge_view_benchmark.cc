#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graphbrew/edge/edge_stream.h"

int main(int argc, char **argv) {
  CLApp cli(argc, argv, "edge-view conversion benchmark");
  if (!cli.ParseArgs())
    return -1;

  Builder builder(cli);
  Graph graph = builder.MakeGraph();

  Timer outgoing_timer;
  outgoing_timer.Start();
  FlatGraph outgoing = graph.flattenGraphOut();
  outgoing_timer.Stop();

  Timer incoming_timer;
  incoming_timer.Start();
  FlatGraph incoming = graphbrew::edge::FlattenIncoming(graph);
  incoming_timer.Stop();

  if (outgoing.num_nodes() != incoming.num_nodes() ||
      outgoing.num_edges() != incoming.num_edges()) {
    throw std::runtime_error("edge-view conversion changed graph shape");
  }

  PrintTime(
      "Outgoing Edge View Build Time", outgoing_timer.Seconds());
  PrintTime(
      "Incoming Edge View Build Time", incoming_timer.Seconds());
  std::cout << std::setprecision(12)
            << "[EDGE_VIEW_BUILD] vertices="
            << static_cast<long long>(outgoing.num_nodes())
            << " directed_edges="
            << static_cast<long long>(outgoing.num_edges())
            << " outgoing_seconds=" << outgoing_timer.Seconds()
            << " incoming_seconds=" << incoming_timer.Seconds()
            << std::endl;
  return 0;
}
