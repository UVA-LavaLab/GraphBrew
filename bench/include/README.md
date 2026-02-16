# Include Layout

```
bench/include/
├── graphbrew/              # GraphBrew extensions
│   ├── graphbrew.h         # Umbrella header (includes everything)
│   ├── reorder/            # Reordering algorithms
│   │   ├── reorder.h       # Main reorder dispatcher
│   │   ├── reorder_types.h # Type definitions
│   │   ├── reorder_basic.h # Original, Random, Sort
│   │   ├── reorder_hub.h   # HubSort, HubCluster, DBG variants
│   │   ├── reorder_classic.h # COrder (10)
│   │   ├── reorder_rabbit.h  # RabbitOrder variants (8:csr, 8:boost)
│   │   ├── reorder_gorder.h  # GOrder variants (9:default, 9:csr, 9:fast)
│   │   ├── reorder_rcm.h     # RCM variants (11:default, 11:bnf)
│   │   ├── reorder_graphbrew.h # GraphBrew unified reordering framework
│   │   └── reorder_adaptive.h # Adaptive ML-based selection
│   └── partition/          # Partitioning algorithms
│       ├── trust.h         # TRUST partitioning
│       └── cagra/popt.h    # Cagra/GraphIT P-OPT helpers
├── external/               # External libraries (bundled)
│   ├── gapbs/              # Core GAPBS runtime
│   │   ├── builder.h       # Graph construction & reorder dispatch
│   │   ├── graph.h         # CSRGraph implementation
│   │   ├── benchmark.h     # Benchmarking utilities
│   │   ├── command_line.h  # CLI parsing
│   │   └── ...             # Other core headers
│   ├── rabbit/             # RabbitOrder (community detection)
│   ├── gorder/             # GOrder (graph ordering)
│   ├── corder/             # COrder (cache-aware ordering)
│   └── leiden/             # GVE-Leiden (community detection)
└── cache_sim/              # Cache simulation
    ├── cache_sim.h         # Cache hierarchy simulation
    └── graph_sim.h         # Graph-specific cache tracking
```

## Include Paths (Makefile)

```makefile
INCLUDE_GAPBS     = $(INC_DIR)/external/gapbs   # Core runtime
INCLUDE_GRAPHBREW = $(INC_DIR)/graphbrew        # GraphBrew extensions
INCLUDE_EXTERNAL  = $(INC_DIR)/external         # All external libs
INCLUDE_CACHE     = $(INC_DIR)/cache_sim        # Cache simulation
```

## Usage

Source files use bare includes resolved via `-I` flags:
```cpp
#include "builder.h"      // From external/gapbs/
#include "graph.h"        // From external/gapbs/
#include <rabbit/rabbit_order.hpp>  // From external/rabbit/
#include <gorder/GoGraph.h>         // From external/gorder/
```
