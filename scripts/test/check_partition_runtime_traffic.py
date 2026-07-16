#!/usr/bin/env python3
"""Exercise partition traffic counters through the real bfs_p traversal."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def write_graph(path: Path) -> None:
    edges = [
        (0, 1),
        (0, 2),
        (1, 32),
        (2, 32),
    ]
    edges.extend((32, vertex) for vertex in range(33, 40))
    edges.extend((32, vertex) for vertex in range(3, 16))
    for source in range(40, 56):
        for delta in (1, 2, 3, 4):
            edges.append(
                (source, 40 + ((source - 40 + delta) % 16))
            )
    path.write_text(
        "".join(f"{source} {target}\n" for source, target in edges)
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfs", type=Path, required=True)
    args = parser.parse_args()
    bfs = args.bfs.resolve()
    if not bfs.is_file():
        raise RuntimeError(f"missing bfs_p binary: {bfs}")

    with tempfile.TemporaryDirectory(
        prefix="graphbrew-runtime-traffic-"
    ) as raw_temp:
        root = Path(raw_temp)
        graph = root / "traffic.el"
        database = root / "database"
        database.mkdir()
        write_graph(graph)
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = "4"
        completed = subprocess.run(
            [
                str(bfs),
                "-f",
                str(graph),
                "-S",
                "-P",
                "2",
                "-B",
                "vertices",
                "-n",
                "1",
                "-r",
                "0",
                "-v",
                "-D",
                str(database),
                "-a",
            ],
            check=True,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        require(
            "Verification:" in completed.stdout
            and "PASS" in completed.stdout,
            "partition traffic BFS verification failed",
        )
        reports = json.loads(
            (database / "benchmarks.json").read_text())
        traffic = (
            reports[-1]["trial_details"][0]["answer"]["partition"]
            ["runtime_traffic"]
        )

    require(
        traffic["schema"] == "graphbrew.partition_runtime_traffic.v1",
        "runtime traffic schema mismatch",
    )
    require(traffic["ghost_slots"] == 16, "ghost-slot count mismatch")
    steps = traffic["bfs"]["steps"]
    require(
        [step["phase"] for step in steps]
        == ["p-bsp-td", "p-bsp-td", "p-bsp-bu", "p-bsp-bu"],
        "unexpected direction-optimizing BFS phases",
    )
    require(
        steps[1]["remote_parent_messages"] == 2,
        "top-down loser proposal was not counted",
    )
    require(
        steps[1]["shards"][0]["remote_parent_messages"] == 2,
        "top-down proposal was assigned to the wrong source shard",
    )
    require(
        all(
            step["cpu_ghost_sync_bytes"] == traffic["ghost_slots"]
            for step in steps[2:]
        ),
        "bottom-up ghost synchronization was not fully counted",
    )
    require(
        traffic["bfs"]["remote_parent_bytes"] == 2 * 4,
        "remote-parent payload byte count mismatch",
    )
    require(
        traffic["bfs"]["cpu_ghost_sync_bytes"] == 2 * 16,
        "ghost synchronization total mismatch",
    )
    require(
        traffic["bfs"]["graphblox_halo_bytes"]
        == len(steps) * 16 * 2 * 4,
        "GraphBlox BFS halo projection mismatch",
    )
    print("[success!] partition runtime traffic traversal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
