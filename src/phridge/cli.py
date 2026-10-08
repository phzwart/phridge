"""``phridge`` command: validate a config, list plugins, inspect a node, refine."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Dispatch ``refine``, ``validate``, ``plugins``, and ``graph``."""
    parser = argparse.ArgumentParser(prog="phridge")
    sub = parser.add_subparsers(dest="command", required=True)

    refine = sub.add_parser("refine", help="run a policy on a config")
    refine.add_argument("config")
    refine.add_argument("--policy", default="stock_macrocycle")
    refine.add_argument("--out", default="phridge_out")

    validate = sub.add_parser("validate", help="load and build a config")
    validate.add_argument("config")

    plugins = sub.add_parser("plugins")
    plugins.add_argument("action", choices=["list"])

    graph = sub.add_parser("graph", help="print stale, cost, and readers for one node")
    graph.add_argument("config")
    graph.add_argument("--node", required=True)

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "refine":
        _refine(args.config, args.policy, args.out)
    elif args.command == "validate":
        _validate(args.config)
    elif args.command == "plugins":
        _plugins()
    else:
        _graph(args.config, args.node)


def _refine(config: str, policy_name: str, out: str) -> None:
    from phridge.api import Refinement

    job = Refinement.from_config(config)
    job.run(_policy(policy_name))
    job.finalize(out)


def _validate(config: str) -> None:
    from phridge.core.config import build, load

    build(load(config))
    print(f"ok {config}")


def _plugins() -> None:
    from phridge.core.config import _plugin_targets

    for name, target in _plugin_targets():
        print(f"{name} {target}")


def _graph(config: str, node: str) -> None:
    from phridge.core.config import build, load

    _registry, graph, _partitions, _bound = build(load(config))
    if node not in graph.nodes:
        raise SystemExit(f"unknown node {node}")
    print(f"stale {graph.stale(node)}")
    print(f"cost {graph.cost(node)}")
    print(f"readers {sorted(graph.readers(node))}")


def _policy(name: str) -> Any:
    from phridge.policies.calibrate_intensity import CalibrateIntensity
    from phridge.policies.stock_macrocycle import StockMacrocycle

    if name == "stock_macrocycle":
        return StockMacrocycle()
    if name == "calibrate_intensity":
        return CalibrateIntensity()
    raise SystemExit(f"unknown policy {name}")


if __name__ == "__main__":
    main(sys.argv[1:])
