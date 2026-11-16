"""Run a full learned-policy episode to collect evidence before pilot qualification.

Uses the frozen campaign scene, controller, cameras and scorer. Results remain
diagnostic: neither a successful score nor this command approves a pilot.
"""

from __future__ import annotations

import argparse
import sys

sys.dont_write_bytecode = True
from pathlib import Path

import benchmark


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("assets", "output", "package", "ik-python"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--benchmark", choices=("2026.11.1",), default="2026.11.1")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--v1-root", type=Path, help="Explicit v1 archive export")
    parser.add_argument("--task", required=True, type=benchmark.task_argument)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--policy-url", required=True, help="XPolicyLab ws:// or wss:// endpoint")
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Unavailable: the frozen reference runtime is maintained in the private archive",
    )
    parser.add_argument(
        "--accept-eula", action="store_true", help="Accept the NVIDIA Omniverse License Agreement"
    )
    from tower_sim.bench.shader_cache import add_arguments

    add_arguments(parser)
    arguments = sys.argv[1:] if argv is None else argv
    options = parser.parse_args(arguments)
    if options.reference:
        benchmark.refuse_reference()
    import benchmark_candidate

    return benchmark_candidate.diagnostic(options)


if __name__ == "__main__":
    raise SystemExit(main())
