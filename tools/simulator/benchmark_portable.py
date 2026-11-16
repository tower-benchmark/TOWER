# SPDX-License-Identifier: Apache-2.0
"""Portable entry for the public benchmark."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.dont_write_bytecode = True
__tower_entry_sha256__ = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
import benchmark


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--assets", type=Path, required=True)
    smoke_parser.add_argument("--task", required=True, type=benchmark.task_argument)
    smoke_parser.add_argument("--seeds", type=int, nargs="+", required=True)
    smoke_parser.add_argument("--output", type=Path, required=True)
    smoke_parser.add_argument("--gpu", type=int, default=0)
    smoke_parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    smoke_parser.add_argument(
        "--accept-eula", action="store_true", help="Accept the NVIDIA Omniverse License Agreement"
    )
    prepare = sub.add_parser("prepare-request")
    prepare.add_argument("--assets", type=Path, required=True)
    prepare.add_argument("--task", required=True, type=benchmark.task_argument)
    prepare.add_argument("--seeds", type=int, nargs="+", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--run-output", type=Path, required=True)
    prepare.add_argument("--gpu", type=int, default=0)
    prepare.add_argument("--package", type=Path, required=True)
    prepare.add_argument("--pilot-review", type=Path, required=True)
    prepare.add_argument("--policy-url", required=True, help="XPolicyLab ws:// or wss:// endpoint")
    prepare.add_argument("--ik-python", type=Path, required=True)
    prepare.add_argument("--isaac-python", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--request", type=Path, required=True)
    run.add_argument(
        "--accept-eula", action="store_true", help="Accept the NVIDIA Omniverse License Agreement"
    )
    for command in sub.choices.values():
        command.add_argument("--benchmark", choices=("2026.11.1",), default="2026.11.1")
        command.add_argument("--v1-root", type=Path, help="Explicit v1 archive export")
        command.add_argument("--data-root", type=Path)
        command.add_argument(
            "--reference",
            action="store_true",
            help="Unavailable: the frozen reference runtime is maintained in the private archive",
        )
    from tower_sim.bench.shader_cache import add_arguments

    add_arguments(smoke_parser)
    add_arguments(run)
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments)
    if args.reference:
        benchmark.refuse_reference()
    import benchmark_candidate

    return benchmark_candidate.portable(args)


if __name__ == "__main__":
    raise SystemExit(main())
