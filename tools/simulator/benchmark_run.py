# SPDX-License-Identifier: Apache-2.0
"""Run a task's public benchmark scenes and aggregate complete episodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
import benchmark


def parse_scenes(value):
    try:
        scenes = []
        for part in value.split(","):
            bounds = part.split("-")
            if len(bounds) == 1:
                scenes.append(int(part))
            elif len(bounds) == 2 and int(bounds[0]) <= int(bounds[1]):
                scenes.extend(range(int(bounds[0]), int(bounds[1]) + 1))
            else:
                raise ValueError
        if not scenes or len(set(scenes)) != len(scenes) or any(n < 0 or n > 49 for n in scenes):
            raise ValueError
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected unique scenes 0..49, e.g. 3,7,10-12") from error
    return scenes


def run(args):
    import benchmark_candidate
    from tower_sim.bench.resources import save_exclusive
    from tower_sim.bench.run_summary import aggregate

    benchmark.require_eula(args.accept_eula)
    if not 1 <= args.batch_size <= 10:
        raise ValueError("Batch size must be 1..10")
    args.output.mkdir(parents=True, exist_ok=False)
    batches, failed = [], False
    for start in range(0, len(args.scenes), args.batch_size):
        name = f"batch-{start // args.batch_size:03d}"
        inputs = SimpleNamespace(**vars(args))
        inputs.seeds = args.scenes[start : start + args.batch_size]
        inputs.output = args.output / "requests" / name
        inputs.run_output = args.output / "runs" / name
        inputs.isaac_python = Path(sys.executable)
        path = benchmark_candidate.prepare(inputs, diagnostic=False)
        try:
            code = benchmark_candidate.launch(path, sys.executable, cache_options=args)
        except Exception as error:  # noqa: BLE001 -- retain failed batch for aggregation
            print(f"Batch {name} failed: {error}", file=sys.stderr)
            code = 1
        inputs.run_output.mkdir(parents=True, exist_ok=True)
        save_exclusive(inputs.run_output / "benchmark-request.json", benchmark.read(path))
        batches.append(inputs.run_output)
        if code:
            failed = True
            if not args.keep_going:
                print("Scenes not yet run:", args.scenes[start + args.batch_size :])
                break
    aggregate(batches, args.output)
    return int(failed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("run")
    for name in ("assets", "output", "package", "ik-python"):
        command.add_argument("--" + name, type=Path, required=True)
    command.add_argument("--task", required=True, type=benchmark.task_argument)
    command.add_argument("--scenes", type=parse_scenes, default=parse_scenes("0-49"))
    command.add_argument("--batch-size", type=int, default=1, choices=range(1, 11))
    command.add_argument("--benchmark", choices=("2026.11.1",), default="2026.11.1")
    command.add_argument("--data-root", type=Path)
    command.add_argument("--policy-url", required=True)
    command.add_argument("--pilot-review", type=Path, help="Maintainer posture qualification")
    command.add_argument("--gpu", type=int, default=0)
    command.add_argument("--keep-going", action="store_true")
    command.add_argument("--accept-eula", action="store_true")
    from tower_sim.bench.shader_cache import add_arguments

    add_arguments(command)
    summary = sub.add_parser("aggregate")
    summary.add_argument("--runs", type=Path, nargs="+", required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    from tower_sim.bench.run_summary import aggregate

    aggregate(args.runs, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
