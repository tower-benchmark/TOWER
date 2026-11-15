"""Score complete diagnostic physics without changing or qualifying trial status."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import argparse
import json
from pathlib import Path

import benchmark


def score_physical_diagnostic(run, task, *, reference=False, benchmark_version="2026.11.1", v1_root=None):
    if reference:
        benchmark.refuse_reference()
    from tower_sim.bench.policy_entry import score_diagnostic
    from tower_sim.bench.resources import Resources
    from tower_sim.bench.tasks import BenchTask
    from tower_sim.bench.verification import envelope, verify_data

    verify_data(benchmark.ROOT, benchmark=benchmark_version, v1_root=v1_root)
    resolved = BenchTask.from_resources(task, Resources.from_roots(checkout=benchmark.ROOT, assets=benchmark.ROOT))
    report = score_diagnostic(run, resolved)
    report.pop("prefix_only")  # This public entry always scores the complete benchmark.
    return dict(report, public=envelope(benchmark=benchmark_version))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", action="store_true", help="Unavailable: the frozen reference runtime is maintained in the private archive")
    parser.add_argument("--benchmark", choices=("2026.11.1",), default="2026.11.1")
    parser.add_argument("--v1-root", type=Path, help="Explicit v1 archive export")
    args = parser.parse_args()
    if args.reference:
        benchmark.refuse_reference()
    if args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError("Write analysis outside the original recording")
    report = score_physical_diagnostic(args.run, args.task, reference=args.reference,
                                      benchmark_version=args.benchmark, v1_root=args.v1_root)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(task=args.task, diagnostic_only=True, scores=report["scores"],
                         simulated_seconds=report["duration_s"])))


if __name__ == "__main__":
    main()
