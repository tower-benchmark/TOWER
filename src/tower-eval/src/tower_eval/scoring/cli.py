# SPDX-License-Identifier: Apache-2.0
"""Write a versioned metric report without modifying input evidence."""

import argparse
import json
from pathlib import Path

from ..tasks import benchmark_legacy_id
from .api import INPUT_PATHS, score_run
from .rulesets import RULESETS


def task_argument(value):
    try:
        return benchmark_legacy_id(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def add_arguments(parser, *, positional=True):
    parser.add_argument(
        "run" if positional else "--run", type=Path, **({} if positional else {"required": True})
    )
    parser.add_argument("--task", required=True, type=task_argument)
    parser.add_argument(
        "--output", required=True, type=Path, help="New JSON file outside the input recording"
    )
    parser.add_argument("--input-path", choices=INPUT_PATHS, default="files")
    parser.add_argument(
        "--ruleset", choices=tuple(RULESETS), help="Default: task registry scorer_ruleset"
    )


def run(args, parser):
    if args.output.resolve().is_relative_to(args.run.resolve()) or args.output.exists():
        parser.error("Output must be a new file outside the input recording")
    report = score_run(args.run, args.task, input_path=args.input_path, ruleset=args.ruleset)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return int(not report.get("valid"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    raise SystemExit(main())
